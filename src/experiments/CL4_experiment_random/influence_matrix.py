from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from huggingface_hub import list_repo_refs
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    default_data_collator,
    set_seed,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

BABYLM_TOKENIZER = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a TICL-style influence matrix from GPT-2 surrogate epoch "
            "checkpoints stored as Hugging Face branches."
        )
    )

    parser.add_argument("--surrogate_repo_id", default="eligran12/babylm_CL4_surrogate")
    parser.add_argument("--surrogate_branches", nargs="+", default=None)
    parser.add_argument("--dataset_name", default="BabyLM-community/BabyLM-2026-Strict-Small")
    parser.add_argument("--tokenizer_name", default=BABYLM_TOKENIZER)
    parser.add_argument("--output_dir", default="./influence_output")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize_mean_gradient",
        action="store_true",
        help=(
            "Normalize the checkpoint-level mean gradient before scoring. "
            "Recommended when you later average columns across checkpoints."
        ),
    )

    return parser.parse_args()


def _epoch_num(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return 10**9


def discover_epoch_branches(repo_id: str) -> list[str]:
    refs = list_repo_refs(repo_id, repo_type="model")
    branches = sorted(
        [ref.name for ref in refs.branches if ref.name.startswith("epoch_")],
        key=_epoch_num,
    )
    if not branches:
        raise ValueError(f"No epoch_* branches found in {repo_id}")
    return branches


def build_token_dataset(
    dataset_name: str,
    tokenizer,
    max_seq_length: int,
    tokenize_batch_size: int,
    seed: int,
    max_examples: int | None = None,
) -> tuple[Dataset, dict]:
    """
    Rebuild the same packed GPT-2 sequence dataset used by the surrogate:
    deterministic raw-document shuffle, EOS after every raw document,
    continuous packing, fixed-length blocks, and only final remainder discarded.

    Matrix rows correspond to packed training sequences, not raw documents.
    """

    raw = load_dataset(dataset_name, split="train", trust_remote_code=True)

    if "text" not in raw.column_names:
        raise ValueError(f"Dataset has no 'text' column: {raw.column_names}")

    if max_examples is not None:
        raw = raw.select(range(min(max_examples, len(raw))))
        logger.warning("DEBUG MODE: using only %s raw examples.", f"{len(raw):,}")

    raw = raw.shuffle(seed=seed)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    stats = {
        "raw_documents": 0,
        "subword_tokens_with_eos": 0,
        "packed_sequences": 0,
        "discarded_final_remainder_tokens": 0,
    }

    def generator() -> Iterator[dict[str, list[int]]]:
        carry: list[int] = []

        for batch in raw.iter(batch_size=tokenize_batch_size):
            encoded = tokenizer(
                batch["text"],
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

            batch_ids: list[int] = []

            for ids in encoded["input_ids"]:
                batch_ids.extend(ids)
                batch_ids.append(eos_id)
                stats["raw_documents"] += 1
                stats["subword_tokens_with_eos"] += len(ids) + 1

            combined = carry + batch_ids
            complete_length = (len(combined) // max_seq_length) * max_seq_length

            for start in range(0, complete_length, max_seq_length):
                block = combined[start : start + max_seq_length]
                stats["packed_sequences"] += 1
                yield {
                    "input_ids": block,
                    "attention_mask": [1] * max_seq_length,
                }

            carry = combined[complete_length:]

        stats["discarded_final_remainder_tokens"] = len(carry)

    dataset = Dataset.from_generator(generator)

    if len(dataset) == 0:
        raise ValueError("No complete packed sequences produced.")

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    logger.info("=" * 76)
    logger.info("RECONSTRUCTED INFLUENCE DATASET")
    logger.info("Raw documents:                %s", f"{stats['raw_documents']:,}")
    logger.info("Subword tokens including EOS: %s", f"{stats['subword_tokens_with_eos']:,}")
    logger.info("Packed sequences:             %s", f"{len(dataset):,}")
    logger.info("Sequence length:              %s", max_seq_length)
    logger.info("Discarded final remainder:    %s", f"{stats['discarded_final_remainder_tokens']:,}")
    logger.info("=" * 76)

    return dataset, stats


def validate_compatibility(
    model: GPT2LMHeadModel,
    tokenizer,
    dataset: Dataset,
    max_seq_length: int,
    checkpoint_label: str,
) -> None:
    model_vocab = model.get_input_embeddings().num_embeddings

    if len(tokenizer) != model_vocab:
        raise ValueError(
            f"{checkpoint_label}: tokenizer length {len(tokenizer)} != model vocab {model_vocab}"
        )

    if max_seq_length > model.config.n_positions:
        raise ValueError(
            f"{checkpoint_label}: max_seq_length={max_seq_length} > n_positions={model.config.n_positions}"
        )

    sample_n = min(1024, len(dataset))
    min_id, max_id = 10**18, -1

    for row in dataset.select(range(sample_n)):
        ids = row["input_ids"]
        min_id = min(min_id, int(ids.min()))
        max_id = max(max_id, int(ids.max()))

    if min_id < 0 or max_id >= model_vocab:
        raise ValueError(
            f"{checkpoint_label}: token ids [{min_id}, {max_id}] outside [0, {model_vocab - 1}]"
        )

    logger.info(
        "%s compatibility OK: vocab=%s, sampled token range=[%s,%s]",
        checkpoint_label,
        f"{model_vocab:,}",
        min_id,
        max_id,
    )


def per_example_input_embedding_gradients(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute one normalized gradient vector per packed training sequence.

    This differentiates with respect to `inputs_embeds`, not directly with
    respect to `model.transformer.wte.weight`.

    This matters for GPT-2 because the input embedding matrix is normally tied
    to the LM head. Gradients on `wte.weight` can include output-softmax/LM-head
    contributions, while `inputs_embeds` isolates the input-side representation.
    """

    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    vectors: list[torch.Tensor] = []

    for i in range(input_ids.shape[0]):
        model.zero_grad(set_to_none=True)

        ids = input_ids[i : i + 1]
        mask = attention_mask[i : i + 1]

        with torch.no_grad():
            embeds = model.transformer.wte(ids)

        embeds = embeds.detach().requires_grad_(True)

        outputs = model(
            inputs_embeds=embeds,
            attention_mask=mask,
            labels=ids,
            use_cache=False,
        )

        grad = torch.autograd.grad(
            outputs.loss,
            embeds,
            retain_graph=False,
            create_graph=False,
        )[0]

        mask_float = mask.unsqueeze(-1).to(grad.dtype)
        denom = mask_float.sum(dim=1).clamp_min(1.0)

        seq_vec = (grad * mask_float).sum(dim=1).squeeze(0) / denom.squeeze(0)
        seq_vec = F.normalize(seq_vec.detach(), p=2, dim=0, eps=1e-12)

        vectors.append(seq_vec.cpu())

    return torch.stack(vectors, dim=0)


def compute_checkpoint_column(
    model: GPT2LMHeadModel,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    label: str,
    normalize_mean_gradient: bool,
) -> tuple[np.ndarray, dict]:
    """
    Let g_i be the normalized input-embedding gradient vector of sequence i.

    Paper-faithful average-gradient influence:
        phi_i = g_i dot mean_j(g_j)

    Optional column-scale-normalized variant:
        phi_i = g_i dot normalize(mean_j(g_j))

    I recommend using --normalize_mean_gradient when your curriculum score is
    phi.mean(axis=1), because it prevents checkpoint columns from being weighted
    by different mean-gradient norms.
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    chunks: list[torch.Tensor] = []
    processed = 0

    model.eval()

    for batch_idx, batch in enumerate(loader):
        grads = per_example_input_embedding_gradients(
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            device=device,
        )

        chunks.append(grads)
        processed += grads.shape[0]

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                "%s: processed %s/%s sequences",
                label,
                f"{processed:,}",
                f"{len(dataset):,}",
            )

    G = torch.cat(chunks, dim=0).float()

    if G.shape[0] != len(dataset):
        raise RuntimeError(f"{label}: collected {G.shape[0]} gradients for {len(dataset)} rows.")

    row_norms = G.norm(dim=1)
    if not torch.isfinite(row_norms).all():
        raise FloatingPointError(f"{label}: non-finite row gradient norm found.")

    mean_grad = G.mean(dim=0)
    mean_grad_norm = float(mean_grad.norm().item())

    if normalize_mean_gradient:
        scoring_grad = F.normalize(mean_grad, p=2, dim=0, eps=1e-12)
        formula = "dot(normalized_example_gradient, normalized_mean_gradient)"
    else:
        scoring_grad = mean_grad
        formula = "dot(normalized_example_gradient, mean_normalized_example_gradient)"

    values = (G @ scoring_grad).numpy().astype(np.float32)

    if not np.isfinite(values).all():
        raise FloatingPointError(f"{label}: non-finite influence values.")

    stats = {
        "checkpoint": label,
        "n_sequences": int(G.shape[0]),
        "hidden_size": int(G.shape[1]),
        "example_gradient_norm_mean": float(row_norms.mean().item()),
        "example_gradient_norm_std": float(row_norms.std(unbiased=False).item()),
        "mean_gradient_norm_before_optional_normalization": mean_grad_norm,
        "normalize_mean_gradient": bool(normalize_mean_gradient),
        "scoring_formula": formula,
        "influence_mean": float(values.mean()),
        "influence_std": float(values.std()),
        "influence_min": float(values.min()),
        "influence_max": float(values.max()),
    }

    return values, stats


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Tokenizer=%s | class=%s | length=%s",
        args.tokenizer_name,
        tokenizer.__class__.__name__,
        f"{len(tokenizer):,}",
    )

    branches = (
        args.surrogate_branches
        if args.surrogate_branches is not None
        else discover_epoch_branches(args.surrogate_repo_id)
    )
    branches = sorted(branches, key=_epoch_num)

    if len(branches) == 0:
        raise ValueError("No surrogate branches provided or discovered.")

    logger.info("Using surrogate branches: %s", branches)

    dataset, dataset_stats = build_token_dataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        tokenize_batch_size=args.tokenize_batch_size,
        seed=args.seed,
        max_examples=args.max_train_examples,
    )

    n_sequences = len(dataset)
    n_checkpoints = len(branches)

    matrix_path = output_dir / "influence_matrix.npy"
    Phi = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_sequences, n_checkpoints),
    )

    per_checkpoint_stats: list[dict] = []

    for column_idx, branch in enumerate(branches):
        logger.info(
            "[%s/%s] Loading surrogate checkpoint: %s @ %s",
            column_idx + 1,
            n_checkpoints,
            args.surrogate_repo_id,
            branch,
        )

        model = GPT2LMHeadModel.from_pretrained(
            args.surrogate_repo_id,
            revision=branch,
        )

        validate_compatibility(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_seq_length=args.max_seq_length,
            checkpoint_label=branch,
        )

        for param in model.parameters():
            param.requires_grad_(False)

        model.config.use_cache = False
        model = model.to(device)
        model.eval()

        values, stats = compute_checkpoint_column(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            label=branch,
            normalize_mean_gradient=args.normalize_mean_gradient,
        )

        Phi[:, column_idx] = values
        Phi.flush()

        per_checkpoint_stats.append(stats)

        logger.info(
            "%s influence: mean=%.6g std=%.6g min=%.6g max=%.6g",
            branch,
            stats["influence_mean"],
            stats["influence_std"],
            stats["influence_min"],
            stats["influence_max"],
        )

        del values, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_step": "ticl_influence_matrix",
        "n_sequences": n_sequences,
        "n_checkpoints": n_checkpoints,
        "surrogate_repo_id": args.surrogate_repo_id,
        "branches_used": branches,
        "dataset": args.dataset_name,
        "tokenizer": args.tokenizer_name,
        "tokenizer_length": len(tokenizer),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_train_examples": args.max_train_examples,
        "matrix_path": str(matrix_path),
        "matrix_shape": [n_sequences, n_checkpoints],
        "dataset_reconstruction": {
            "unit": "packed_fixed_length_training_sequence",
            "raw_document_shuffle": True,
            "shuffle_seed": args.seed,
            "eos_after_each_raw_document": True,
            "continuous_carry_across_tokenization_batches": True,
            "discard_only_final_remainder": True,
            **dataset_stats,
        },
        "gradient_definition": {
            "model_family": "GPT-2 causal LM",
            "loss": "next-token cross-entropy via labels=input_ids",
            "gradient_target": "inputs_embeds",
            "why_inputs_embeds": (
                "Avoids mixing input-embedding gradients with tied LM-head/output "
                "weight gradients from model.transformer.wte.weight."
            ),
            "per_sequence_reduction": "attention-mask-weighted mean over token positions",
            "per_sequence_normalization": "L2 normalization",
            "mean_gradient": "mean of normalized per-sequence gradients across the full packed dataset",
            "normalize_mean_gradient": bool(args.normalize_mean_gradient),
            "influence_score": (
                "dot(normalized per-sequence input-embedding gradient, checkpoint mean gradient)"
            ),
        },
        "per_checkpoint_stats": per_checkpoint_stats,
    }

    metadata_path = output_dir / "influence_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("=" * 76)
    logger.info("INFLUENCE MATRIX COMPLETE")
    logger.info("Matrix:   %s", matrix_path)
    logger.info("Metadata: %s", metadata_path)
    logger.info("Shape:    %s", [n_sequences, n_checkpoints])
    logger.info("=" * 76)


if __name__ == "__main__":
    main()