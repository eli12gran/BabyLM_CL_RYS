from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed

from experiments.cl4_document_level_fixed.cl4_common import (
    BABYLM_DATASET,
    BABYLM_TOKENIZER,
    TRAINING_CONFIG,
    build_tokenizer,
    load_document_corpus,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute document-level CL4/TICL influence matrix from GPT-2 surrogate "
            "epoch checkpoints stored as Hugging Face branches."
        )
    )
    parser.add_argument("--surrogate_repo_id", default="eligran12/babylm_CL4_surrogate")
    parser.add_argument("--surrogate_branches", nargs="+", default=None)
    parser.add_argument("--dataset_name", default=BABYLM_DATASET)
    parser.add_argument("--tokenizer_name", default=BABYLM_TOKENIZER)
    parser.add_argument("--output_dir", default="./influence_output_document_level")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--min_document_words", type=int, default=3)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize_mean_gradient", action="store_true")
    parser.add_argument(
        "--max_chunks_per_document",
        type=int,
        default=None,
        help=(
            "Optional speed/debug cap on chunks per document. Leave unset for the "
            "faithful full-document computation."
        ),
    )
    parser.add_argument("--log_every", type=int, default=500)
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


def validate_model_tokenizer(model: GPT2LMHeadModel, tokenizer, max_seq_length: int, label: str) -> None:
    vocab = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != vocab:
        raise ValueError(f"{label}: tokenizer length {len(tokenizer)} != model vocab {vocab}")
    if max_seq_length > model.config.n_positions:
        raise ValueError(f"{label}: max_seq_length={max_seq_length} > n_positions={model.config.n_positions}")


def chunk_ids(ids: list[int], max_seq_length: int) -> list[list[int]]:
    return [ids[i:i + max_seq_length] for i in range(0, len(ids), max_seq_length)]


def document_gradient_vector(
    model: GPT2LMHeadModel,
    tokenizer,
    text: str,
    device: torch.device,
    max_seq_length: int,
    max_chunks_per_document: int | None,
) -> torch.Tensor | None:
    """
    Return one normalized input-embedding gradient vector for one raw document.

    Long documents are split into max_seq_length chunks. Each chunk gets its own
    input-embedding gradient vector; the document vector is the normalized mean
    of its valid chunk vectors.

    This is document-level influence: one row in Phi == one raw document.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    ids = list(encoded["input_ids"])

    # Append EOS so the document boundary is represented consistently.
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    chunks = [c for c in chunk_ids(ids, max_seq_length) if len(c) >= 2]
    if max_chunks_per_document is not None:
        chunks = chunks[:max_chunks_per_document]

    if not chunks:
        return None

    chunk_vectors: list[torch.Tensor] = []

    for chunk in chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)

        model.zero_grad(set_to_none=True)

        with torch.no_grad():
            embeds = model.transformer.wte(input_ids)

        embeds = embeds.detach().requires_grad_(True)

        outputs = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
        )

        if not torch.isfinite(outputs.loss):
            continue

        grad = torch.autograd.grad(
            outputs.loss,
            embeds,
            retain_graph=False,
            create_graph=False,
        )[0]

        vec = grad.mean(dim=1).squeeze(0)
        vec = F.normalize(vec.detach(), p=2, dim=0, eps=1e-12)
        chunk_vectors.append(vec.cpu())

    if not chunk_vectors:
        return None

    doc_vec = torch.stack(chunk_vectors, dim=0).mean(dim=0)
    doc_vec = F.normalize(doc_vec, p=2, dim=0, eps=1e-12)
    return doc_vec


def compute_checkpoint_column(
    model: GPT2LMHeadModel,
    tokenizer,
    documents,
    device: torch.device,
    label: str,
    max_seq_length: int,
    normalize_mean_gradient: bool,
    max_chunks_per_document: int | None,
    log_every: int,
) -> tuple[np.ndarray, dict]:
    vectors: list[torch.Tensor] = []
    skipped = 0

    model.eval()

    for i, row in enumerate(documents):
        vec = document_gradient_vector(
            model=model,
            tokenizer=tokenizer,
            text=row["text"],
            device=device,
            max_seq_length=max_seq_length,
            max_chunks_per_document=max_chunks_per_document,
        )

        if vec is None:
            skipped += 1
            # Keep matrix row alignment stable. This should be rare after
            # min_document_words filtering. Zero vector means no influence.
            vec = torch.zeros(model.config.n_embd, dtype=torch.float32)

        vectors.append(vec)

        if log_every > 0 and (i + 1) % log_every == 0:
            logger.info(
                "%s: processed %s/%s documents",
                label,
                f"{i + 1:,}",
                f"{len(documents):,}",
            )

    G = torch.stack(vectors, dim=0).float()

    if G.shape[0] != len(documents):
        raise RuntimeError(f"{label}: got {G.shape[0]} vectors for {len(documents)} documents")

    mean_grad = G.mean(dim=0)
    mean_grad_norm = float(mean_grad.norm().item())

    if normalize_mean_gradient:
        scoring_grad = F.normalize(mean_grad, p=2, dim=0, eps=1e-12)
        formula = "dot(normalized_document_gradient, normalized_mean_document_gradient)"
    else:
        scoring_grad = mean_grad
        formula = "dot(normalized_document_gradient, mean_normalized_document_gradient)"

    values = (G @ scoring_grad).numpy().astype(np.float32)

    if not np.isfinite(values).all():
        raise FloatingPointError(f"{label}: non-finite influence values")

    norms = G.norm(dim=1)
    stats = {
        "checkpoint": label,
        "n_documents": int(G.shape[0]),
        "hidden_size": int(G.shape[1]),
        "skipped_documents_replaced_by_zero_vector": int(skipped),
        "document_gradient_norm_mean": float(norms.mean().item()),
        "document_gradient_norm_std": float(norms.std(unbiased=False).item()),
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

    tokenizer = build_tokenizer(args.tokenizer_name)

    branches = args.surrogate_branches or discover_epoch_branches(args.surrogate_repo_id)
    branches = sorted(branches, key=_epoch_num)
    logger.info("Using branches: %s", branches)

    documents, total_words, corpus_stats = load_document_corpus(
        dataset_name=args.dataset_name,
        min_document_words=args.min_document_words,
        max_train_examples=args.max_train_examples,
        logger=logger,
    )

    n_documents = len(documents)
    n_checkpoints = len(branches)

    matrix_path = output_dir / "influence_matrix.npy"
    Phi = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_documents, n_checkpoints),
    )

    document_ids = np.asarray(documents["doc_id"], dtype=np.int64)
    word_counts = np.asarray(documents["word_count"], dtype=np.int64)
    np.save(output_dir / "document_ids.npy", document_ids)
    np.save(output_dir / "document_word_counts.npy", word_counts)

    per_checkpoint_stats = []

    for col, branch in enumerate(branches):
        logger.info("[%s/%s] Loading %s @ %s", col + 1, n_checkpoints, args.surrogate_repo_id, branch)
        model = GPT2LMHeadModel.from_pretrained(args.surrogate_repo_id, revision=branch)
        validate_model_tokenizer(model, tokenizer, args.max_seq_length, branch)

        for p in model.parameters():
            p.requires_grad_(False)

        model.config.use_cache = False
        model = model.to(device)
        model.eval()

        values, stats = compute_checkpoint_column(
            model=model,
            tokenizer=tokenizer,
            documents=documents,
            device=device,
            label=branch,
            max_seq_length=args.max_seq_length,
            normalize_mean_gradient=args.normalize_mean_gradient,
            max_chunks_per_document=args.max_chunks_per_document,
            log_every=args.log_every,
        )

        Phi[:, col] = values
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
        "pipeline_step": "cl4_document_level_influence_matrix",
        "matrix_path": str(matrix_path),
        "matrix_shape": [n_documents, n_checkpoints],
        "n_documents": n_documents,
        "n_checkpoints": n_checkpoints,
        "surrogate_repo_id": args.surrogate_repo_id,
        "branches_used": branches,
        "dataset": args.dataset_name,
        "tokenizer": args.tokenizer_name,
        "tokenizer_length": len(tokenizer),
        "total_words_after_document_filter": total_words,
        "max_seq_length": args.max_seq_length,
        "min_document_words": args.min_document_words,
        "max_train_examples": args.max_train_examples,
        "max_chunks_per_document": args.max_chunks_per_document,
        "seed": args.seed,
        "curriculum_unit": "raw_document",
        "document_ids_path": str(output_dir / "document_ids.npy"),
        "document_word_counts_path": str(output_dir / "document_word_counts.npy"),
        "corpus": corpus_stats,
        "gradient_definition": {
            "loss": "next-token cross-entropy via labels=input_ids",
            "gradient_target": "inputs_embeds",
            "document_representation": (
                "For each raw document, compute normalized input-embedding gradient "
                "vectors for chunks up to max_seq_length, average chunk vectors, "
                "and L2-normalize the document vector."
            ),
            "mean_gradient": "mean of normalized document vectors over all documents",
            "normalize_mean_gradient": bool(args.normalize_mean_gradient),
        },
        "per_checkpoint_stats": per_checkpoint_stats,
    }

    metadata_path = output_dir / "influence_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    logger.info("=" * 76)
    logger.info("DOCUMENT-LEVEL INFLUENCE MATRIX COMPLETE")
    logger.info("Matrix:   %s", matrix_path)
    logger.info("Metadata: %s", metadata_path)
    logger.info("Shape:    %s", [n_documents, n_checkpoints])
    logger.info("=" * 76)


if __name__ == "__main__":
    main()
