from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import SequentialSampler
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from experiments.cl4_document_level_fixed.cl4_common import (
    BABYLM_DATASET,
    BABYLM_TOKENIZER,
    TRAINING_CONFIG,
    build_tokenizer,
    create_gpt2_model,
    load_document_corpus,
    pack_documents_to_lm_blocks,
    make_training_arguments,
    select_precision,
    safe_float,
    WordExposureCheckpointCallback,
    DetailedCheckpointCallback,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh GPT-2 model on a document-level descending influence "
            "curriculum. Documents are sorted first, then packed into LM blocks."
        )
    )
    parser.add_argument("--influence_matrix", default="./influence_output_document_level/influence_matrix.npy")
    parser.add_argument("--influence_metadata", default=None)
    parser.add_argument("--dataset_name", default=BABYLM_DATASET)
    parser.add_argument("--tokenizer_name", default=BABYLM_TOKENIZER)
    parser.add_argument("--output_dir", default="./model_ticl_descending_document_level")
    parser.add_argument("--babylm_checkpoint_dir", default="./babylm_checkpoints_ticl_descending_document_level")
    parser.add_argument("--detailed_checkpoint_dir", default="./checkpoints_detailed_ticl_descending_document_level")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)
    parser.add_argument("--min_document_words", type=int, default=3)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--bin_size", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def load_and_validate_influence(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    phi = np.load(path, mmap_mode="r")
    if phi.ndim != 2 or phi.shape[1] == 0:
        raise ValueError(f"Expected 2D influence matrix, found {phi.shape}")
    if not np.isfinite(phi).all():
        raise ValueError("Influence matrix contains NaN or infinite values.")
    return phi


def validate_metadata(path: Path, phi: np.ndarray, args: argparse.Namespace) -> dict[str, Any] | None:
    if not path.is_file():
        logger.warning("No influence metadata found at %s", path)
        return None

    meta = json.loads(path.read_text(encoding="utf-8"))
    errors = []

    expected = {
        "dataset": args.dataset_name,
        "tokenizer": args.tokenizer_name,
        "max_seq_length": args.max_seq_length,
        "min_document_words": args.min_document_words,
    }
    for key, value in expected.items():
        if key in meta and meta[key] != value:
            errors.append(f"{key}: {meta[key]!r} != {value!r}")

    if meta.get("curriculum_unit") not in (None, "raw_document"):
        errors.append(f"curriculum_unit: {meta.get('curriculum_unit')!r} != 'raw_document'")

    if meta.get("n_documents") not in (None, int(phi.shape[0])):
        errors.append(f"n_documents: {meta.get('n_documents')} != {phi.shape[0]}")

    if errors:
        raise ValueError("Influence metadata mismatch:\n- " + "\n- ".join(errors))

    return meta


def build_document_curriculum(phi: np.ndarray, bin_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    mean_score = np.asarray(phi.mean(axis=1), dtype=np.float32)

    # Descending = highest influence first. According to the TICL interpretation,
    # high-influence examples are easier / more aligned with the average learning
    # direction, so this is easy-to-hard.
    indices = np.argsort(-mean_score, kind="stable").astype(np.int64)

    rng = np.random.default_rng(seed)
    for start in range(0, len(indices), bin_size):
        rng.shuffle(indices[start:start + bin_size])

    if not np.array_equal(np.sort(indices), np.arange(len(indices))):
        raise RuntimeError("Document curriculum is not a valid permutation.")

    meta = {
        "n_documents": int(phi.shape[0]),
        "n_surrogate_checkpoints": int(phi.shape[1]),
        "score": "mean document influence across checkpoint columns",
        "sort": "descending stable sort",
        "interpretation": "higher influence first, easy-to-hard",
        "bin_size": int(bin_size),
        "within_bin_shuffle": True,
        "shuffle_seed": int(seed),
        "static_across_epochs": True,
        "curriculum_unit": "raw_document",
        "packing_after_sorting": True,
        "score_min": float(mean_score.min()),
        "score_max": float(mean_score.max()),
        "score_mean": float(mean_score.mean()),
        "score_std": float(mean_score.std()),
    }
    return indices, mean_score, meta


class StaticCurriculumTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return None if dataset is None else SequentialSampler(dataset)


def build_trainer(model, tokenizer, train_dataset: Dataset, total_words: int, args, curriculum_meta: dict[str, Any]):
    t = TRAINING_CONFIG
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    bf16, fp16 = select_precision()

    train_args = make_training_arguments(
        output_dir=args.output_dir,
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        adam_beta1=t["adam_beta1"],
        adam_beta2=t["adam_beta2"],
        adam_epsilon=t["adam_epsilon"],
        max_grad_norm=t["max_grad_norm"],
        logging_strategy="steps",
        logging_steps=t["logging_steps"],
        logging_first_step=True,
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        optim="adamw_torch",
        seed=t["seed"],
        data_seed=t["seed"],
        dataloader_drop_last=True,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=t["dataloader_num_workers"],
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=t["gradient_checkpointing"],
        remove_unused_columns=False,
        report_to="none",
        group_by_length=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    word_cb = WordExposureCheckpointCallback(
        tokenizer=tokenizer,
        output_dir=args.babylm_checkpoint_dir,
        total_words=total_words,
        num_epochs=t["num_epochs"],
        max_seq_length=args.max_seq_length,
        pipeline_step="cl4_document_level_descending_curriculum",
        extra_metadata={
            "curriculum_unit": "raw_document",
            "curriculum_order": "descending_mean_document_influence",
            "packing_after_sorting": True,
            "static_across_epochs": True,
        },
    )
    detail_cb = DetailedCheckpointCallback(
        output_dir=args.detailed_checkpoint_dir,
        every=t["detailed_checkpoint_every_n_steps"],
    )

    trainer = StaticCurriculumTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[word_cb, detail_cb],
    )
    return trainer, word_cb


def main() -> None:
    args = parse_args()
    set_seed(TRAINING_CONFIG["seed"])

    tokenizer = build_tokenizer(args.tokenizer_name)

    influence_path = Path(args.influence_matrix)
    metadata_path = Path(args.influence_metadata) if args.influence_metadata else influence_path.with_name("influence_metadata.json")

    phi = load_and_validate_influence(influence_path)
    source_metadata = validate_metadata(metadata_path, phi, args)

    documents, total_words, corpus_stats = load_document_corpus(
        dataset_name=args.dataset_name,
        min_document_words=args.min_document_words,
        max_train_examples=args.max_train_examples,
        logger=logger,
    )

    if len(documents) != phi.shape[0]:
        raise ValueError(
            f"Document/influence mismatch: loaded {len(documents):,} documents, "
            f"but influence matrix has {phi.shape[0]:,} rows. The document filtering, "
            "segmentation, min_document_words, debug cap, dataset, and tokenizer must "
            "match the influence-matrix run exactly."
        )

    indices, scores, curriculum_meta = build_document_curriculum(
        phi=phi,
        bin_size=args.bin_size,
        seed=TRAINING_CONFIG["seed"],
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "document_curriculum_indices.npy", indices)
    np.save(out / "mean_document_influence_scores.npy", scores)

    ordered_documents = documents.select(indices.tolist())

    # Correct document-level curriculum behavior:
    # document influence -> sort documents -> local shuffle -> THEN tokenize/pack.
    curriculum_dataset, packing_stats = pack_documents_to_lm_blocks(
        documents=ordered_documents,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        tokenize_batch_size=args.tokenize_batch_size,
        shuffle_seed=None,
        logger=logger,
    )

    full_meta = {
        "created_at": datetime.now().isoformat(),
        "pipeline_step": "cl4_document_level_descending_curriculum_training",
        "influence_matrix": str(influence_path),
        "influence_metadata": source_metadata,
        "dataset": args.dataset_name,
        "tokenizer": args.tokenizer_name,
        "corpus": corpus_stats,
        "curriculum": curriculum_meta,
        "packing": packing_stats,
        "training_unit": "packed_lm_block_after_document_sorting",
        "curriculum_unit": "raw_document",
    }
    (out / "curriculum_metadata.json").write_text(
        json.dumps(full_meta, indent=2, default=str),
        encoding="utf-8",
    )

    model = create_gpt2_model(tokenizer)
    logger.info("Fresh model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    trainer, word_cb = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=curriculum_dataset,
        total_words=total_words,
        args=args,
        curriculum_meta=full_meta,
    )

    world = max(int(trainer.args.world_size), 1)
    effective_batch = TRAINING_CONFIG["batch_size"] * TRAINING_CONFIG["gradient_accumulation_steps"] * world

    logger.info("=" * 76)
    logger.info("DOCUMENT-LEVEL CURRICULUM TRAINING PLAN")
    logger.info("Documents:        %s", f"{len(documents):,}")
    logger.info("Packed LM blocks: %s", f"{len(curriculum_dataset):,}")
    logger.info("Effective batch:  %s", f"{effective_batch:,}")
    logger.info("Epochs:           %s", TRAINING_CONFIG["num_epochs"])
    logger.info("Order:            descending mean document influence with within-bin shuffle")
    logger.info("=" * 76)

    resume = args.resume_from_checkpoint
    if resume is None and TRAINING_CONFIG["auto_resume"] and os.path.isdir(args.output_dir):
        resume = get_last_checkpoint(args.output_dir)

    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    final_metadata = {
        "completed_at": datetime.now().isoformat(),
        "pipeline_role": "final_cl4_document_level_descending_curriculum_model",
        "training_order": "descending_mean_document_influence_within_bin_shuffle",
        "curriculum_unit": "raw_document",
        "training_unit": "packed_lm_block_after_document_sorting",
        "static_across_epochs": True,
        "dataset": args.dataset_name,
        "track": "BabyLM-2026-Strict-Small",
        "total_words_after_document_filter": total_words,
        "planned_word_exposure": total_words * TRAINING_CONFIG["num_epochs"],
        "total_documents": len(documents),
        "total_packed_blocks": len(curriculum_dataset),
        "final_global_step": trainer.state.global_step,
        "final_epoch": safe_float(trainer.state.epoch),
        "curriculum_metadata": full_meta,
        "training_config": TRAINING_CONFIG,
    }
    (final_dir / "final_model_metadata.json").write_text(
        json.dumps(final_metadata, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Training complete. Saved %s BabyLM checkpoints.", len(word_cb.saved))


if __name__ == "__main__":
    main()
