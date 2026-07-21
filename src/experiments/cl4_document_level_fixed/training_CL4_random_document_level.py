from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, set_seed
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
    latest_log,
    WordExposureCheckpointCallback,
    DetailedCheckpointCallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


class EpochCheckpointCallback(TrainerCallback):
    """Save one surrogate checkpoint at the end of each epoch."""

    def __init__(self, tokenizer, output_dir: str, corpus_metadata: dict):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_metadata = corpus_metadata

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or state.epoch is None:
            return control

        epoch_number = int(round(state.epoch))
        save_dir = self.output_dir / f"epoch_{epoch_number:02d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(save_dir)

        log = latest_log(state)
        metadata = {
            "epoch": epoch_number,
            "global_step": state.global_step,
            "loss": safe_float(log.get("loss", log.get("train_loss"))),
            "learning_rate": safe_float(log.get("learning_rate")),
            "timestamp": datetime.now().isoformat(),
            "purpose": "CL4 document-level TICL surrogate checkpoint for influence computation",
            "influence_unit_expected_downstream": "raw_document",
            "corpus_metadata": self.corpus_metadata,
        }
        (save_dir / "epoch_checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved surrogate epoch checkpoint: %s", save_dir)
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the random-order GPT-2 surrogate for document-level CL4/TICL. "
            "The surrogate is trained on randomly ordered packed LM blocks, while "
            "the downstream influence matrix scores raw documents."
        )
    )
    parser.add_argument("--dataset_name", default=BABYLM_DATASET)
    parser.add_argument("--tokenizer_name", default=BABYLM_TOKENIZER)
    parser.add_argument("--output_dir", default="./model_surrogate")
    parser.add_argument("--babylm_checkpoint_dir", default="./babylm_checkpoints_surrogate")
    parser.add_argument("--detailed_checkpoint_dir", default="./checkpoints_detailed_surrogate")
    parser.add_argument("--surrogate_epoch_checkpoint_dir", default="./surrogate_epoch_checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)
    parser.add_argument("--min_document_words", type=int, default=3)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def build_trainer(model, tokenizer, train_dataset, total_words: int, args, corpus_metadata: dict):
    train_cfg = TRAINING_CONFIG
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    bf16, fp16 = select_precision()

    training_args = make_training_arguments(
        output_dir=args.output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        adam_epsilon=train_cfg["adam_epsilon"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_strategy="steps",
        logging_steps=train_cfg["logging_steps"],
        logging_first_step=True,
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        optim="adamw_torch",
        seed=train_cfg["seed"],
        data_seed=train_cfg["seed"],
        dataloader_drop_last=True,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        remove_unused_columns=False,
        report_to="none",
        group_by_length=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    word_cb = WordExposureCheckpointCallback(
        tokenizer=tokenizer,
        output_dir=args.babylm_checkpoint_dir,
        total_words=total_words,
        num_epochs=train_cfg["num_epochs"],
        max_seq_length=args.max_seq_length,
        pipeline_step="cl4_document_level_surrogate_random_order",
        extra_metadata={
            "surrogate_training_unit": "packed_lm_block",
            "downstream_influence_unit": "raw_document",
            "min_document_words": args.min_document_words,
        },
    )
    epoch_cb = EpochCheckpointCallback(
        tokenizer=tokenizer,
        output_dir=args.surrogate_epoch_checkpoint_dir,
        corpus_metadata=corpus_metadata,
    )
    detail_cb = DetailedCheckpointCallback(
        output_dir=args.detailed_checkpoint_dir,
        every=train_cfg["detailed_checkpoint_every_n_steps"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[word_cb, epoch_cb, detail_cb],
    )
    return trainer, word_cb


def main() -> None:
    args = parse_args()
    set_seed(TRAINING_CONFIG["seed"])

    logger.info("=" * 76)
    logger.info("CL4 document-level TICL — random surrogate training")
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Tokenizer: %s", args.tokenizer_name)
    logger.info("Min document words: %s", args.min_document_words)
    logger.info("=" * 76)

    tokenizer = build_tokenizer(args.tokenizer_name)

    documents, total_words, corpus_stats = load_document_corpus(
        dataset_name=args.dataset_name,
        min_document_words=args.min_document_words,
        max_train_examples=args.max_train_examples,
        logger=logger,
    )

    # Random surrogate training: shuffle documents before packing. This gives the
    # surrogate a random-order presentation while preserving a document-level corpus
    # definition for the influence stage.
    train_dataset, packing_stats = pack_documents_to_lm_blocks(
        documents=documents,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        tokenize_batch_size=args.tokenize_batch_size,
        shuffle_seed=TRAINING_CONFIG["seed"],
        logger=logger,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_metadata = {
        "corpus": corpus_stats,
        "packing": packing_stats,
        "note": (
            "Surrogate is trained on randomly packed LM blocks; influence will be "
            "computed over raw documents using the same filtered corpus."
        ),
    }
    (output_dir / "document_corpus_metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2, default=str),
        encoding="utf-8",
    )

    model = create_gpt2_model(tokenizer)
    logger.info("Surrogate parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    trainer, word_cb = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        total_words=total_words,
        args=args,
        corpus_metadata=dataset_metadata,
    )

    world_size = max(int(trainer.args.world_size), 1)
    effective_batch = TRAINING_CONFIG["batch_size"] * TRAINING_CONFIG["gradient_accumulation_steps"] * world_size
    steps_per_epoch = len(train_dataset) // effective_batch
    logger.info("=" * 76)
    logger.info("TRAINING PLAN")
    logger.info("Documents after filter: %s", f"{len(documents):,}")
    logger.info("Packed LM blocks:       %s", f"{len(train_dataset):,}")
    logger.info("Effective batch:        %s", f"{effective_batch:,}")
    logger.info("Approx steps/epoch:     %s", f"{steps_per_epoch:,}")
    logger.info("Epochs:                 %s", TRAINING_CONFIG["num_epochs"])
    logger.info("Planned word exposure:  %s", f"{total_words * TRAINING_CONFIG['num_epochs']:,}")
    logger.info("=" * 76)

    resume = args.resume_from_checkpoint
    if resume is None and TRAINING_CONFIG["auto_resume"] and os.path.isdir(args.output_dir):
        resume = get_last_checkpoint(args.output_dir)

    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    final_metadata = {
        "completed_at": datetime.now().isoformat(),
        "pipeline_role": "random_surrogate_for_document_level_influence",
        "model_architecture": "GPT-2",
        "objective": "causal_language_modeling",
        "surrogate_training_unit": "packed_lm_block",
        "downstream_influence_unit": "raw_document",
        "dataset": args.dataset_name,
        "total_words_after_document_filter": total_words,
        "planned_word_exposure": total_words * TRAINING_CONFIG["num_epochs"],
        "final_global_step": trainer.state.global_step,
        "final_epoch": safe_float(trainer.state.epoch),
        "dataset_metadata": dataset_metadata,
        "training_config": TRAINING_CONFIG,
    }
    (final_dir / "final_model_metadata.json").write_text(
        json.dumps(final_metadata, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("Surrogate training complete.")
    logger.info("BabyLM checkpoints saved: %s", len(word_cb.saved))


if __name__ == "__main__":
    main()
