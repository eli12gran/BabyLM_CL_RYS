from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

TRAINING_CONFIG: dict[str, Any] = {
    "model": {
        "expected_vocab_size": 16_384,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3_072,
        "n_positions": 1_024,
        "n_ctx": 1_024,
        "attn_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "activation_function": "gelu_new",
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-5,
    },

    "training": {
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
        "num_epochs": 10,
        "warmup_ratio": 0.01,
        "weight_decay": 0.0,
        "lr_scheduler_type": "cosine",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "logging_steps": 100,
        "seed": 42,
        "max_grad_norm": 1.0,
        "save_steps": 500,
        "save_total_limit": 3,
        "dataloader_num_workers": 0,
        "detailed_checkpoint_every_n_steps": 500,
        "auto_resume": True,
        "gradient_checkpointing": True,
    },

    "data": {
        "max_seq_length": 128,
        "tokenize_batch_size": 1_000,
        "hf_dataset": "BabyLM-community/BabyLM-2026-Strict-Small",
        "tokenizer_name": (
            "BabyLM-community/"
            "BabyLM-2026-Baseline-GPT2-Strict-Small"
        ),
    },

    "output_dir": "./model_surrogate",
    "babylm_checkpoint_dir": "./babylm_checkpoints_surrogate",
    "detailed_checkpoint_dir": "./checkpoints_detailed_surrogate",
    "surrogate_epoch_checkpoint_dir": "./surrogate_epoch_checkpoints",

    "checkpoint_intervals": [
        1_000_000,
        2_000_000,
        3_000_000,
        4_000_000,
        5_000_000,
        6_000_000,
        7_000_000,
        8_000_000,
        9_000_000,
        10_000_000,
        20_000_000,
        30_000_000,
        40_000_000,
        50_000_000,
        60_000_000,
        70_000_000,
        80_000_000,
        90_000_000,
        100_000_000,
    ],
}


# =============================================================================
# Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the random-order GPT-2 surrogate used in the TICL pipeline."
        )
    )

    parser.add_argument(
        "--output_dir",
        default=TRAINING_CONFIG["output_dir"],
    )
    parser.add_argument(
        "--babylm_checkpoint_dir",
        default=TRAINING_CONFIG["babylm_checkpoint_dir"],
    )
    parser.add_argument(
        "--detailed_checkpoint_dir",
        default=TRAINING_CONFIG["detailed_checkpoint_dir"],
    )
    parser.add_argument(
        "--surrogate_epoch_checkpoint_dir",
        default=TRAINING_CONFIG["surrogate_epoch_checkpoint_dir"],
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=TRAINING_CONFIG["data"]["max_seq_length"],
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Optional raw-example cap for debugging.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help=(
            "Explicit Hugging Face Trainer checkpoint. When omitted, the "
            "script automatically resumes from the latest checkpoint-* "
            "inside output_dir when auto_resume is enabled."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_whitespace_words(text: str) -> int:
    return len(text.split()) if text else 0


def latest_training_log(state) -> dict[str, Any]:
    for entry in reversed(state.log_history):
        if any(key in entry for key in ("loss", "train_loss", "learning_rate")):
            return entry
    return {}


def select_precision() -> tuple[bool, bool]:
    """Return (use_bf16, use_fp16)."""
    if not torch.cuda.is_available():
        return False, False

    use_bf16 = bool(
        hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    return use_bf16, not use_bf16


def validate_config(config: dict[str, Any]) -> None:
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    if data_cfg["max_seq_length"] > model_cfg["n_positions"]:
        raise ValueError(
            f"max_seq_length={data_cfg['max_seq_length']} exceeds "
            f"n_positions={model_cfg['n_positions']}."
        )

    if model_cfg["n_ctx"] != model_cfg["n_positions"]:
        raise ValueError("n_ctx and n_positions must match.")

    if not 0.0 <= train_cfg["warmup_ratio"] < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")

    if train_cfg["num_epochs"] <= 0:
        raise ValueError("num_epochs must be positive.")


# =============================================================================
# Dataset preprocessing
# =============================================================================

def load_and_prepare_dataset(
    config: dict[str, Any],
    tokenizer,
    max_train_examples: int | None = None,
) -> tuple[Dataset, int]:

    data_cfg = config["data"]
    seed = config["training"]["seed"]
    max_seq_length = data_cfg["max_seq_length"]
    tokenize_batch_size = data_cfg["tokenize_batch_size"]

    logger.info("Loading dataset: %s", data_cfg["hf_dataset"])

    raw_dataset = load_dataset(
        data_cfg["hf_dataset"],
        split="train",
        trust_remote_code=True,
    )

    if "text" not in raw_dataset.column_names:
        raise ValueError(
            f"Dataset has no 'text' column. Columns: "
            f"{raw_dataset.column_names}"
        )

    if max_train_examples is not None:
        limit = min(max_train_examples, len(raw_dataset))
        logger.warning(
            "DEBUG MODE: retaining only %s raw examples.",
            f"{limit:,}",
        )
        raw_dataset = raw_dataset.select(range(limit))

    total_words = sum(
        count_whitespace_words(text)
        for text in raw_dataset["text"]
    )

    logger.info(
        "Raw corpus: %s documents, %s whitespace words",
        f"{len(raw_dataset):,}",
        f"{total_words:,}",
    )

    # This deterministic shuffle defines the stable sequence row identities used
    # by the influence matrix. The Trainer will additionally randomize sampling.
    raw_dataset = raw_dataset.shuffle(seed=seed)
    logger.info(
        "Applied deterministic raw-document shuffle with seed=%s.",
        seed,
    )

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")

    stats = {
        "documents": 0,
        "subword_tokens_with_eos": 0,
        "blocks": 0,
        "discarded_remainder": 0,
    }

    def block_generator() -> Iterator[dict[str, list[int]]]:
        carry: list[int] = []

        for batch in raw_dataset.iter(batch_size=tokenize_batch_size):
            encoded = tokenizer(
                batch["text"],
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

            batch_ids: list[int] = []

            for document_ids in encoded["input_ids"]:
                batch_ids.extend(document_ids)
                batch_ids.append(eos_id)

                stats["documents"] += 1
                stats["subword_tokens_with_eos"] += len(document_ids) + 1

            combined = carry + batch_ids
            complete_length = (
                len(combined) // max_seq_length
            ) * max_seq_length

            for start in range(0, complete_length, max_seq_length):
                block = combined[start : start + max_seq_length]
                stats["blocks"] += 1

                yield {
                    "input_ids": block,
                    "attention_mask": [1] * max_seq_length,
                }

            carry = combined[complete_length:]

        stats["discarded_remainder"] = len(carry)

    train_dataset = Dataset.from_generator(block_generator)

    if len(train_dataset) == 0:
        raise ValueError(
            "Preprocessing produced no complete training sequences."
        )

    logger.info("=" * 76)
    logger.info("RANDOM-SURROGATE DATASET")
    logger.info("Documents processed:             %s", f"{stats['documents']:,}")
    logger.info(
        "Subword tokens including EOS:    %s",
        f"{stats['subword_tokens_with_eos']:,}",
    )
    logger.info("Fixed-length sequences:          %s", f"{len(train_dataset):,}")
    logger.info("Sequence length:                 %s", max_seq_length)
    logger.info(
        "Only final remainder discarded: %s tokens",
        f"{stats['discarded_remainder']:,}",
    )
    logger.info("=" * 76)

    return train_dataset, total_words


# =============================================================================
# Model
# =============================================================================

def create_model(
    config: dict[str, Any],
    tokenizer,
) -> GPT2LMHeadModel:
    model_cfg = config["model"]

    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=model_cfg["hidden_size"],
        n_layer=model_cfg["num_hidden_layers"],
        n_head=model_cfg["num_attention_heads"],
        n_inner=model_cfg["intermediate_size"],
        n_positions=model_cfg["n_positions"],
        n_ctx=model_cfg["n_ctx"],
        attn_pdrop=model_cfg["attn_pdrop"],
        embd_pdrop=model_cfg["embd_pdrop"],
        resid_pdrop=model_cfg["resid_pdrop"],
        activation_function=model_cfg["activation_function"],
        initializer_range=model_cfg["initializer_range"],
        layer_norm_epsilon=model_cfg["layer_norm_epsilon"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        use_cache=True,
    )

    model = GPT2LMHeadModel(model_config)

    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError(
            "Model embedding rows do not match tokenizer length."
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    logger.info("=" * 76)
    logger.info("SURROGATE MODEL")
    logger.info("Parameters:              %s", f"{total_parameters:,}")
    logger.info("Vocabulary:              %s", f"{len(tokenizer):,}")
    logger.info("Model positional capacity:%s", model.config.n_positions)
    logger.info(
        "Training sequence length:%s",
        config["data"]["max_seq_length"],
    )
    logger.info("=" * 76)

    return model


# =============================================================================
# BabyLM word-exposure checkpoints
# =============================================================================

class WordExposureCheckpointCallback(TrainerCallback):

    def __init__(
        self,
        config: dict[str, Any],
        tokenizer,
        total_words_in_corpus: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.total_words_in_corpus = total_words_in_corpus
        self.num_epochs = config["training"]["num_epochs"]
        self.max_seq_length = config["data"]["max_seq_length"]
        self.total_planned_word_exposure = int(
            total_words_in_corpus * self.num_epochs
        )

        self.checkpoint_intervals = sorted(
            config["checkpoint_intervals"]
        )
        self.output_dir = Path(config["babylm_checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_saved: set[int] = set()
        self.total_words_seen = 0.0

    def _global_effective_batch_size(self, args) -> int:
        world_size = max(int(getattr(args, "world_size", 1)), 1)
        return (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * world_size
        )

    def _subword_positions_per_step(self, args) -> int:
        return (
            self._global_effective_batch_size(args)
            * self.max_seq_length
        )

    def _subword_positions_seen(self, args, state) -> int:
        return state.global_step * self._subword_positions_per_step(args)

    def _estimated_words_seen(self, state) -> int:
        if not state.max_steps:
            return 0

        progress = min(
            max(state.global_step / state.max_steps, 0.0),
            1.0,
        )
        return int(progress * self.total_planned_word_exposure)

    @staticmethod
    def _checkpoint_name(words: int) -> str:
        return f"chck_{words // 1_000_000}M"

    def _save(
        self,
        model,
        args,
        state,
        threshold_words: int,
    ) -> None:
        name = self._checkpoint_name(threshold_words)
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(
            save_dir,
            safe_serialization=True,
        )
        self.tokenizer.save_pretrained(save_dir)

        with (save_dir / "training_args.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                args.to_dict(),
                handle,
                indent=2,
                default=str,
            )

        latest_log = latest_training_log(state)
        estimated_words = self._estimated_words_seen(state)
        self.total_words_seen = float(estimated_words)
        subwords_seen = self._subword_positions_seen(args, state)

        metadata = {
            "checkpoint_name": name,
            "checkpoint_word_exposure_threshold": threshold_words,
            "estimated_word_exposure_at_save": estimated_words,
            "total_words_in_corpus": self.total_words_in_corpus,
            "planned_epochs": self.num_epochs,
            "total_planned_word_exposure": (
                self.total_planned_word_exposure
            ),
            "subword_token_positions_seen": subwords_seen,
            "subword_positions_per_optimizer_step": (
                self._subword_positions_per_step(args)
            ),
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": safe_float(state.epoch),
            "loss": safe_float(
                latest_log.get("loss", latest_log.get("train_loss"))
            ),
            "learning_rate": safe_float(
                latest_log.get("learning_rate")
            ),
            "model_context_length": model.config.n_positions,
            "training_sequence_length": self.max_seq_length,
            "per_device_train_batch_size": (
                args.per_device_train_batch_size
            ),
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "world_size": int(getattr(args, "world_size", 1)),
            "timestamp": datetime.now().isoformat(),
            "checkpoint_type": "babylm_evaluation_checkpoint",
            "checkpoint_unit": "estimated_whitespace_word_exposure",
            "track": "BabyLM-2026-Strict-Small",
            "pipeline_step": "ticl_surrogate_random_order",
        }

        with (save_dir / "babylm_checkpoint_metadata.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metadata,
                handle,
                indent=2,
                default=str,
            )

        logger.info("=" * 76)
        logger.info("Saved BabyLM checkpoint: %s", name)
        logger.info(
            "Word threshold=%s | estimated exposure=%s",
            f"{threshold_words:,}",
            f"{estimated_words:,}",
        )
        logger.info("Subword positions=%s", f"{subwords_seen:,}")
        logger.info("Path: %s", save_dir)
        logger.info("=" * 76)

    def on_train_begin(self, args, state, control, **kwargs):
        estimated_words = self._estimated_words_seen(state)
        self.total_words_seen = float(estimated_words)

        self.checkpoints_saved = {
            threshold
            for threshold in self.checkpoint_intervals
            if threshold <= estimated_words
        }

        logger.info("=" * 76)
        logger.info("BABYLM WORD-EXPOSURE CHECKPOINTING")
        logger.info("Corpus words:              %s", f"{self.total_words_in_corpus:,}")
        logger.info(
            "Planned word exposure:     %s",
            f"{self.total_planned_word_exposure:,}",
        )
        logger.info(
            "Subword positions/step:    %s",
            f"{self._subword_positions_per_step(args):,}",
        )
        logger.info(
            "Initial estimated exposure:%s",
            f"{estimated_words:,}",
        )
        logger.info("=" * 76)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control

        estimated_words = self._estimated_words_seen(state)
        self.total_words_seen = float(estimated_words)

        for threshold in self.checkpoint_intervals:
            if (
                threshold <= self.total_planned_word_exposure
                and estimated_words >= threshold
                and threshold not in self.checkpoints_saved
            ):
                self.checkpoints_saved.add(threshold)
                self._save(
                    model=model,
                    args=args,
                    state=state,
                    threshold_words=threshold,
                )

        return control


# =============================================================================
# Per-epoch surrogate checkpoints
# =============================================================================

class EpochCheckpointCallback(TrainerCallback):
    """
    Save a complete model/tokenizer bundle at the end of every epoch.
    """

    def __init__(self, tokenizer, output_dir: str) -> None:
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            logger.warning(
                "EpochCheckpointCallback received no model; skipping."
            )
            return control

        if state.epoch is None:
            logger.warning("Epoch is unavailable; skipping epoch checkpoint.")
            return control

        epoch_number = int(round(state.epoch))
        save_dir = self.output_dir / f"epoch_{epoch_number:02d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(
            save_dir,
            safe_serialization=True,
        )
        self.tokenizer.save_pretrained(save_dir)

        latest_log = latest_training_log(state)

        metadata = {
            "epoch": epoch_number,
            "global_step": state.global_step,
            "loss": safe_float(
                latest_log.get("loss", latest_log.get("train_loss"))
            ),
            "learning_rate": safe_float(
                latest_log.get("learning_rate")
            ),
            "timestamp": datetime.now().isoformat(),
            "purpose": "TICL surrogate checkpoint for influence computation",
        }

        with (save_dir / "epoch_checkpoint_metadata.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2)

        logger.info(
            "Saved surrogate epoch checkpoint: %s",
            save_dir,
        )
        return control


# =============================================================================
# Detailed JSON logger
# =============================================================================

class DetailedCheckpointCallback(TrainerCallback):
    """Write lightweight JSON progress records every N optimizer steps."""

    def __init__(
        self,
        output_dir: str,
        save_every_n_steps: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_every_n_steps = save_every_n_steps
        self.records: dict[int, dict[str, Any]] = {}

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step

        if step <= 0 or step % self.save_every_n_steps != 0:
            return control

        latest_log = latest_training_log(state)
        progress = (
            step / state.max_steps * 100
            if state.max_steps
            else 0.0
        )
        world_size = max(int(getattr(args, "world_size", 1)), 1)

        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "loss": safe_float(
                latest_log.get("loss", latest_log.get("train_loss"))
            ),
            "learning_rate": safe_float(
                latest_log.get("learning_rate", args.learning_rate)
            ),
            "epoch": safe_float(state.epoch),
            "total_steps": state.max_steps,
            "progress_percent": round(progress, 2),
            "per_device_batch_size": (
                args.per_device_train_batch_size
            ),
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "world_size": world_size,
            "global_effective_batch_size": (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * world_size
            ),
        }

        step_path = (
            self.output_dir
            / f"checkpoint_step_{step:06d}.json"
        )
        with step_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)

        self.records[step] = record

        with (self.output_dir / "checkpoint_log.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(self.records, handle, indent=2)

        loss = record["loss"]
        loss_text = f"{loss:.4f}" if loss is not None else "N/A"

        logger.info(
            "Step %s | loss=%s | progress=%.1f%%",
            f"{step:,}",
            loss_text,
            progress,
        )
        return control


# =============================================================================
# Transformers compatibility
# =============================================================================

def make_training_arguments(**kwargs) -> TrainingArguments:
    supported = set(
        inspect.signature(TrainingArguments.__init__).parameters
    )

    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in supported
    }
    omitted = sorted(set(kwargs) - set(accepted))

    if omitted:
        logger.warning(
            "Unsupported TrainingArguments omitted for this transformers "
            "version: %s",
            ", ".join(omitted),
        )

    return TrainingArguments(**accepted)


# =============================================================================
# Training setup
# =============================================================================

def build_trainer(
    config: dict[str, Any],
    model: GPT2LMHeadModel,
    tokenizer,
    train_dataset: Dataset,
    total_words: int,
) -> tuple[Trainer, WordExposureCheckpointCallback]:
    train_cfg = config["training"]

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    use_bf16, use_fp16 = select_precision()

    training_args = make_training_arguments(
        output_dir=config["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=(
            train_cfg["gradient_accumulation_steps"]
        ),
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
        dataloader_num_workers=(
            train_cfg["dataloader_num_workers"]
        ),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=(
            train_cfg["gradient_checkpointing"]
        ),
        remove_unused_columns=False,
        report_to="none",
        group_by_length=False,
    )

    # Standard Trainer is intentional here: its random sampler randomizes
    # sequence presentation each epoch.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    word_callback = WordExposureCheckpointCallback(
        config=config,
        tokenizer=tokenizer,
        total_words_in_corpus=total_words,
    )
    epoch_callback = EpochCheckpointCallback(
        tokenizer=tokenizer,
        output_dir=config["surrogate_epoch_checkpoint_dir"],
    )
    detailed_callback = DetailedCheckpointCallback(
        output_dir=config["detailed_checkpoint_dir"],
        save_every_n_steps=train_cfg[
            "detailed_checkpoint_every_n_steps"
        ],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[
            word_callback,
            epoch_callback,
            detailed_callback,
        ],
    )

    logger.info(
        "Trainer sampler: standard random sampler "
        "(random-order surrogate)."
    )

    return trainer, word_callback


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    cfg = TRAINING_CONFIG
    cfg["output_dir"] = args.output_dir
    cfg["babylm_checkpoint_dir"] = args.babylm_checkpoint_dir
    cfg["detailed_checkpoint_dir"] = args.detailed_checkpoint_dir
    cfg["surrogate_epoch_checkpoint_dir"] = (
        args.surrogate_epoch_checkpoint_dir
    )
    cfg["data"]["max_seq_length"] = args.max_seq_length

    validate_config(cfg)
    set_seed(cfg["training"]["seed"])

    logger.info("=" * 76)
    logger.info("BabyLM 2026 Strict-Small — TICL Random Surrogate")
    logger.info(
        "GPU: %s",
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
    )
    logger.info("Dataset: %s", cfg["data"]["hf_dataset"])
    logger.info("Tokenizer: %s", cfg["data"]["tokenizer_name"])
    logger.info("Output: %s", cfg["output_dir"])
    logger.info("=" * 76)

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["data"]["tokenizer_name"]
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    expected_vocab_size = cfg["model"]["expected_vocab_size"]

    if len(tokenizer) != expected_vocab_size:
        raise ValueError(
            f"Expected tokenizer size {expected_vocab_size:,}, "
            f"but loaded {len(tokenizer):,}."
        )

    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer has no BOS token.")

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token.")

    logger.info("Tokenizer class: %s", tokenizer.__class__.__name__)
    logger.info("Tokenizer length: %s", f"{len(tokenizer):,}")
    logger.info(
        "BOS=%r (%s), EOS=%r (%s), PAD=%r (%s)",
        tokenizer.bos_token,
        tokenizer.bos_token_id,
        tokenizer.eos_token,
        tokenizer.eos_token_id,
        tokenizer.pad_token,
        tokenizer.pad_token_id,
    )

    # -------------------------------------------------------------------------
    # Dataset and stable sequence index
    # -------------------------------------------------------------------------
    train_dataset, total_words = load_and_prepare_dataset(
        config=cfg,
        tokenizer=tokenizer,
        max_train_examples=args.max_train_examples,
    )

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_index_metadata = {
        "dataset": cfg["data"]["hf_dataset"],
        "tokenizer": cfg["data"]["tokenizer_name"],
        "tokenizer_length": len(tokenizer),
        "raw_document_shuffle_seed": cfg["training"]["seed"],
        "total_sequences": len(train_dataset),
        "total_words_in_corpus": total_words,
        "max_seq_length": cfg["data"]["max_seq_length"],
        "document_boundary": "EOS appended after every document",
        "chunking": (
            "continuous carry across tokenizer batches; "
            "only final corpus remainder discarded"
        ),
        "note": (
            "Influence computation and curriculum reconstruction must use "
            "this exact preprocessing recipe."
        ),
    }

    with (output_dir / "dataset_sequence_index_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            dataset_index_metadata,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Model and Trainer
    # -------------------------------------------------------------------------
    model = create_model(cfg, tokenizer)

    trainer, word_callback = build_trainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        total_words=total_words,
    )

    world_size = max(int(trainer.args.world_size), 1)
    global_effective_batch = (
        cfg["training"]["batch_size"]
        * cfg["training"]["gradient_accumulation_steps"]
        * world_size
    )
    steps_per_epoch = len(train_dataset) // global_effective_batch
    approximate_total_steps = math.ceil(
        steps_per_epoch * cfg["training"]["num_epochs"]
    )

    logger.info("=" * 76)
    logger.info("TRAINING PLAN")
    logger.info("Training order: random")
    logger.info(
        "Global effective batch: %s sequences",
        f"{global_effective_batch:,}",
    )
    logger.info(
        "Global subword positions/step: %s",
        f"{global_effective_batch * cfg['data']['max_seq_length']:,}",
    )
    logger.info(
        "Approximate optimizer steps/epoch: %s",
        f"{steps_per_epoch:,}",
    )
    logger.info(
        "Approximate total optimizer steps: %s",
        f"{approximate_total_steps:,}",
    )
    logger.info(
        "Planned word exposure: %s",
        f"{total_words * cfg['training']['num_epochs']:,}",
    )
    logger.info(
        "Per-epoch influence checkpoints: %s",
        cfg["surrogate_epoch_checkpoint_dir"],
    )
    logger.info("=" * 76)

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------
    resume_from_checkpoint: str | None = args.resume_from_checkpoint

    if (
        resume_from_checkpoint is None
        and cfg["training"]["auto_resume"]
        and os.path.isdir(cfg["output_dir"])
    ):
        resume_from_checkpoint = get_last_checkpoint(
            cfg["output_dir"]
        )

    if resume_from_checkpoint:
        logger.info(
            "Resuming from Trainer checkpoint: %s",
            resume_from_checkpoint,
        )
    else:
        logger.info("Starting a fresh surrogate training run.")

    # -------------------------------------------------------------------------
    # Train and save
    # -------------------------------------------------------------------------
    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    final_dir = Path(cfg["output_dir"]) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    final_metadata = {
        "completed_at": datetime.now().isoformat(),
        "model_architecture": "GPT-2",
        "objective": "causal_language_modeling",
        "training_order": "random",
        "pipeline_role": "surrogate_for_influence_computation",
        "dataset": cfg["data"]["hf_dataset"],
        "track": "BabyLM-2026-Strict-Small",
        "total_words_in_corpus": total_words,
        "planned_word_exposure": (
            total_words * cfg["training"]["num_epochs"]
        ),
        "total_sequences": len(train_dataset),
        "tokenizer_name": cfg["data"]["tokenizer_name"],
        "tokenizer_length": len(tokenizer),
        "model_context_length": model.config.n_positions,
        "training_sequence_length": cfg["data"]["max_seq_length"],
        "final_global_step": trainer.state.global_step,
        "final_epoch": safe_float(trainer.state.epoch),
        "training_config": cfg,
        "next_step": (
            "Run influence computation using the checkpoints in "
            "surrogate_epoch_checkpoint_dir and reconstruct the sequence "
            "dataset using the exact same preprocessing function."
        ),
    }

    with (final_dir / "final_model_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            final_metadata,
            handle,
            indent=2,
            default=str,
        )

    epoch_dir = Path(cfg["surrogate_epoch_checkpoint_dir"])
    epoch_checkpoints = (
        sorted(
            path.name
            for path in epoch_dir.iterdir()
            if path.is_dir() and path.name.startswith("epoch_")
        )
        if epoch_dir.exists()
        else []
    )

    logger.info("=" * 76)
    logger.info("SURROGATE TRAINING COMPLETE")
    logger.info("Final model: %s", final_dir)
    logger.info(
        "BabyLM checkpoints saved: %s/%s",
        len(word_callback.checkpoints_saved),
        len(
            [
                threshold
                for threshold in cfg["checkpoint_intervals"]
                if threshold
                <= total_words * cfg["training"]["num_epochs"]
            ]
        ),
    )
    logger.info(
        "Estimated total word exposure: %s",
        f"{word_callback.total_words_seen:,.0f}",
    )
    logger.info(
        "Per-epoch checkpoints (%s): %s",
        len(epoch_checkpoints),
        epoch_checkpoints,
    )
    logger.info("=" * 76)


if __name__ == "__main__":
    main()