from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import SequentialSampler
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
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO,
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
        "save_total_limit": 2,
        "dataloader_num_workers": 0,
        "detailed_checkpoint_every_n_steps": 500,
        "auto_resume": True,
    },

    "data": {
        "max_seq_length": 128,
        "tokenize_batch_size": 1_000,
        "hf_dataset": "flakoash/babylm-curriculum-tiered-4bands",
        "tokenizer_name": (
            "BabyLM-community/"
            "BabyLM-2026-Baseline-GPT2-Strict-Small"
        ),
        "curriculum_files": [
            "curriculum/epoch_00.jsonl",
            "curriculum/epoch_01.jsonl",
            "curriculum/epoch_02.jsonl",
            "curriculum/epoch_03.jsonl",
        ],
    },

    "output_dir": "./model_tiered",
    "babylm_checkpoint_dir": "./babylm_checkpoints_tiered",
    "detailed_checkpoint_dir": "./checkpoints_detailed_tiered",

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
    """Return the newest log entry containing useful optimization data."""
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
        raise ValueError("n_ctx and n_positions must match in this setup.")

    if not 0.0 <= train_cfg["warmup_ratio"] < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")

    if train_cfg["num_epochs"] <= 0:
        raise ValueError("num_epochs must be positive.")

    if not data_cfg["curriculum_files"]:
        raise ValueError("At least one curriculum file is required.")


# =============================================================================
# Ordered curriculum preprocessing
# =============================================================================

def load_curriculum_bands(
    config: dict[str, Any],
) -> list[tuple[str, Dataset, int]]:
    """
    Load each curriculum band independently.
    """
    data_cfg = config["data"]
    bands: list[tuple[str, Dataset, int]] = []

    for curriculum_file in data_cfg["curriculum_files"]:
        logger.info("Loading curriculum band: %s", curriculum_file)

        dataset = load_dataset(
            data_cfg["hf_dataset"],
            data_files=curriculum_file,
            split="train",
        )

        if "text" not in dataset.column_names:
            raise ValueError(
                f"{curriculum_file} has no 'text' column. "
                f"Columns: {dataset.column_names}"
            )

        word_count = sum(
            count_whitespace_words(text)
            for text in dataset["text"]
        )

        logger.info(
            "%s: %s documents, %s whitespace words",
            curriculum_file,
            f"{len(dataset):,}",
            f"{word_count:,}",
        )

        bands.append((curriculum_file, dataset, word_count))

    return bands


def chunk_band_in_order(
    dataset: Dataset,
    tokenizer,
    max_seq_length: int,
    tokenize_batch_size: int,
    band_name: str,
) -> Dataset:
    """
    Convert one curriculum band into dense fixed-length blocks.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")

    stats = {
        "documents": 0,
        "subword_tokens_with_eos": 0,
        "blocks": 0,
        "discarded_remainder": 0,
    }

    def generate_blocks() -> Iterator[dict[str, list[int]]]:
        carry: list[int] = []

        for batch in dataset.iter(batch_size=tokenize_batch_size):
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

        logger.info(
            "%s preprocessing complete: documents=%s, subwords=%s, "
            "blocks=%s, final remainder discarded=%s",
            band_name,
            f"{stats['documents']:,}",
            f"{stats['subword_tokens_with_eos']:,}",
            f"{stats['blocks']:,}",
            f"{stats['discarded_remainder']:,}",
        )

    chunked = Dataset.from_generator(generate_blocks)

    if len(chunked) == 0:
        raise ValueError(
            f"Curriculum band {band_name} produced no complete "
            f"{max_seq_length}-token blocks."
        )

    return chunked


def build_ordered_curriculum(
    config: dict[str, Any],
    tokenizer,
) -> tuple[Dataset, int]:
    """
    Tokenize each band independently and concatenate the completed blocks in
    configured curriculum order.
    """
    data_cfg = config["data"]
    raw_bands = load_curriculum_bands(config)

    chunked_bands: list[Dataset] = []
    total_words = 0

    for band_name, raw_dataset, band_words in raw_bands:
        chunked_band = chunk_band_in_order(
            dataset=raw_dataset,
            tokenizer=tokenizer,
            max_seq_length=data_cfg["max_seq_length"],
            tokenize_batch_size=data_cfg["tokenize_batch_size"],
            band_name=band_name,
        )

        chunked_bands.append(chunked_band)
        total_words += band_words

        logger.info(
            "%s: %s completed training blocks",
            band_name,
            f"{len(chunked_band):,}",
        )

    full_dataset = concatenate_datasets(chunked_bands)

    logger.info("=" * 76)
    logger.info("ORDERED CL3 DATASET")
    logger.info("Curriculum order:")
    for filename in data_cfg["curriculum_files"]:
        logger.info("  %s", filename)
    logger.info("Total documents' whitespace words: %s", f"{total_words:,}")
    logger.info("Total 128-token blocks:           %s", f"{len(full_dataset):,}")
    logger.info(
        "Subword positions per epoch:       %s",
        f"{len(full_dataset) * data_cfg['max_seq_length']:,}",
    )
    logger.info("=" * 76)

    return full_dataset, total_words


# =============================================================================
# Trainer preserving curriculum order
# =============================================================================

class CurriculumTrainer(Trainer):
    """Use sequential sampling so Trainer does not shuffle the curriculum."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = (
            train_dataset
            if train_dataset is not None
            else self.train_dataset
        )

        if dataset is None or not hasattr(dataset, "__len__"):
            return None

        return SequentialSampler(dataset)


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
        self.output_dir = config["babylm_checkpoint_dir"]
        self.checkpoints_saved: set[int] = set()

        os.makedirs(self.output_dir, exist_ok=True)

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
        checkpoint_name = self._checkpoint_name(threshold_words)
        save_dir = Path(self.output_dir) / checkpoint_name
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
        subwords_seen = self._subword_positions_seen(args, state)

        metadata = {
            "checkpoint_name": checkpoint_name,
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
            "track": "strict-small",
            "curriculum": "CL3_tiered_static",
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
        logger.info("Saved BabyLM checkpoint: %s", checkpoint_name)
        logger.info("Path: %s", save_dir)
        logger.info(
            "Word threshold: %s | estimated exposure: %s",
            f"{threshold_words:,}",
            f"{estimated_words:,}",
        )
        logger.info("Subword positions seen: %s", f"{subwords_seen:,}")
        logger.info("Global step: %s", f"{state.global_step:,}")
        logger.info("=" * 76)

    def on_train_begin(self, args, state, control, **kwargs):
        estimated_words = self._estimated_words_seen(state)

        # Handles normal Trainer resume: thresholds below the resumed progress
        # are not re-created.
        self.checkpoints_saved = {
            threshold
            for threshold in self.checkpoint_intervals
            if threshold <= estimated_words
        }

        logger.info("=" * 76)
        logger.info("BABYLM WORD-EXPOSURE CHECKPOINTING")
        logger.info(
            "Corpus words:                    %s",
            f"{self.total_words_in_corpus:,}",
        )
        logger.info(
            "Planned word exposure:           %s",
            f"{self.total_planned_word_exposure:,}",
        )
        logger.info(
            "Subword positions/optimizer step:%s",
            f"{self._subword_positions_per_step(args):,}",
        )
        logger.info(
            "Initial estimated word exposure: %s",
            f"{estimated_words:,}",
        )
        logger.info("=" * 76)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control

        estimated_words = self._estimated_words_seen(state)

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
# Detailed JSON logger
# =============================================================================

class DetailedCheckpointCallback(TrainerCallback):
    """Write lightweight JSON monitoring records every N optimizer steps."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.checkpoint_dir = Path(config["detailed_checkpoint_dir"])
        self.save_every_n_steps = config["training"][
            "detailed_checkpoint_every_n_steps"
        ]
        self.checkpoint_info: dict[int, dict[str, Any]] = {}

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step

        if step > 0 and step % self.save_every_n_steps == 0:
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
                    latest_log.get(
                        "loss",
                        latest_log.get("train_loss"),
                    )
                ),
                "learning_rate": safe_float(
                    latest_log.get(
                        "learning_rate",
                        args.learning_rate,
                    )
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
                self.checkpoint_dir
                / f"checkpoint_step_{step:06d}.json"
            )
            with step_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)

            self.checkpoint_info[step] = record

            with (
                self.checkpoint_dir / "checkpoint_log.json"
            ).open("w", encoding="utf-8") as handle:
                json.dump(
                    self.checkpoint_info,
                    handle,
                    indent=2,
                )

            loss = record["loss"]
            loss_text = f"{loss:.4f}" if loss is not None else "N/A"
            epoch = record["epoch"]
            epoch_text = f"{epoch:.3f}" if epoch is not None else "N/A"

            logger.info(
                "Step %s | loss=%s | epoch=%s | progress=%.1f%%",
                f"{step:,}",
                loss_text,
                epoch_text,
                progress,
            )

        return control


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    cfg = TRAINING_CONFIG
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    validate_config(cfg)
    set_seed(train_cfg["seed"])

    local_subwords_per_step = (
        train_cfg["batch_size"]
        * train_cfg["gradient_accumulation_steps"]
        * data_cfg["max_seq_length"]
    )

    logger.info("=" * 76)
    logger.info("BabyLM 2026 Strict-Small — CL3 Tiered Curriculum")
    logger.info(
        "Model: GPT-2 %sd / %s layers / %s heads",
        model_cfg["hidden_size"],
        model_cfg["num_hidden_layers"],
        model_cfg["num_attention_heads"],
    )
    logger.info(
        "Model positional capacity: %s",
        f"{model_cfg['n_positions']:,}",
    )
    logger.info(
        "Training sequence length:  %s",
        f"{data_cfg['max_seq_length']:,}",
    )
    logger.info(
        "Single-GPU subword positions/optimizer step: %s",
        f"{local_subwords_per_step:,}",
    )
    logger.info("=" * 76)

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------
    logger.info(
        "Loading BabyLM tokenizer from %s",
        data_cfg["tokenizer_name"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        data_cfg["tokenizer_name"]
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if len(tokenizer) != model_cfg["expected_vocab_size"]:
        raise ValueError(
            f"Expected tokenizer size "
            f"{model_cfg['expected_vocab_size']:,}, "
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
    # Dataset
    # -------------------------------------------------------------------------
    train_dataset, total_words = build_ordered_curriculum(
        config=cfg,
        tokenizer=tokenizer,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    logger.info("Initializing GPT-2 weights from scratch.")

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
        raise ValueError("Model embeddings do not match tokenizer length.")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    logger.info(
        "Model parameters: %s (%.1fM)",
        f"{total_parameters:,}",
        total_parameters / 1e6,
    )

    # -------------------------------------------------------------------------
    # Training arguments
    # -------------------------------------------------------------------------
    use_bf16, use_fp16 = select_precision()

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
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
        remove_unused_columns=False,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    word_checkpoint_callback = WordExposureCheckpointCallback(
        config=cfg,
        tokenizer=tokenizer,
        total_words_in_corpus=total_words,
    )
    detailed_callback = DetailedCheckpointCallback(cfg)

    trainer = CurriculumTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[
            word_checkpoint_callback,
            detailed_callback,
        ],
    )

    # -------------------------------------------------------------------------
    # Report calculated training plan
    # -------------------------------------------------------------------------
    world_size = max(int(training_args.world_size), 1)
    global_effective_batch = (
        train_cfg["batch_size"]
        * train_cfg["gradient_accumulation_steps"]
        * world_size
    )
    steps_per_epoch = len(train_dataset) // global_effective_batch
    approximate_total_steps = math.ceil(
        steps_per_epoch * train_cfg["num_epochs"]
    )

    logger.info("=" * 76)
    logger.info("TRAINING PLAN")
    logger.info("Curriculum sampler: SequentialSampler")
    logger.info(
        "Static order repeated each epoch: %s",
        " -> ".join(data_cfg["curriculum_files"]),
    )
    logger.info(
        "Global effective batch: %s sequences",
        f"{global_effective_batch:,}",
    )
    logger.info(
        "Global subword positions/step: %s",
        f"{global_effective_batch * data_cfg['max_seq_length']:,}",
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
        "Corpus words per pass: %s",
        f"{total_words:,}",
    )
    logger.info(
        "Planned word exposure: %s",
        f"{total_words * train_cfg['num_epochs']:,}",
    )
    logger.info(
        "Precision: bf16=%s, fp16=%s",
        use_bf16,
        use_fp16,
    )
    logger.info("=" * 76)

    # -------------------------------------------------------------------------
    # Resume automatically from the latest standard Trainer checkpoint
    # -------------------------------------------------------------------------
    resume_from_checkpoint: str | None = None

    if train_cfg["auto_resume"] and os.path.isdir(cfg["output_dir"]):
        resume_from_checkpoint = get_last_checkpoint(cfg["output_dir"])

    if resume_from_checkpoint:
        logger.info(
            "Resuming from Trainer checkpoint: %s",
            resume_from_checkpoint,
        )
    else:
        logger.info("Starting a fresh training run.")

    # -------------------------------------------------------------------------
    # Train and save
    # -------------------------------------------------------------------------
    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    logger.info("Saving final model and tokenizer to %s", cfg["output_dir"])
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    final_metadata = {
        "completed_at": datetime.now().isoformat(),
        "track": "BabyLM-2026-Strict-Small",
        "curriculum_name": "CL3_tiered_static",
        "curriculum_files": data_cfg["curriculum_files"],
        "curriculum_sampler": "sequential",
        "curriculum_repeated_each_epoch": True,
        "total_words_in_corpus": total_words,
        "num_epochs": train_cfg["num_epochs"],
        "planned_word_exposure": (
            total_words * train_cfg["num_epochs"]
        ),
        "tokenizer_name": data_cfg["tokenizer_name"],
        "tokenizer_length": len(tokenizer),
        "model_context_length": model.config.n_positions,
        "training_sequence_length": data_cfg["max_seq_length"],
        "total_parameters": total_parameters,
        "training_config": cfg,
        "final_global_step": trainer.state.global_step,
        "final_epoch": safe_float(trainer.state.epoch),
    }

    metadata_path = Path(cfg["output_dir"]) / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            final_metadata,
            handle,
            indent=2,
            default=str,
        )

    logger.info("=" * 76)
    logger.info("CL3 training complete.")
    logger.info("Final model: %s", cfg["output_dir"])
    logger.info(
        "Evaluation checkpoints: %s",
        cfg["babylm_checkpoint_dir"],
    )
    logger.info("=" * 76)


if __name__ == "__main__":
    main()