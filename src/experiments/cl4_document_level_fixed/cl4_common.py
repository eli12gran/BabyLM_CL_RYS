from __future__ import annotations

import inspect
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
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
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BABYLM_DATASET = "BabyLM-community/BabyLM-2026-Strict-Small"
BABYLM_TOKENIZER = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"

MODEL_CONFIG: dict[str, Any] = {
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
}

TRAINING_CONFIG: dict[str, Any] = {
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
}

CHECKPOINT_INTERVALS = [
    1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
    6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
    20_000_000, 30_000_000, 40_000_000, 50_000_000,
    60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000,
]


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def latest_log(state) -> dict[str, Any]:
    for row in reversed(state.log_history):
        if any(k in row for k in ("loss", "train_loss", "learning_rate")):
            return row
    return {}


def select_precision() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    bf16 = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    return bf16, not bf16


def make_training_arguments(**kwargs) -> TrainingArguments:
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    accepted = {k: v for k, v in kwargs.items() if k in supported}
    omitted = sorted(set(kwargs) - set(accepted))
    if omitted:
        logging.getLogger(__name__).warning(
            "Unsupported TrainingArguments omitted: %s", ", ".join(omitted)
        )
    return TrainingArguments(**accepted)


def build_tokenizer(tokenizer_name: str = BABYLM_TOKENIZER):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer must define bos_token_id.")
    if len(tokenizer) != MODEL_CONFIG["expected_vocab_size"]:
        raise ValueError(
            f"Expected tokenizer length {MODEL_CONFIG['expected_vocab_size']:,}, "
            f"got {len(tokenizer):,}."
        )
    return tokenizer


def save_portable_tokenizer(tokenizer, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out)
    cfg_path = out / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("backend", "is_local", "local_files_only"):
            cfg.pop(key, None)
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg["model_max_length"] = 1024
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    special = {
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
    }
    (out / "special_tokens_map.json").write_text(json.dumps(special, indent=2), encoding="utf-8")


def create_gpt2_model(tokenizer) -> GPT2LMHeadModel:
    m = MODEL_CONFIG
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=m["hidden_size"],
        n_layer=m["num_hidden_layers"],
        n_head=m["num_attention_heads"],
        n_inner=m["intermediate_size"],
        n_positions=m["n_positions"],
        n_ctx=m["n_ctx"],
        attn_pdrop=m["attn_pdrop"],
        embd_pdrop=m["embd_pdrop"],
        resid_pdrop=m["resid_pdrop"],
        activation_function=m["activation_function"],
        initializer_range=m["initializer_range"],
        layer_norm_epsilon=m["layer_norm_epsilon"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        use_cache=True,
    )
    return GPT2LMHeadModel(cfg)


def load_document_corpus(
    dataset_name: str,
    min_document_words: int,
    max_train_examples: int | None,
    logger: logging.Logger,
) -> tuple[Dataset, int, dict[str, Any]]:
    raw = load_dataset(dataset_name, split="train", trust_remote_code=True)
    if "text" not in raw.column_names:
        raise ValueError(f"Dataset has no text column: {raw.column_names}")

    original_rows = len(raw)

    if max_train_examples is not None:
        raw = raw.select(range(min(max_train_examples, len(raw))))
        logger.warning("DEBUG MODE: retaining only %s raw rows.", f"{len(raw):,}")

    if min_document_words > 0:
        raw = raw.filter(lambda ex: ex.get("text") is not None and count_words(ex["text"]) >= min_document_words)

    # Stable document ids after filtering. These ids are the row ids used by the
    # document-level influence matrix and the final document curriculum.
    def add_fields(example: dict, idx: int) -> dict:
        text = example["text"]
        return {
            "doc_id": idx,
            "word_count": count_words(text),
        }

    raw = raw.map(add_fields, with_indices=True)

    total_words = int(sum(raw["word_count"]))
    stats = {
        "dataset": dataset_name,
        "original_rows_before_debug_cap": original_rows,
        "rows_after_debug_cap_and_filter": len(raw),
        "min_document_words": min_document_words,
        "total_whitespace_words_after_filter": total_words,
        "unit": "raw_document",
    }
    logger.info("=" * 76)
    logger.info("DOCUMENT CORPUS")
    logger.info("Original rows:       %s", f"{original_rows:,}")
    logger.info("Rows after filtering:%s", f"{len(raw):,}")
    logger.info("Min document words:  %s", min_document_words)
    logger.info("Words after filter:  %s", f"{total_words:,}")
    logger.info("=" * 76)
    return raw, total_words, stats


def pack_documents_to_lm_blocks(
    documents: Dataset,
    tokenizer,
    max_seq_length: int,
    tokenize_batch_size: int,
    shuffle_seed: int | None,
    logger: logging.Logger,
) -> tuple[Dataset, dict[str, int]]:
    """
    Pack documents into fixed-length CLM blocks. If shuffle_seed is not None,
    documents are shuffled before packing. If shuffle_seed is None, the incoming
    document order is preserved. EOS is appended after each document.
    """

    docs = documents.shuffle(seed=shuffle_seed) if shuffle_seed is not None else documents
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    stats = {
        "documents": 0,
        "subword_tokens_with_eos": 0,
        "blocks": 0,
        "discarded_remainder": 0,
        "max_seq_length": max_seq_length,
        "shuffled_before_packing": shuffle_seed is not None,
        "shuffle_seed": shuffle_seed,
    }

    def generator() -> Iterator[dict[str, list[int]]]:
        carry: list[int] = []
        for batch in docs.iter(batch_size=tokenize_batch_size):
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
                stats["documents"] += 1
                stats["subword_tokens_with_eos"] += len(ids) + 1

            combined = carry + batch_ids
            complete = (len(combined) // max_seq_length) * max_seq_length
            for start in range(0, complete, max_seq_length):
                block = combined[start:start + max_seq_length]
                stats["blocks"] += 1
                yield {"input_ids": block, "attention_mask": [1] * max_seq_length}
            carry = combined[complete:]

        stats["discarded_remainder"] = len(carry)

    dataset = Dataset.from_generator(generator)
    if len(dataset) == 0:
        raise ValueError("No complete packed training blocks produced.")
    logger.info("=" * 76)
    logger.info("PACKED LM DATASET")
    logger.info("Documents packed:       %s", f"{stats['documents']:,}")
    logger.info("Blocks:                 %s", f"{len(dataset):,}")
    logger.info("Sequence length:        %s", max_seq_length)
    logger.info("Shuffled before packing:%s", stats["shuffled_before_packing"])
    logger.info("Final remainder tokens: %s", f"{stats['discarded_remainder']:,}")
    logger.info("=" * 76)
    return dataset, stats


class WordExposureCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        output_dir: str,
        total_words: int,
        num_epochs: int,
        max_seq_length: int,
        pipeline_step: str,
        extra_metadata: dict[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.total_words = int(total_words)
        self.num_epochs = int(num_epochs)
        self.max_seq_length = int(max_seq_length)
        self.planned = int(total_words * num_epochs)
        self.thresholds = sorted(CHECKPOINT_INTERVALS)
        self.saved: set[int] = set()
        self.total_words_seen = 0.0
        self.pipeline_step = pipeline_step
        self.extra_metadata = extra_metadata or {}

    def estimated(self, state) -> int:
        if not state.max_steps:
            return 0
        return int(min(max(state.global_step / state.max_steps, 0.0), 1.0) * self.planned)

    def on_train_begin(self, args, state, control, **kwargs):
        seen = self.estimated(state)
        self.total_words_seen = float(seen)
        self.saved = {x for x in self.thresholds if x <= seen}
        return control

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        seen = self.estimated(state)
        self.total_words_seen = float(seen)
        for threshold in self.thresholds:
            if threshold <= self.planned and seen >= threshold and threshold not in self.saved:
                self.saved.add(threshold)
                name = f"chck_{threshold // 1_000_000}M"
                out = self.out / name
                model.save_pretrained(out, safe_serialization=True)
                save_portable_tokenizer(self.tokenizer, out)
                log = latest_log(state)
                metadata = {
                    "checkpoint_name": name,
                    "checkpoint_word_exposure_threshold": threshold,
                    "estimated_word_exposure_at_save": seen,
                    "total_words_in_corpus": self.total_words,
                    "planned_epochs": self.num_epochs,
                    "total_planned_word_exposure": self.planned,
                    "global_step": state.global_step,
                    "max_steps": state.max_steps,
                    "epoch": safe_float(state.epoch),
                    "loss": safe_float(log.get("loss", log.get("train_loss"))),
                    "learning_rate": safe_float(log.get("learning_rate")),
                    "model_context_length": model.config.n_positions,
                    "training_sequence_length": self.max_seq_length,
                    "timestamp": datetime.now().isoformat(),
                    "checkpoint_type": "babylm_evaluation_checkpoint",
                    "checkpoint_unit": "estimated_whitespace_word_exposure",
                    "track": "BabyLM-2026-Strict-Small",
                    "pipeline_step": self.pipeline_step,
                    **self.extra_metadata,
                }
                (out / "babylm_checkpoint_metadata.json").write_text(
                    json.dumps(metadata, indent=2, default=str),
                    encoding="utf-8",
                )
                logging.getLogger(__name__).info("Saved %s at estimated exposure %s", name, f"{seen:,}")
        return control


class DetailedCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str, every: int):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.every = int(every)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every == 0:
            log = latest_log(state)
            row = {
                "step": state.global_step,
                "timestamp": datetime.now().isoformat(),
                "loss": safe_float(log.get("loss", log.get("train_loss"))),
                "learning_rate": safe_float(log.get("learning_rate", args.learning_rate)),
                "epoch": safe_float(state.epoch),
                "total_steps": state.max_steps,
            }
            (self.out / f"checkpoint_step_{state.global_step:06d}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
        return control
