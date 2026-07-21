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

import numpy as np
import torch
from datasets import Dataset, load_dataset
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CONFIG: dict[str, Any] = {
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
        "tokenizer_name": "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small",
    },
    "curriculum": {"bin_size": 1_000, "order": "descending", "static": True},
    "output_dir": "./model_ticl_descending",
    "babylm_checkpoint_dir": "./babylm_checkpoints_ticl_descending",
    "detailed_checkpoint_dir": "./checkpoints_detailed_ticl_descending",
    "checkpoint_intervals": [
        1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
        6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
        20_000_000, 30_000_000, 40_000_000, 50_000_000,
        60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000,
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TICL descending curriculum model.")
    p.add_argument("--influence_matrix", default="./influence_output/influence_matrix.npy")
    p.add_argument("--influence_metadata", default=None)
    p.add_argument("--bin_size", type=int, default=CONFIG["curriculum"]["bin_size"])
    p.add_argument("--output_dir", default=CONFIG["output_dir"])
    p.add_argument("--babylm_checkpoint_dir", default=CONFIG["babylm_checkpoint_dir"])
    p.add_argument("--detailed_checkpoint_dir", default=CONFIG["detailed_checkpoint_dir"])
    p.add_argument("--max_seq_length", type=int, default=CONFIG["data"]["max_seq_length"])
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", default=None)
    return p.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_log(state) -> dict[str, Any]:
    for row in reversed(state.log_history):
        if any(k in row for k in ("loss", "train_loss", "learning_rate")):
            return row
    return {}


def make_training_arguments(**kwargs) -> TrainingArguments:
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    accepted = {k: v for k, v in kwargs.items() if k in supported}
    omitted = sorted(set(kwargs) - set(accepted))
    if omitted:
        logger.warning("Unsupported TrainingArguments omitted: %s", ", ".join(omitted))
    return TrainingArguments(**accepted)


def select_precision() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    bf16 = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    return bf16, not bf16


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


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def reconstruct_dataset(tokenizer, max_examples: int | None = None) -> tuple[Dataset, int, dict[str, int]]:
    data_cfg = CONFIG["data"]
    raw = load_dataset(data_cfg["hf_dataset"], split="train", trust_remote_code=True)
    if "text" not in raw.column_names:
        raise ValueError(f"Dataset has no text column: {raw.column_names}")
    if max_examples is not None:
        raw = raw.select(range(min(max_examples, len(raw))))
    total_words = sum(count_words(text) for text in raw["text"])
    raw = raw.shuffle(seed=CONFIG["training"]["seed"])
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no EOS token.")
    seq_len = data_cfg["max_seq_length"]
    stats = {"documents": 0, "tokens_with_eos": 0, "blocks": 0, "discarded_remainder": 0}

    def generator() -> Iterator[dict[str, list[int]]]:
        carry: list[int] = []
        for batch in raw.iter(batch_size=data_cfg["tokenize_batch_size"]):
            encoded = tokenizer(
                batch["text"], add_special_tokens=False, truncation=False,
                padding=False, return_attention_mask=False,
            )
            batch_ids: list[int] = []
            for ids in encoded["input_ids"]:
                batch_ids.extend(ids)
                batch_ids.append(eos_id)
                stats["documents"] += 1
                stats["tokens_with_eos"] += len(ids) + 1
            combined = carry + batch_ids
            complete = (len(combined) // seq_len) * seq_len
            for start in range(0, complete, seq_len):
                block = combined[start:start + seq_len]
                stats["blocks"] += 1
                yield {"input_ids": block, "attention_mask": [1] * seq_len}
            carry = combined[complete:]
        stats["discarded_remainder"] = len(carry)

    dataset = Dataset.from_generator(generator)
    if not len(dataset):
        raise ValueError("No complete sequences produced.")
    logger.info("Reconstructed %s sequences; final remainder=%s tokens", f"{len(dataset):,}", stats["discarded_remainder"])
    return dataset, total_words, stats


def load_and_validate_influence(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    phi = np.load(path, mmap_mode="r")
    if phi.ndim != 2 or phi.shape[1] == 0:
        raise ValueError(f"Expected 2D influence matrix, found {phi.shape}")
    if not np.isfinite(phi).all():
        raise ValueError("Influence matrix contains NaN or infinite values.")
    return phi


def build_curriculum(phi: np.ndarray, bin_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")
    mean_score = np.asarray(phi.mean(axis=1), dtype=np.float32)
    indices = np.argsort(-mean_score, kind="stable").astype(np.int64)
    rng = np.random.default_rng(seed)
    for start in range(0, len(indices), bin_size):
        rng.shuffle(indices[start:start + bin_size])
    if not np.array_equal(np.sort(indices), np.arange(len(indices))):
        raise RuntimeError("Curriculum is not a permutation of dataset rows.")
    meta = {
        "n_sequences": int(phi.shape[0]),
        "n_surrogate_checkpoints": int(phi.shape[1]),
        "score": "mean influence across checkpoint columns",
        "sort": "descending stable sort",
        "bin_size": bin_size,
        "within_bin_shuffle": True,
        "shuffle_seed": seed,
        "static_across_epochs": True,
        "score_min": float(mean_score.min()),
        "score_max": float(mean_score.max()),
        "score_mean": float(mean_score.mean()),
        "score_std": float(mean_score.std()),
    }
    return indices, mean_score, meta


def validate_metadata(path: Path, phi: np.ndarray) -> dict[str, Any] | None:
    if not path.is_file():
        logger.warning("No influence metadata found at %s", path)
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": CONFIG["data"]["hf_dataset"],
        "tokenizer": CONFIG["data"]["tokenizer_name"],
        "max_seq_length": CONFIG["data"]["max_seq_length"],
        "seed": CONFIG["training"]["seed"],
    }
    errors = []
    for key, value in expected.items():
        if key in meta and meta[key] != value:
            errors.append(f"{key}: {meta[key]!r} != {value!r}")
    if meta.get("n_sequences") not in (None, int(phi.shape[0])):
        errors.append(f"n_sequences: {meta.get('n_sequences')} != {phi.shape[0]}")
    if errors:
        raise ValueError("Influence metadata mismatch:\n- " + "\n- ".join(errors))
    return meta


def create_model(tokenizer) -> GPT2LMHeadModel:
    m = CONFIG["model"]
    cfg = GPT2Config(
        vocab_size=len(tokenizer), n_embd=m["hidden_size"],
        n_layer=m["num_hidden_layers"], n_head=m["num_attention_heads"],
        n_inner=m["intermediate_size"], n_positions=m["n_positions"],
        n_ctx=m["n_ctx"], attn_pdrop=m["attn_pdrop"],
        embd_pdrop=m["embd_pdrop"], resid_pdrop=m["resid_pdrop"],
        activation_function=m["activation_function"],
        initializer_range=m["initializer_range"],
        layer_norm_epsilon=m["layer_norm_epsilon"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True, use_cache=True,
    )
    model = GPT2LMHeadModel(cfg)
    logger.info("Fresh model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    return model


class WordExposureCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer, total_words: int):
        self.tokenizer = tokenizer
        self.total_words = total_words
        self.epochs = CONFIG["training"]["num_epochs"]
        self.planned = int(total_words * self.epochs)
        self.thresholds = sorted(CONFIG["checkpoint_intervals"])
        self.out = Path(CONFIG["babylm_checkpoint_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.saved: set[int] = set()
        self.total_words_seen = 0.0

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
            if threshold <= self.planned and threshold <= seen and threshold not in self.saved:
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
                    "planned_epochs": self.epochs,
                    "total_planned_word_exposure": self.planned,
                    "global_step": state.global_step,
                    "max_steps": state.max_steps,
                    "epoch": safe_float(state.epoch),
                    "loss": safe_float(log.get("loss", log.get("train_loss"))),
                    "learning_rate": safe_float(log.get("learning_rate")),
                    "model_context_length": model.config.n_positions,
                    "training_sequence_length": CONFIG["data"]["max_seq_length"],
                    "timestamp": datetime.now().isoformat(),
                    "pipeline_step": "ticl_descending_curriculum",
                    "curriculum_order": "descending_mean_influence",
                    "static_across_epochs": True,
                }
                (out / "babylm_checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                logger.info("Saved %s at estimated exposure %s", name, f"{seen:,}")
        return control


class DetailedCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.out = Path(CONFIG["detailed_checkpoint_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.every = CONFIG["training"]["detailed_checkpoint_every_n_steps"]

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
            (self.out / f"checkpoint_step_{state.global_step:06d}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        return control


class StaticCurriculumTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return None if dataset is None else SequentialSampler(dataset)


def build_trainer(model, tokenizer, dataset: Dataset, total_words: int):
    t = CONFIG["training"]
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    bf16, fp16 = select_precision()
    args = make_training_arguments(
        output_dir=CONFIG["output_dir"], num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"],
        adam_beta1=t["adam_beta1"], adam_beta2=t["adam_beta2"],
        adam_epsilon=t["adam_epsilon"], max_grad_norm=t["max_grad_norm"],
        logging_strategy="steps", logging_steps=t["logging_steps"],
        logging_first_step=True, save_strategy="steps", save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"], optim="adamw_torch",
        seed=t["seed"], data_seed=t["seed"], dataloader_drop_last=True,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=t["dataloader_num_workers"], bf16=bf16, fp16=fp16,
        gradient_checkpointing=t["gradient_checkpointing"],
        remove_unused_columns=False, report_to="none", group_by_length=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    word_cb = WordExposureCheckpointCallback(tokenizer, total_words)
    trainer = StaticCurriculumTrainer(
        model=model, args=args, train_dataset=dataset, data_collator=collator,
        callbacks=[word_cb, DetailedCheckpointCallback()],
    )
    return trainer, word_cb


def main() -> None:
    args = parse_args()
    CONFIG["output_dir"] = args.output_dir
    CONFIG["babylm_checkpoint_dir"] = args.babylm_checkpoint_dir
    CONFIG["detailed_checkpoint_dir"] = args.detailed_checkpoint_dir
    CONFIG["data"]["max_seq_length"] = args.max_seq_length
    CONFIG["curriculum"]["bin_size"] = args.bin_size

    if CONFIG["data"]["max_seq_length"] > CONFIG["model"]["n_positions"]:
        raise ValueError("max_seq_length exceeds model positional capacity.")
    set_seed(CONFIG["training"]["seed"])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["data"]["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) != CONFIG["model"]["expected_vocab_size"]:
        raise ValueError(f"Expected tokenizer length 16384, got {len(tokenizer)}")

    influence_path = Path(args.influence_matrix)
    metadata_path = Path(args.influence_metadata) if args.influence_metadata else influence_path.with_name("influence_metadata.json")
    phi = load_and_validate_influence(influence_path)
    source_metadata = validate_metadata(metadata_path, phi)

    base_dataset, total_words, preprocessing = reconstruct_dataset(tokenizer, args.max_train_examples)
    if len(base_dataset) != phi.shape[0]:
        raise ValueError(
            f"Dataset/influence mismatch: reconstructed {len(base_dataset):,} sequences, "
            f"matrix has {phi.shape[0]:,} rows. Use the identical tokenizer, seed, EOS, "
            "carry-based packing, sequence length, and debug cap."
        )

    indices, scores, curriculum_meta = build_curriculum(
        phi, CONFIG["curriculum"]["bin_size"], CONFIG["training"]["seed"]
    )
    out = Path(CONFIG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "curriculum_indices.npy", indices)
    np.save(out / "mean_influence_scores.npy", scores)
    full_meta = {
        "created_at": datetime.now().isoformat(),
        "influence_matrix": str(influence_path),
        "influence_metadata": source_metadata,
        "dataset": CONFIG["data"]["hf_dataset"],
        "tokenizer": CONFIG["data"]["tokenizer_name"],
        "preprocessing": preprocessing,
        **curriculum_meta,
    }
    (out / "curriculum_metadata.json").write_text(json.dumps(full_meta, indent=2, default=str), encoding="utf-8")

    curriculum_dataset = base_dataset.select(indices.tolist())
    model = create_model(tokenizer)
    trainer, word_cb = build_trainer(model, tokenizer, curriculum_dataset, total_words)

    world = max(int(trainer.args.world_size), 1)
    effective_batch = CONFIG["training"]["batch_size"] * CONFIG["training"]["gradient_accumulation_steps"] * world
    logger.info("Only intervention: sequence order")
    logger.info("Static descending curriculum with within-bin shuffle; bin=%s", args.bin_size)
    logger.info("Sequences=%s | effective batch=%s | epochs=%s", f"{len(curriculum_dataset):,}", effective_batch, CONFIG["training"]["num_epochs"])
    logger.info("LR=%s | warmup_ratio=%s | scheduler=%s", CONFIG["training"]["learning_rate"], CONFIG["training"]["warmup_ratio"], CONFIG["training"]["lr_scheduler_type"])

    resume = args.resume_from_checkpoint
    if resume is None and CONFIG["training"]["auto_resume"] and os.path.isdir(CONFIG["output_dir"]):
        resume = get_last_checkpoint(CONFIG["output_dir"])
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    final_dir = out / "final"
    trainer.save_model(final_dir)
    save_portable_tokenizer(tokenizer, final_dir)
    final_metadata = {
        "completed_at": datetime.now().isoformat(),
        "pipeline_role": "final_ticl_descending_curriculum_model",
        "training_order": "descending_mean_influence_within_bin_shuffle",
        "static_across_epochs": True,
        "dataset": CONFIG["data"]["hf_dataset"],
        "track": "BabyLM-2026-Strict-Small",
        "total_words_in_corpus": total_words,
        "planned_word_exposure": total_words * CONFIG["training"]["num_epochs"],
        "total_sequences": len(curriculum_dataset),
        "final_global_step": trainer.state.global_step,
        "final_epoch": safe_float(trainer.state.epoch),
        "curriculum_metadata": full_meta,
        "training_config": CONFIG,
    }
    (final_dir / "final_model_metadata.json").write_text(json.dumps(final_metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Training complete. Saved %s BabyLM checkpoints.", len(word_cb.saved))


if __name__ == "__main__":
    main()