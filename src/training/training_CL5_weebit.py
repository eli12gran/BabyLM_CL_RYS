import argparse
import json
import logging
import os
import random
from datetime import datetime
from itertools import chain
from typing import Optional, Tuple

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
    set_seed,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

TRAINING_CONFIG = {
    "model_name": "gpt2_surrogate",

    # GPT-2 small architecture
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,

    "training": {
        "batch_size": 16,
        "gradient_accumulation_steps": 8,   # effective batch = 128 sequences
        "learning_rate": 1e-3,
        "num_epochs": 10,                   
        "warmup_steps": 200,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "seed": 42,
        "max_grad_norm": 1.0,
    },

    "data": {
        "max_seq_length": 128,          
        "tokenize_batch_size": 1000,
        "chunk_batch_size": 1000,
    },

    "output_dir": "./model_surrogate",
    "babylm_checkpoint_dir": "./babylm_checkpoints_surrogate",
    "detailed_checkpoint_dir": "./checkpoints_detailed_surrogate",

    # BabyLM Checkpoints intervals
    "checkpoint_intervals": [
        1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
        6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
        20_000_000, 30_000_000, 40_000_000, 50_000_000,
        60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000,
    ],
}

TOKENIZER_NAME = "gpt2"


# ============================================================================
# ARGS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GPT-2 model with customized checkpointing."
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="BabyLM-community/BabyLM-2026-Strict-Small",
        help="Nombre o ruta del dataset de Hugging Face a utilizar."
    )
    parser.add_argument("--output_dir", type=str, default=TRAINING_CONFIG["output_dir"])
    parser.add_argument("--babylm_checkpoint_dir", type=str, default=TRAINING_CONFIG["babylm_checkpoint_dir"])
    parser.add_argument("--detailed_checkpoint_dir", type=str, default=TRAINING_CONFIG["detailed_checkpoint_dir"])
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=TRAINING_CONFIG["data"]["max_seq_length"],
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Optional HF Trainer checkpoint path to resume from.",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Optional debug cap on the number of raw examples loaded.",
    )
    return parser.parse_args()


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


# ============================================================================
# DATA
# ============================================================================

def count_whitespace_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def load_and_prepare_dataset(dataset_name: str, tokenizer, max_seq_length: int, seed: int, tokenize_batch_size: int = 1000, chunk_batch_size: int = 1000, max_train_examples: Optional[int] = None) -> Tuple[Dataset, int]:
    logger.info("=" * 70)
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info("=" * 70)

    raw_dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)

    if max_train_examples is not None:
        logger.warning(f"DEBUG CAP: limiting to {max_train_examples:,} raw examples.")
        raw_dataset = raw_dataset.select(range(min(max_train_examples, len(raw_dataset))))

    # Count official BabyLM words (whitespace tokens)
    total_words = sum(count_whitespace_words(ex.get("text", "")) for ex in raw_dataset)
    logger.info(f"Total whitespace words in corpus: {total_words:,}")

    # Random shuffle
    # raw_dataset = raw_dataset.shuffle(seed=seed)
    # logger.info(f"Dataset shuffled (seed={seed}).")

    # Curriculum Learning: Sort by complexity (easiest first)
    raw_dataset = raw_dataset.sort("complexity_age_interval")
    logger.info("Dataset sorted by 'complexity_age_interval' for Curriculum Learning.")

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_attention_mask=False,
        )

    tokenized = raw_dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    def chunk_fn(examples):
        all_ids = list(chain.from_iterable(examples["input_ids"]))
        total_len = (len(all_ids) // max_seq_length) * max_seq_length
        if total_len == 0:
            return {"input_ids": []}
        return {
            "input_ids": [
                all_ids[i: i + max_seq_length]
                for i in range(0, total_len, max_seq_length)
            ]
        }

    chunked = tokenized.map(
        chunk_fn,
        batched=True,
        batch_size=chunk_batch_size,
        remove_columns=tokenized.column_names,
        desc="Chunking",
    )

    logger.info(f"✓ {len(chunked):,} fixed-length sequences of {max_seq_length} tokens")
    return chunked, total_words


# ============================================================================
# MODEL
# ============================================================================

def create_model(tokenizer, max_seq_length: int) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=max_seq_length,
        n_embd=TRAINING_CONFIG["n_embd"],
        n_layer=TRAINING_CONFIG["n_layer"],
        n_head=TRAINING_CONFIG["n_head"],
        n_inner=TRAINING_CONFIG["n_inner"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created GPT-2 model with {num_params:,} parameters")
    return model


# ============================================================================
# CALLBACKS
# ============================================================================

class WordExposureCheckpointCallback(TrainerCallback):
    """Guarda los checkpoints requeridos por BabyLM basados en palabras vistas."""
    def __init__(self, tokenizer, total_words_in_corpus: int, train_dataset_size: int, max_seq_length: int, checkpoint_intervals: list, output_dir: str):
        self.tokenizer = tokenizer
        self.total_words_in_corpus = total_words_in_corpus
        self.train_dataset_size = train_dataset_size
        self.max_seq_length = max_seq_length
        self.checkpoint_intervals = checkpoint_intervals
        self.output_dir = output_dir

        self.total_words_seen = 0.0
        self.checkpoints_saved = set()
        os.makedirs(self.output_dir, exist_ok=True)

    def _words_per_optimizer_step(self, args) -> float:
        seqs_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
        return (seqs_per_step / self.train_dataset_size) * self.total_words_in_corpus

    def _bpe_tokens_per_step(self, args) -> int:
        return args.per_device_train_batch_size * args.gradient_accumulation_steps * self.max_seq_length

    def _checkpoint_name(self, words: int) -> str:
        if words < 1_000_000:
            return f"chck_{words // 1_000}K"
        return f"chck_{words // 1_000_000}M"

    def _save(self, model, args, state, words: int) -> None:
        name = self._checkpoint_name(words)
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(save_dir)

        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump(args.to_dict(), f, indent=2)

        latest_log = state.log_history[-1] if state.log_history else {}
        metadata = {
            "checkpoint_name": name,
            "checkpoint_words": words,
            "checkpoint_words_millions": words / 1_000_000,
            "actual_words_seen_estimate": self.total_words_seen,
            "total_words_in_corpus": self.total_words_in_corpus,
            "approx_bpe_tokens_seen": state.global_step * self._bpe_tokens_per_step(args),
            "global_step": state.global_step,
            "epoch": safe_float(state.epoch),
            "loss": safe_float(latest_log.get("loss")),
            "learning_rate": safe_float(latest_log.get("learning_rate")),
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(save_dir, "babylm_checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved BabyLM exposure checkpoint: {name} (Words threshold: {words:,})")

    def on_train_begin(self, args, state, control, **kwargs):
        wps = self._words_per_optimizer_step(args)
        self.total_words_seen = state.global_step * wps
        self.checkpoints_saved = {w for w in self.checkpoint_intervals if w <= self.total_words_seen}

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control

        wps = self._words_per_optimizer_step(args)
        self.total_words_seen = state.global_step * wps

        for w in self.checkpoint_intervals:
            if self.total_words_seen >= w and w not in self.checkpoints_saved:
                self.checkpoints_saved.add(w)
                self._save(model, args, state, w)

        return control


class DetailedCheckpointCallback(TrainerCallback):
    """Guarda logs ligeros en JSON cada N pasos de optimización para seguridad."""
    def __init__(self, checkpoint_dir: str, save_every_n_steps: int = 500):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.checkpoint_info = {}
        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_every_n_steps == 0 and state.global_step > 0:
            self._save(state.global_step, state, args)

    def _save(self, step: int, state, args) -> None:
        latest_log = state.log_history[-1] if state.log_history else {}
        data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "loss": safe_float(latest_log.get("loss")),
            "learning_rate": safe_float(latest_log.get("learning_rate", args.learning_rate)),
            "epoch": safe_float(state.epoch),
        }
        with open(os.path.join(self.checkpoint_dir, f"checkpoint_step_{step:06d}.json"), "w") as f:
            json.dump(data, f, indent=2)


# ============================================================================
# TRAINING SETUP
# ============================================================================

def setup_training(model: GPT2LMHeadModel, train_dataset: Dataset, tokenizer, total_words_in_corpus: int, max_seq_length: int) -> Tuple[Trainer, WordExposureCheckpointCallback]:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["training"]["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["training"]["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["training"]["gradient_accumulation_steps"],
        logging_steps=TRAINING_CONFIG["training"]["logging_steps"],
        save_strategy="steps",
        save_steps=500,  # <-- ESTE GUARDA LOS CHECKPOINTS DE SEGURIDAD PARA RETOMAR D1
        learning_rate=TRAINING_CONFIG["training"]["learning_rate"],
        warmup_steps=TRAINING_CONFIG["training"]["warmup_steps"],
        weight_decay=TRAINING_CONFIG["training"]["weight_decay"],
        max_grad_norm=TRAINING_CONFIG["training"]["max_grad_norm"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim="adamw_torch",
        gradient_checkpointing=True,
        seed=TRAINING_CONFIG["training"]["seed"],
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    word_exposure_callback = WordExposureCheckpointCallback(
        tokenizer=tokenizer,
        total_words_in_corpus=total_words_in_corpus,
        train_dataset_size=len(train_dataset),
        max_seq_length=max_seq_length,
        checkpoint_intervals=TRAINING_CONFIG["checkpoint_intervals"],
        output_dir=TRAINING_CONFIG["babylm_checkpoint_dir"],
    )

    detailed_callback = DetailedCheckpointCallback(
        checkpoint_dir=TRAINING_CONFIG["detailed_checkpoint_dir"],
        save_every_n_steps=500,
    )

    # Nota: Ya no se incluye el epoch_callback aquí.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[word_exposure_callback, detailed_callback],
    )

    return trainer, word_exposure_callback


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    TRAINING_CONFIG["output_dir"] = args.output_dir
    TRAINING_CONFIG["babylm_checkpoint_dir"] = args.babylm_checkpoint_dir
    TRAINING_CONFIG["detailed_checkpoint_dir"] = args.detailed_checkpoint_dir
    TRAINING_CONFIG["data"]["max_seq_length"] = args.max_seq_length

    seed = TRAINING_CONFIG["training"]["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Dataset seleccionado: {args.dataset_name}")

    logger.info("[1/4] Loading GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("[2/4] Loading and preparing dataset")
    train_dataset, total_words = load_and_prepare_dataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        seed=seed,
        tokenize_batch_size=TRAINING_CONFIG["data"]["tokenize_batch_size"],
        chunk_batch_size=TRAINING_CONFIG["data"]["chunk_batch_size"],
        max_train_examples=args.max_train_examples,
    )

    logger.info("[3/4] Creating GPT-2 model")
    model = create_model(tokenizer, max_seq_length=args.max_seq_length)

    logger.info("[4/4] Setting up Trainer")
    trainer, word_callback = setup_training(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        total_words_in_corpus=total_words,
        max_seq_length=args.max_seq_length,
    )

    train_kwargs = {}
    if args.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint

    logger.info("STARTING TRAINING...")
    trainer.train(**train_kwargs)
    logger.info("TRAINING COMPLETE.")


if __name__ == "__main__":
    main()