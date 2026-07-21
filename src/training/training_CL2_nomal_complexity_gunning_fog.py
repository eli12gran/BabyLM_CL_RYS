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
from datasets import load_dataset, Dataset
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
    "model_name": "gpt2_cl_gunning_fog",

    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,


    "n_positions": 1024,
    "n_ctx": 1024,

    "attn_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "activation_function": "gelu_new",
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-5,

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
    },

    "data": {
        "max_seq_length": 128,
        "tokenize_batch_size": 1000,
        "chunk_batch_size": 1000,
    },

    "output_dir": "./model_cl_gunning_fog",
    "babylm_checkpoint_dir": "./babylm_checkpoints_cl_gunning_fog",
    "detailed_checkpoint_dir": "./checkpoints_detailed_cl_gunning_fog",

    "checkpoint_intervals": [
        1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
        6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
        20_000_000, 30_000_000, 40_000_000, 50_000_000,
        60_000_000, 70_000_000, 80_000_000, 90_000_000,
        100_000_000,
    ],
}

HF_DATASET       = "eligran12/babylm_complexity_metrics"
COMPLEXITY_COL   = "complexity_gunning_fog"
TOKENIZER_NAME   = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"


# ============================================================================
# ARGS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GPT-2 with Curriculum Learning ordered by Gunning Fog (easy→hard)."
    )
    parser.add_argument("--output_dir",             type=str, default=TRAINING_CONFIG["output_dir"])
    parser.add_argument("--babylm_checkpoint_dir",  type=str, default=TRAINING_CONFIG["babylm_checkpoint_dir"])
    parser.add_argument("--detailed_checkpoint_dir",type=str, default=TRAINING_CONFIG["detailed_checkpoint_dir"])
    parser.add_argument("--max_seq_length",         type=int, default=TRAINING_CONFIG["data"]["max_seq_length"])
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
    parser.add_argument(
        "--drop_fog_nulls",
        action="store_true",
        default=True,
        help="Drop rows where complexity_gunning_fog is null (default: True).",
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
# DATA  —  load, sort easy→hard, tokenise, chunk
# ============================================================================

def count_whitespace_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def load_and_prepare_dataset(
    tokenizer,
    max_seq_length: int,
    tokenize_batch_size: int = 1000,
    chunk_batch_size: int = 1000,
    max_train_examples: Optional[int] = None,
    drop_fog_nulls: bool = True,
) -> Tuple[Dataset, int]:
    """
    Load data, sort ascending by
    complexity_gunning_fog (easy → hard), then tokenise and chunk.

    The sort order is preserved through the entire pipeline so the Trainer
    sees examples in curriculum order from the very first step of epoch 1.
    Subsequent epochs also replay in the same sorted order, which is the
    standard 'fixed curriculum' approach.
    """
    logger.info("=" * 70)
    logger.info(f"Loading {HF_DATASET}")
    logger.info("=" * 70)

    raw_dataset = load_dataset(HF_DATASET, split="train", trust_remote_code=True)
    logger.info(f"Raw rows loaded: {len(raw_dataset):,}")
    logger.info(f"Columns: {raw_dataset.column_names}")

    if max_train_examples is not None:
        logger.warning(f"DEBUG CAP: limiting to {max_train_examples:,} raw examples.")
        raw_dataset = raw_dataset.select(range(min(max_train_examples, len(raw_dataset))))

    # ── Drop / fill null Gunning Fog values ──────────────────────────────────
    if drop_fog_nulls:
        before = len(raw_dataset)
        raw_dataset = raw_dataset.filter(
            lambda ex: ex[COMPLEXITY_COL] is not None
                       and ex[COMPLEXITY_COL] == ex[COMPLEXITY_COL],  # NaN check
            desc=f"Filtering null {COMPLEXITY_COL}",
        )
        dropped = before - len(raw_dataset)
        logger.info(f"Dropped {dropped:,} rows with null {COMPLEXITY_COL} ({before:,} → {len(raw_dataset):,})")

    # ── Sort by Gunning Fog ascending (easy → hard) ──────────────────────────
    logger.info(f"Sorting {len(raw_dataset):,} rows by {COMPLEXITY_COL} ascending (easy → hard) …")
    fog_scores = raw_dataset[COMPLEXITY_COL]
    sorted_indices = sorted(range(len(fog_scores)), key=lambda i: fog_scores[i])
    raw_dataset = raw_dataset.select(sorted_indices)

    fog_min  = fog_scores[sorted_indices[0]]
    fog_max  = fog_scores[sorted_indices[-1]]
    fog_median = sorted(fog_scores)[len(fog_scores) // 2]
    logger.info(f"Gunning Fog range after sort: min={fog_min:.2f}  median={fog_median:.2f}  max={fog_max:.2f}")
    logger.info("Dataset sorted — curriculum order locked in.")

    # ── Word count (for BabyLM exposure tracking) ────────────────────────────
    total_words = sum(count_whitespace_words(ex.get("text", "")) for ex in raw_dataset)
    logger.info(f"Total whitespace words in corpus: {total_words:,}")

    # ── Tokenise ─────────────────────────────────────────────────────────────
    def tokenize_fn(examples):
        encoded = tokenizer(
            examples["text"],
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )

        eos_id = tokenizer.eos_token_id

        encoded["input_ids"] = [
            token_ids + [eos_id]
            for token_ids in encoded["input_ids"]
        ]

        return encoded

    tokenized = raw_dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    # ── Chunk into fixed-length sequences ────────────────────────────────────
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
    logger.info("Curriculum order preserved through tokenisation and chunking.")
    return chunked, total_words


# ============================================================================
# MODEL
# ============================================================================

def create_model(tokenizer, max_seq_length: int) -> GPT2LMHeadModel:
    config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=TRAINING_CONFIG["n_positions"],
    n_ctx=TRAINING_CONFIG["n_ctx"],
    n_embd=TRAINING_CONFIG["n_embd"],
    n_layer=TRAINING_CONFIG["n_layer"],
    n_head=TRAINING_CONFIG["n_head"],
    n_inner=TRAINING_CONFIG["n_inner"],
    activation_function="gelu_new",
    attn_pdrop=0.1,
    embd_pdrop=0.1,
    resid_pdrop=0.1,
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    tie_word_embeddings=True,
    )
    model = GPT2LMHeadModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created GPT-2 CL model with {num_params:,} parameters")
    logger.info(f"Vocab size: {config.vocab_size:,} | Seq length: {max_seq_length}")
    return model


# ============================================================================
# CALLBACKS
# ============================================================================

class WordExposureCheckpointCallback(TrainerCallback):

    def __init__(
        self,
        tokenizer,
        total_words_in_corpus: int,
        train_dataset_size: int,
        max_seq_length: int,
        checkpoint_intervals: list,
        output_dir: str,
    ):
        self.tokenizer              = tokenizer
        self.total_words_in_corpus  = total_words_in_corpus
        self.train_dataset_size     = train_dataset_size
        self.max_seq_length         = max_seq_length
        self.checkpoint_intervals   = checkpoint_intervals
        self.output_dir             = output_dir

        self.total_words_seen   = 0.0
        self.checkpoints_saved  = set()
        os.makedirs(self.output_dir, exist_ok=True)

    def _words_per_optimizer_step(self, args) -> float:
        seqs_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
        return (seqs_per_step / self.train_dataset_size) * self.total_words_in_corpus

    def _bpe_tokens_per_step(self, args) -> int:
        return (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.max_seq_length
        )

    def _checkpoint_name(self, words: int) -> str:
        if words < 1_000_000:
            return f"chck_{words // 1_000}K"
        return f"chck_{words // 1_000_000}M"

    def _save(self, model, args, state, words: int) -> None:
        name     = self._checkpoint_name(words)
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(save_dir)

        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump(args.to_dict(), f, indent=2)

        latest_log = state.log_history[-1] if state.log_history else {}
        metadata = {
            "checkpoint_name":              name,
            "checkpoint_words":             words,
            "checkpoint_words_millions":    words / 1_000_000,
            "actual_words_seen_estimate":   self.total_words_seen,
            "total_words_in_corpus":        self.total_words_in_corpus,
            "approx_bpe_tokens_seen":       state.global_step * self._bpe_tokens_per_step(args),
            "global_step":                  state.global_step,
            "epoch":                        safe_float(state.epoch),
            "loss":                         safe_float(latest_log.get("loss")),
            "learning_rate":                safe_float(latest_log.get("learning_rate")),
            "max_seq_length":               self.max_seq_length,
            "per_device_train_batch_size":  args.per_device_train_batch_size,
            "gradient_accumulation_steps":  args.gradient_accumulation_steps,
            "timestamp":                    datetime.now().isoformat(),
            "checkpoint_type":              "babylm_evaluation_checkpoint",
            "model_architecture":           "GPT-2",
            "objective":                    "causal_language_modeling",
            "exposure_unit":                "whitespace_words",
            "track":                        "BabyLM-2026-Strict-Small",
            "pipeline_step":                "curriculum_learning_gunning_fog",
            "curriculum_metric":            COMPLEXITY_COL,
            "curriculum_order":             "easy_to_hard",
        }
        with open(os.path.join(save_dir, "babylm_checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 70)
        logger.info(f"Saved BabyLM exposure checkpoint: {name}")
        logger.info(f"  Words threshold: {words:,} | Estimated seen: {self.total_words_seen:,.0f}")
        logger.info(f"  Global step: {state.global_step} | Path: {save_dir}")
        logger.info("=" * 70)

    def on_train_begin(self, args, state, control, **kwargs):
        wps = self._words_per_optimizer_step(args)
        self.total_words_seen  = state.global_step * wps
        self.checkpoints_saved = {
            w for w in self.checkpoint_intervals if w <= self.total_words_seen
        }
        logger.info("=" * 70)
        logger.info("WORD-EXPOSURE CHECKPOINT CALLBACK INITIALIZED")
        logger.info(f"  Words per epoch (corpus): {self.total_words_in_corpus:,}")
        logger.info(f"  Train sequences:          {self.train_dataset_size:,}")
        logger.info(f"  Words per optimizer step: {wps:,.1f}")
        logger.info(f"  Already passed thresholds:{len(self.checkpoints_saved)}")
        logger.info("=" * 70)

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
    """Saves lightweight JSON progress logs every N optimizer steps."""

    def __init__(self, checkpoint_dir: str, save_every_n_steps: int = 500):
        self.checkpoint_dir     = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.checkpoint_info    = {}
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"DetailedCheckpointCallback: JSON logs every {save_every_n_steps} steps")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_every_n_steps == 0 and state.global_step > 0:
            self._save(state.global_step, state, args)

    def _save(self, step: int, state, args) -> None:
        latest_log = state.log_history[-1] if state.log_history else {}
        data = {
            "step":                         step,
            "timestamp":                    datetime.now().isoformat(),
            "loss":                         safe_float(latest_log.get("loss")),
            "learning_rate":                safe_float(latest_log.get("learning_rate", args.learning_rate)),
            "epoch":                        safe_float(state.epoch),
            "total_steps":                  state.max_steps,
            "batch_size":                   args.per_device_train_batch_size,
            "gradient_accumulation_steps":  args.gradient_accumulation_steps,
            "effective_batch_size":         args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "progress_percent":             (step / state.max_steps * 100) if state.max_steps else 0.0,
        }
        with open(os.path.join(self.checkpoint_dir, f"checkpoint_step_{step:06d}.json"), "w") as f:
            json.dump(data, f, indent=2)
        self.checkpoint_info[step] = data
        with open(os.path.join(self.checkpoint_dir, "checkpoint_log.json"), "w") as f:
            json.dump(self.checkpoint_info, f, indent=2)

        loss_text  = f"{data['loss']:.4f}"  if data["loss"]  is not None else "NA"
        epoch_text = f"{data['epoch']:.2f}" if data["epoch"] is not None else "NA"
        logger.info(
            f"JSON checkpoint {step}: Loss={loss_text} | Epoch={epoch_text} | "
            f"Progress={data['progress_percent']:.1f}%"
        )


# ============================================================================
# TRAINING SETUP
# ============================================================================

def setup_training(
    model: GPT2LMHeadModel,
    train_dataset: Dataset,
    tokenizer,
    total_words_in_corpus: int,
    max_seq_length: int,
) -> Tuple[Trainer, WordExposureCheckpointCallback]:

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    training_args = TrainingArguments(
    output_dir=TRAINING_CONFIG["output_dir"],

    num_train_epochs=TRAINING_CONFIG["training"]["num_epochs"],

    per_device_train_batch_size=(
        TRAINING_CONFIG["training"]["batch_size"]
    ),
    gradient_accumulation_steps=(
        TRAINING_CONFIG["training"]["gradient_accumulation_steps"]
    ),

    learning_rate=TRAINING_CONFIG["training"]["learning_rate"],
    lr_scheduler_type=(
        TRAINING_CONFIG["training"]["lr_scheduler_type"]
    ),
    warmup_ratio=TRAINING_CONFIG["training"]["warmup_ratio"],

    weight_decay=TRAINING_CONFIG["training"]["weight_decay"],
    adam_beta1=TRAINING_CONFIG["training"]["adam_beta1"],
    adam_beta2=TRAINING_CONFIG["training"]["adam_beta2"],
    adam_epsilon=TRAINING_CONFIG["training"]["adam_epsilon"],

    max_grad_norm=TRAINING_CONFIG["training"]["max_grad_norm"],

    logging_strategy="steps",
    logging_steps=TRAINING_CONFIG["training"]["logging_steps"],
    logging_first_step=True,

    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    optim="adamw_torch",

    seed=TRAINING_CONFIG["training"]["seed"],
    data_seed=TRAINING_CONFIG["training"]["seed"],

    dataloader_drop_last=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=torch.cuda.is_available(),

    fp16=(
        torch.cuda.is_available()
        and not (
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
    ),
    bf16=(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ),

    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[word_exposure_callback, detailed_callback],
    )

    # Patch the internal sampler to preserve sort order (no shuffle)
    # HF Trainer uses a RandomSampler by default; we replace it with a
    # SequentialSampler so the curriculum order survives into the DataLoader.
    from torch.utils.data import SequentialSampler

    def _get_train_sampler(self_trainer, dataset=None):
        # transformers versions differ:
        # some call _get_train_sampler()
        # newer ones call _get_train_sampler(dataset)
        dataset = dataset if dataset is not None else self_trainer.train_dataset
        return SequentialSampler(dataset)

    import types
    trainer._get_train_sampler = types.MethodType(_get_train_sampler, trainer)

    words_per_step = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        / len(train_dataset)
        * total_words_in_corpus
    )

    logger.info("=" * 70)
    logger.info("CURRICULUM LEARNING TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info("Architecture:         GPT-2 (causal LM)")
    logger.info(f"Curriculum metric:    {COMPLEXITY_COL} (easy → hard)")
    logger.info(f"Dataset:              {HF_DATASET}")
    logger.info(f"Corpus words:         {total_words_in_corpus:,}")
    logger.info(f"Sequences (chunked):  {len(train_dataset):,}")
    logger.info(f"Sequence length:      {max_seq_length}")
    logger.info(f"Batch size:           {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accum:       {training_args.gradient_accumulation_steps}")
    logger.info(f"Effective batch:      {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Words / opt step:     {words_per_step:,.1f}")
    logger.info(f"Learning rate:        {training_args.learning_rate}")
    logger.info(f"Epochs:               {TRAINING_CONFIG['training']['num_epochs']}")
    logger.info(f"BabyLM checkpoints:   {len(TRAINING_CONFIG['checkpoint_intervals'])}")
    logger.info(f"DataLoader sampler:   SequentialSampler (curriculum order preserved)")
    logger.info("=" * 70)

    return trainer, word_exposure_callback


# ============================================================================
# SAVE FINAL BUNDLE
# ============================================================================

def save_final_bundle(model, tokenizer, final_dir: str, total_words: int) -> None:
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "final_model_metadata.json"), "w") as f:
        json.dump(
            {
                "timestamp":            datetime.now().isoformat(),
                "model_architecture":   "GPT-2",
                "objective":            "causal_language_modeling",
                "training_order":       "easy_to_hard_curriculum",
                "curriculum_metric":    COMPLEXITY_COL,
                "dataset":              HF_DATASET,
                "track":                "BabyLM-2026-Strict-Small",
                "total_words_in_corpus": total_words,
                "training_config":      TRAINING_CONFIG,
            },
            f,
            indent=2,
        )
    logger.info(f"✓ Final CL model bundle saved to {final_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    TRAINING_CONFIG["output_dir"]              = args.output_dir
    TRAINING_CONFIG["babylm_checkpoint_dir"]   = args.babylm_checkpoint_dir
    TRAINING_CONFIG["detailed_checkpoint_dir"] = args.detailed_checkpoint_dir
    TRAINING_CONFIG["data"]["max_seq_length"]  = args.max_seq_length

    seed = TRAINING_CONFIG["training"]["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("=" * 70)
    logger.info("BabyLM 2026 Strict-Small — Curriculum Learning (Gunning Fog, easy→hard)")
    logger.info("=" * 70)
    logger.info(f"GPU:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Dataset:  {HF_DATASET}")
    logger.info(f"Metric:   {COMPLEXITY_COL}")
    logger.info(f"Seed:     {seed}")
    logger.info("=" * 70)

    logger.info("[1/4] Loading GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token = eos_token ({tokenizer.eos_token!r})")

    logger.info("[2/4] Loading dataset and sorting by Gunning Fog (easy → hard)")
    train_dataset, total_words = load_and_prepare_dataset(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        tokenize_batch_size=TRAINING_CONFIG["data"]["tokenize_batch_size"],
        chunk_batch_size=TRAINING_CONFIG["data"]["chunk_batch_size"],
        max_train_examples=args.max_train_examples,
        drop_fog_nulls=args.drop_fog_nulls,
    )

    logger.info("[3/4] Creating GPT-2 model")
    model = create_model(tokenizer, max_seq_length=args.max_seq_length)

    logger.info("[4/4] Setting up Trainer with SequentialSampler (curriculum order)")
    trainer, word_callback = setup_training(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        total_words_in_corpus=total_words,
        max_seq_length=args.max_seq_length,
    )

    logger.info("=" * 70)
    logger.info("STARTING CURRICULUM TRAINING (Gunning Fog easy → hard)")
    logger.info("=" * 70)
    logger.info(f"Corpus:          {total_words:,} whitespace words")
    logger.info(f"Sequences/epoch: {len(train_dataset):,}")
    logger.info(f"10 epochs →      up to ~{total_words * 10:,} total word exposures")
    logger.info("=" * 70)

    train_kwargs = {}
    if args.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    trainer.train(**train_kwargs)

    logger.info("=" * 70)
    logger.info("CURRICULUM TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(
        f"BabyLM checkpoints saved: "
        f"{len(word_callback.checkpoints_saved)}/{len(TRAINING_CONFIG['checkpoint_intervals'])}"
    )
    logger.info(f"Estimated total word exposure: {word_callback.total_words_seen:,.0f}")

    final_dir = os.path.join(TRAINING_CONFIG["output_dir"], "final")
    save_final_bundle(model, tokenizer, final_dir, total_words)

    if os.path.exists(TRAINING_CONFIG["babylm_checkpoint_dir"]):
        blm_ckpts = sorted([
            d for d in os.listdir(TRAINING_CONFIG["babylm_checkpoint_dir"])
            if d.startswith("chck_")
        ])
        logger.info(f"✓ {len(blm_ckpts)} BabyLM exposure checkpoints: {blm_ckpts}")


if __name__ == "__main__":
    main()