"""Tests for train_curriculum — uses a tiny in-memory model and temp files."""
import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer


def _tiny_model():
    # vocab_size must match gpt2 tokenizer (50257) to avoid embedding index errors
    cfg = GPT2Config(n_embd=32, n_layer=2, n_head=2, n_inner=128,
                     vocab_size=50257, n_positions=16)
    return GPT2LMHeadModel(cfg)


def _write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": f"doc_{i}", "text": t}) + "\n")


def test_train_curriculum_returns_correct_shape(tmp_path):
    """train_curriculum returns (checkpoint_paths, history) with expected lengths."""
    from influence_curriculum.train import train_curriculum, TrainingConfig

    phase0 = tmp_path / "phase0.jsonl"
    phase1 = tmp_path / "phase1.jsonl"
    texts = ["hello world foo bar"] * 8
    _write_jsonl(phase0, texts)
    _write_jsonl(phase1, texts)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = _tiny_model()

    cfg = TrainingConfig(
        epochs=1,  # ignored by train_curriculum; phases control epochs
        per_device_batch_size=4,
        effective_batch_size=4,
        max_seq_len=8,
        fp16=False,
    )

    phases = [(str(phase0), 2), (str(phase1), 1)]  # 2 epochs phase0, 1 epoch phase1
    ckpt_paths, history = train_curriculum(
        model, tokenizer, phases, str(tmp_path), cfg, seed=0, device="cpu"
    )

    assert len(ckpt_paths) == 3, f"expected 3 checkpoints, got {len(ckpt_paths)}"
    assert len(history) == 3, f"expected 3 history entries, got {len(history)}"
    assert history[0]["phase"] == 0
    assert history[0]["epoch"] == 0
    assert isinstance(history[0]["loss"], float)
    assert isinstance(history[0]["lr"], float)
    assert history[2]["phase"] == 1
    assert history[2]["epoch"] == 0


def test_train_curriculum_checkpoint_names(tmp_path):
    """Checkpoint directories are named phase_{p:02d}_epoch_{e:02d}."""
    from influence_curriculum.train import train_curriculum, TrainingConfig

    phase0 = tmp_path / "p0.jsonl"
    _write_jsonl(phase0, ["foo bar baz"] * 4)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = _tiny_model()

    cfg = TrainingConfig(per_device_batch_size=4, effective_batch_size=4, max_seq_len=8, fp16=False)
    ckpt_paths, _ = train_curriculum(
        model, tokenizer, [(str(phase0), 2)], str(tmp_path), cfg, seed=0, device="cpu"
    )

    names = [Path(p).name for p in ckpt_paths]
    assert names == ["phase_00_epoch_00", "phase_00_epoch_01"], f"got {names}"


def test_train_curriculum_resume_skips_early(tmp_path):
    """start_phase=0, start_epoch=1 skips epoch 0 and only runs epoch 1."""
    from influence_curriculum.train import train_curriculum, TrainingConfig

    phase0 = tmp_path / "p0.jsonl"
    _write_jsonl(phase0, ["foo bar"] * 4)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = _tiny_model()

    cfg = TrainingConfig(per_device_batch_size=4, effective_batch_size=4, max_seq_len=8, fp16=False)
    ckpt_paths, history = train_curriculum(
        model, tokenizer, [(str(phase0), 2)], str(tmp_path), cfg,
        seed=0, device="cpu", start_phase=0, start_epoch=1,
    )

    assert len(ckpt_paths) == 1, f"expected 1 checkpoint (only epoch 1), got {len(ckpt_paths)}"
    assert Path(ckpt_paths[0]).name == "phase_00_epoch_01"
    assert history[0]["epoch"] == 1
