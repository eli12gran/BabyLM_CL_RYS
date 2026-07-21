"""Tests for train_word_aware — word-count milestone checkpointing."""
import json
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer


def _tiny_model():
    cfg = GPT2Config(n_embd=32, n_layer=2, n_head=2, n_inner=128,
                     vocab_size=50257, n_positions=16)
    return GPT2LMHeadModel(cfg)


def _write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": f"doc_{i}", "text": t}) + "\n")


def _tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def _cfg():
    from influence_curriculum.train import TrainingConfig
    return TrainingConfig(per_device_batch_size=4, effective_batch_size=4,
                          max_seq_len=8, fp16=False)


# ── corpus helpers ────────────────────────────────────────────────────────────
# Each text is exactly 5 words → easy to compute expected word counts.
FIVE_WORD_TEXT = "one two three four five"   # 5 words


def test_train_word_aware_milestone_checkpoints_created(tmp_path):
    """Checkpoints are saved at the right word-count milestones."""
    from influence_curriculum.train import train_word_aware

    # 8 docs × 5 words = 40 words per epoch.
    # Milestones at [20, 40] → one hit mid-epoch, one at/after epoch end.
    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * 8)

    ckpt_paths, _ = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), 1)],
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[20, 40],
    )

    names = {Path(p).name for p in ckpt_paths}
    assert "chck_20" in names, f"expected chck_20 in {names}"
    assert "chck_40" in names, f"expected chck_40 in {names}"


def test_train_word_aware_history_includes_words_processed(tmp_path):
    """Each history entry records cumulative words_processed."""
    from influence_curriculum.train import train_word_aware

    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * 8)

    _, history = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), 2)],
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[],
    )

    assert len(history) == 2
    for entry in history:
        assert "words_processed" in entry, f"missing words_processed in {entry}"
        assert isinstance(entry["words_processed"], int)
    # second epoch should have seen more words than the first
    assert history[1]["words_processed"] > history[0]["words_processed"]


def test_train_word_aware_words_processed_matches_corpus(tmp_path):
    """Total words_processed after all epochs = docs × words_per_doc × epochs."""
    from influence_curriculum.train import train_word_aware

    n_docs, n_epochs = 8, 3
    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * n_docs)

    _, history = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), n_epochs)],
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[],
    )

    expected_total = n_docs * 5 * n_epochs
    assert history[-1]["words_processed"] == expected_total, (
        f"expected {expected_total}, got {history[-1]['words_processed']}"
    )


def test_train_word_aware_milestone_not_duplicated(tmp_path):
    """Each milestone is checkpointed exactly once even across multiple epochs."""
    from influence_curriculum.train import train_word_aware

    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * 8)   # 40 words/epoch

    ckpt_paths, _ = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), 3)],                # 120 words total
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[40],                   # crossed every epoch — save only once
    )

    milestone_paths = [p for p in ckpt_paths if "chck_" in Path(p).name]
    assert len(milestone_paths) == 1, (
        f"milestone chck_40 should appear exactly once, got {len(milestone_paths)}: {milestone_paths}"
    )


def test_train_word_aware_skips_milestones_before_start_words(tmp_path):
    """Milestones already passed (below start_words) are not re-checkpointed."""
    from influence_curriculum.train import train_word_aware

    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * 8)

    ckpt_paths, _ = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), 1)],
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[20, 40],
        start_words=25,                          # milestone at 20 already passed
    )

    names = {Path(p).name for p in ckpt_paths}
    assert "chck_20" not in names, f"chck_20 should be skipped (start_words=25)"
    assert "chck_40" in names, f"chck_40 should still be saved"


def test_train_word_aware_stops_at_total_word_target(tmp_path):
    """Training stops once total_word_target is reached, even if n_epochs would run longer."""
    from influence_curriculum.train import train_word_aware

    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * 8)  # 40 words/epoch

    # 3 specified epochs = 120 words; target = 60 → should stop after 2 epochs
    _, history = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), 3)],
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[],
        total_word_target=60,
    )

    assert len(history) == 2, (
        f"expected 2 epochs (target=60 words, 40 words/epoch), got {len(history)}"
    )
    assert history[-1]["words_processed"] >= 60


def test_train_word_aware_extends_last_phase_to_reach_target(tmp_path):
    """When specified epochs fall short, last phase is extended to reach total_word_target."""
    from influence_curriculum.train import train_word_aware

    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * 8)  # 40 words/epoch

    # 1 specified epoch = 40 words; target = 100 → needs 3 epochs to reach 100
    _, history = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), 1)],
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[],
        total_word_target=100,
    )

    assert history[-1]["words_processed"] >= 100, (
        f"expected ≥100 words, got {history[-1]['words_processed']}"
    )
    assert len(history) >= 3, (
        f"expected ≥3 epochs to reach 100 words at 40 words/epoch, got {len(history)}"
    )


def test_train_word_aware_total_target_on_resume(tmp_path):
    """Extension is computed from the resume point, not from scratch.

    If start_words=40 and target=100, we need 60 more words.
    With 40 words/epoch and 0 scheduled epochs remaining, that's 2 more epochs.
    The buggy code would compare target against total_scheduled (including
    already-done work) and under-extend.
    """
    from influence_curriculum.train import train_word_aware

    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * 8)  # 40 words/epoch

    # 1 specified epoch, resuming after it's done (start_epoch=1).
    # No scheduled epochs remain → all 60 words must come from extension.
    _, history = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), 1)],
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[],
        total_word_target=100,
        start_words=40,
        start_phase=0,
        start_epoch=1,
    )

    assert history[-1]["words_processed"] >= 100, (
        f"expected ≥100 total words from resume, got {history[-1]['words_processed']}"
    )
    assert len(history) >= 2, "expected ≥2 epochs from the resume point"


def test_train_word_aware_total_target_resume_mid_extension(tmp_path):
    """Extension is correct when resuming beyond the originally specified epochs.

    Simulates: specified 1 epoch, already ran 3 epochs via extension (120 words).
    Target = 200, start_words = 120, start_epoch = 3.
    Remaining = 80 words → ceil(80/40) = 2 more epochs needed.
    """
    from influence_curriculum.train import train_word_aware

    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * 8)  # 40 words/epoch

    _, history = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), 1)],
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[],
        total_word_target=200,
        start_words=120,
        start_phase=0,
        start_epoch=3,
    )

    assert history[-1]["words_processed"] >= 200, (
        f"expected ≥200 words, got {history[-1]['words_processed']}"
    )
    assert len(history) >= 2, "expected ≥2 epochs from resume point"


def test_train_word_aware_per_epoch_checkpoints_saved(tmp_path):
    """Per-epoch checkpoints (phase_XX_epoch_YY) are saved alongside milestones."""
    from influence_curriculum.train import train_word_aware

    phase = tmp_path / "phase.jsonl"
    _write_jsonl(phase, [FIVE_WORD_TEXT] * 8)

    ckpt_paths, _ = train_word_aware(
        _tiny_model(), _tokenizer(),
        phases=[(str(phase), 2)],
        output_dir=str(tmp_path),
        config=_cfg(),
        seed=0, device="cpu",
        word_checkpoints=[],
    )

    names = [Path(p).name for p in ckpt_paths]
    assert "phase_00_epoch_00" in names
    assert "phase_00_epoch_01" in names
