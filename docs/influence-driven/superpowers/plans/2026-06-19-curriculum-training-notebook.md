# Curriculum Training Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `train_curriculum` to `train.py` and create `notebooks/05_curriculum_training.ipynb` that pulls curriculum JSONL files from HF Hub, trains a fresh GPT-2 model in curriculum phases with per-epoch Hub checkpointing and resume support, then plots training curves.

**Architecture:** `train_curriculum` is a new function alongside `train_surrogate` in `train.py`; it owns the multi-phase loop, optimizer/scheduler, checkpoint saving, and Hub push. The notebook is config-only: it sets up paths, downloads curriculum, runs the resume check, calls `train_curriculum`, pushes the final model, and plots curves.

**Tech Stack:** PyTorch, HuggingFace Transformers (GPT2Config, GPT2LMHeadModel), huggingface_hub, matplotlib, tqdm.

## Global Constraints

- Checkpoint naming: `phase_{p:02d}_epoch_{e:02d}` — exact format, used for resume parsing
- Model config: `GPT2Config(n_embd=384, n_layer=8, n_head=6, n_inner=1536, vocab_size=16384, n_positions=1024)`
- Default config: `PHASE_EPOCHS=[5,5]`, `PASS_BATCH_SIZE=64`, `EFFECTIVE_BATCH_SIZE=256`, `LEARNING_RATE=7e-4`, `MAX_SEQ_LEN=128`
- No changes to `train_surrogate` or any existing function
- Notebook target env: RunPod A40 48 GB, `/workspace` persistent volume
- Python 3.12, PyTorch ≥ 2.1

---

### Task 1: `train_curriculum` function in `train.py`

**Files:**
- Modify: `src/influence_curriculum/train.py` (append after `train_surrogate`)
- Test: `tests/test_train_curriculum.py`

**Interfaces:**
- Consumes: `TrainingConfig` (already defined in `train.py`)
- Produces: `train_curriculum(model, tokenizer, phases, output_dir, config, seed, device, start_phase=0, start_epoch=0, hub_repo=None, hub_token=None) -> tuple[list[str], list[dict]]`
  - `phases`: `list[tuple[str, int]]` — each element is `(jsonl_path, n_epochs)`
  - returns `(checkpoint_paths, history)` where `history` is a list of `{"phase": int, "epoch": int, "loss": float, "lr": float}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_curriculum.py`:

```python
"""Tests for train_curriculum — uses a tiny in-memory model and temp files."""
import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer


def _tiny_model():
    cfg = GPT2Config(n_embd=32, n_layer=2, n_head=2, n_inner=128,
                     vocab_size=100, n_positions=16)
    return GPT2LMHeadModel(cfg)


def _tiny_tokenizer():
    # Use a real tokenizer but with a tiny vocab model — just need encode/decode
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    tok = Tokenizer(BPE())
    from tokenizers.pre_tokenizers import Whitespace
    tok.pre_tokenizer = Whitespace()
    # Build minimal vocab: 0-99 as single-char tokens
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tok)
    wrapped.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]"})
    return wrapped


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
    assert history[0] == {"phase": 0, "epoch": 0, "loss": pytest.approx(history[0]["loss"], abs=1e9), "lr": pytest.approx(history[0]["lr"], abs=1e9)}
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cd /Users/adrian.rojas/dev/babylm
python -m pytest tests/test_train_curriculum.py -v 2>&1 | tail -20
```

Expected: all three tests FAIL with `ImportError: cannot import name 'train_curriculum'`

- [ ] **Step 3: Implement `train_curriculum` in `train.py`**

Append the following after the closing `return checkpoint_paths` line of `train_surrogate` (after line 133):

```python


def train_curriculum(
    model: torch.nn.Module,
    tokenizer,
    phases: list[tuple[str, int]],
    output_dir: str,
    config: TrainingConfig,
    seed: int,
    device: str,
    start_phase: int = 0,
    start_epoch: int = 0,
    hub_repo: str | None = None,
    hub_token: str | None = None,
) -> tuple[list[str], list[dict]]:
    """Train a model through curriculum phases with per-epoch checkpointing.

    phases: [(jsonl_path, n_epochs), ...] — each phase trains on its JSONL for n_epochs.
    start_phase / start_epoch: resume by skipping already-completed work.
    hub_repo: if set, push each checkpoint after saving.
    Returns (checkpoint_paths, history) where history entries are
    {"phase": int, "epoch": int, "loss": float, "lr": float}.
    """
    import json as _json

    torch.manual_seed(seed)
    rng = random.Random(seed)
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.train()

    grad_accum = max(1, config.effective_batch_size // config.per_device_batch_size)
    pin = device == "cuda"

    # ── Compute total optimizer steps across ALL phases ───────────────────────
    total_steps = 0
    phase_steps: list[int] = []
    for jsonl_path, n_epochs in phases:
        texts = [_json.loads(l)["text"] for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        n_docs = len(texts)
        steps = max(1, n_epochs * math.ceil(math.ceil(n_docs / config.per_device_batch_size) / grad_accum))
        phase_steps.append(steps)
        total_steps += steps

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=max(1, total_steps),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    # ── Fast-forward scheduler to resume point ────────────────────────────────
    completed_steps = 0
    for p_idx in range(start_phase):
        for _ in range(phase_steps[p_idx]):
            scheduler.step()
            completed_steps += 1
    # fast-forward within start_phase for completed epochs
    if start_phase < len(phases):
        jsonl_path, _ = phases[start_phase]
        texts_fp = [_json.loads(l)["text"] for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        n_docs_fp = len(texts_fp)
        steps_per_epoch_fp = max(1, math.ceil(math.ceil(n_docs_fp / config.per_device_batch_size) / grad_accum))
        for _ in range(start_epoch * steps_per_epoch_fp):
            scheduler.step()
            completed_steps += 1

    pbar = tqdm(total=total_steps, initial=completed_steps, desc="curriculum", unit="step")

    checkpoint_paths: list[str] = []
    history: list[dict] = []

    for p_idx, (jsonl_path, n_epochs) in enumerate(phases):
        if p_idx < start_phase:
            continue

        texts = [_json.loads(l)["text"] for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        batch_out = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        dataset = TensorDataset(batch_out["input_ids"], batch_out["attention_mask"])
        indices = list(range(len(dataset)))

        epoch_start = start_epoch if p_idx == start_phase else 0

        for epoch in range(epoch_start, n_epochs):
            rng.shuffle(indices)
            loader = DataLoader(
                dataset,
                batch_size=config.per_device_batch_size,
                sampler=indices,
                pin_memory=pin,
            )
            optimizer.zero_grad()
            running_loss, accum_count = 0.0, 0
            epoch_total_loss, epoch_total_count = 0.0, 0

            for step, (ids, mask) in enumerate(loader):
                ids  = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = ids.clone()
                labels[mask == 0] = -100
                with torch.cuda.amp.autocast(enabled=config.fp16):
                    outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                scaler.scale(outputs.loss / grad_accum).backward()
                running_loss += outputs.loss.item()
                accum_count += 1
                epoch_total_loss += outputs.loss.item()
                epoch_total_count += 1

                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                    pbar.set_postfix(
                        phase=p_idx, epoch=epoch,
                        loss=f"{running_loss/accum_count:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    running_loss, accum_count = 0.0, 0

            if len(loader) % grad_accum != 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)

            epoch_loss = epoch_total_loss / epoch_total_count if epoch_total_count else float("nan")
            current_lr = scheduler.get_last_lr()[0]
            history.append({"phase": p_idx, "epoch": epoch, "loss": epoch_loss, "lr": current_lr})

            ckpt_name = f"phase_{p_idx:02d}_epoch_{epoch:02d}"
            ckpt_path = str(ckpt_dir / ckpt_name)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            checkpoint_paths.append(ckpt_path)
            tqdm.write(f"phase {p_idx:02d} epoch {epoch:02d} done — saved {ckpt_name}  loss={epoch_loss:.4f}")

            if hub_repo:
                try:
                    from huggingface_hub import HfApi
                    HfApi().upload_folder(
                        folder_path=ckpt_path,
                        repo_id=hub_repo,
                        path_in_repo=ckpt_name,
                        repo_type="model",
                        token=hub_token,
                        commit_message=f"checkpoint {ckpt_name}",
                    )
                    tqdm.write(f"  → pushed {ckpt_name} to {hub_repo}")
                except Exception as e:
                    tqdm.write(f"  ⚠ Hub push failed: {e}")

    pbar.close()
    model.eval()
    return checkpoint_paths, history
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_train_curriculum.py -v 2>&1 | tail -20
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/influence_curriculum/train.py tests/test_train_curriculum.py
git commit -m "feat: add train_curriculum for multi-phase curriculum training"
```

---

### Task 2: Notebook `05_curriculum_training.ipynb`

**Files:**
- Create: `notebooks/05_curriculum_training.ipynb`

**Interfaces:**
- Consumes: `train_curriculum` from Task 1 (exact signature above)
- Produces: trained model pushed to `HUB_MODEL_REPO`, local `history` list for plotting

- [ ] **Step 1: Create the notebook**

Create `/Users/adrian.rojas/dev/babylm/notebooks/05_curriculum_training.ipynb` as a valid Jupyter notebook JSON with the following 8 cells. Write the complete file:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b001",
   "metadata": {},
   "source": ["# 05 — Curriculum Training\n\nPulls curriculum JSONL files from HF Hub, trains a fresh GPT-2 model in curriculum phases,\npushes per-epoch checkpoints to Hub (enables resume), and plots training curves.\n\n**Resume:** If the pod is interrupted, re-run all cells. Cell 5 will find the latest\n`phase_XX_epoch_YY` checkpoint on Hub and pick up from there."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b002",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Cell 1: RunPod setup (run once per session) ──\n",
    "# Uncomment and run this block when starting a fresh RunPod session.\n",
    "\n",
    "# !git clone https://github.com/flakoash/influece_driven_curriculum_sorter.git /workspace/babylm\n",
    "# %cd /workspace/babylm\n",
    "# !pip install -e \".[dev]\" --quiet\n",
    "\n",
    "print(\"Setup complete.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b003",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Cell 2: Config — edit these before running ──\n",
    "\n",
    "TOKENIZER_NAME      = \"BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small\"\n",
    "HUB_CURRICULUM_REPO = None   # \"yourname/babylm-curriculum-run001\" — source of epoch_XX.jsonl files\n",
    "HUB_MODEL_REPO      = None   # \"yourname/babylm-trained-run001\"    — destination for checkpoints\n",
    "HF_TOKEN            = None   # or: import os; HF_TOKEN = os.environ[\"HF_TOKEN\"]\n",
    "\n",
    "PHASE_EPOCHS         = [5, 5]   # epochs per curriculum phase (one entry per epoch_XX.jsonl)\n",
    "MAX_SEQ_LEN          = 128\n",
    "SEED                 = 0\n",
    "PASS_BATCH_SIZE      = 64    # per-device batch size (A40 48 GB safe)\n",
    "EFFECTIVE_BATCH_SIZE = 256\n",
    "LEARNING_RATE        = 7e-4\n",
    "\n",
    "print(f\"Phases        : {PHASE_EPOCHS}  ({sum(PHASE_EPOCHS)} total epochs)\")\n",
    "print(f\"Curriculum Hub: {HUB_CURRICULUM_REPO}\")\n",
    "print(f\"Model Hub     : {HUB_MODEL_REPO}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b004",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Cell 3: Imports + device ──\n",
    "import json, os, sys\n",
    "from pathlib import Path\n",
    "\n",
    "os.environ.setdefault(\"PYTORCH_CUDA_ALLOC_CONF\", \"expandable_segments:True\")\n",
    "\n",
    "import torch\n",
    "from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel\n",
    "\n",
    "for _candidate in [Path(\"../src\"), Path(\"src\"), Path(\"/workspace/babylm/src\")]:\n",
    "    if _candidate.exists() and str(_candidate.resolve()) not in sys.path:\n",
    "        sys.path.insert(0, str(_candidate.resolve()))\n",
    "\n",
    "from influence_curriculum.train import TrainingConfig, train_curriculum\n",
    "\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "OUT    = Path(\"/workspace/outputs/curriculum_training\")\n",
    "OUT.mkdir(parents=True, exist_ok=True)\n",
    "CURRICULUM_DIR = Path(\"/workspace/curriculum\")\n",
    "CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)\n",
    "tokenizer.pad_token = tokenizer.eos_token\n",
    "\n",
    "print(f\"Device: {DEVICE}\")\n",
    "if DEVICE == \"cuda\":\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b005",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Cell 4: Pull curriculum JSONL files from Hub ──\n",
    "# Downloads epoch_00.jsonl, epoch_01.jsonl, ... into /workspace/curriculum/\n",
    "# Skips files already present.\n",
    "\n",
    "phase_files = []\n",
    "\n",
    "if HUB_CURRICULUM_REPO:\n",
    "    from huggingface_hub import hf_hub_download, list_repo_files\n",
    "    remote_files = [f for f in list_repo_files(HUB_CURRICULUM_REPO, repo_type=\"dataset\", token=HF_TOKEN)\n",
    "                    if f.startswith(\"curriculum/epoch_\") and f.endswith(\".jsonl\")]\n",
    "    remote_files = sorted(remote_files)\n",
    "    print(f\"Found {len(remote_files)} curriculum files on Hub: {remote_files}\")\n",
    "    for remote_path in remote_files:\n",
    "        fname = Path(remote_path).name\n",
    "        local = CURRICULUM_DIR / fname\n",
    "        if not local.exists():\n",
    "            hf_hub_download(HUB_CURRICULUM_REPO, remote_path, local_dir=str(CURRICULUM_DIR),\n",
    "                            repo_type=\"dataset\", token=HF_TOKEN)\n",
    "            print(f\"  downloaded {fname}\")\n",
    "        else:\n",
    "            print(f\"  {fname} already present, skipping\")\n",
    "        phase_files.append(str(local))\n",
    "else:\n",
    "    # Fallback: use local files if Hub not set\n",
    "    phase_files = sorted(str(p) for p in CURRICULUM_DIR.glob(\"epoch_*.jsonl\"))\n",
    "    print(f\"HUB_CURRICULUM_REPO not set — using local files: {[Path(p).name for p in phase_files]}\")\n",
    "\n",
    "assert len(phase_files) >= len(PHASE_EPOCHS), (\n",
    "    f\"Need {len(PHASE_EPOCHS)} JSONL files for PHASE_EPOCHS={PHASE_EPOCHS}, \"\n",
    "    f\"found {len(phase_files)}: {phase_files}\"\n",
    ")\n",
    "phase_files = phase_files[:len(PHASE_EPOCHS)]\n",
    "phases = list(zip(phase_files, PHASE_EPOCHS))\n",
    "print(f\"\\nTraining phases:\")\n",
    "for i, (f, e) in enumerate(phases):\n",
    "    n = sum(1 for _ in open(f))\n",
    "    print(f\"  Phase {i}: {Path(f).name}  {n:,} docs  {e} epochs\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b006",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Cell 5: Resume check ──\n",
    "# Scans HUB_MODEL_REPO for the latest phase_XX_epoch_YY checkpoint.\n",
    "# Sets start_phase / start_epoch and loads weights if a checkpoint is found.\n",
    "\n",
    "import re\n",
    "\n",
    "start_phase, start_epoch = 0, 0\n",
    "model = GPT2LMHeadModel(GPT2Config(\n",
    "    n_embd=384, n_layer=8, n_head=6, n_inner=1536,\n",
    "    vocab_size=16384, n_positions=1024,\n",
    "    bos_token_id=tokenizer.bos_token_id,\n",
    "    eos_token_id=tokenizer.eos_token_id,\n",
    "))\n",
    "print(f\"Model params: {sum(p.numel() for p in model.parameters()):,}\")\n",
    "\n",
    "if HUB_MODEL_REPO:\n",
    "    try:\n",
    "        from huggingface_hub import list_repo_files, snapshot_download\n",
    "        ckpt_pattern = re.compile(r\"^phase_(\\d+)_epoch_(\\d+)/config\\.json$\")\n",
    "        remote_ckpts = [\n",
    "            (int(m.group(1)), int(m.group(2)))\n",
    "            for f in list_repo_files(HUB_MODEL_REPO, repo_type=\"model\", token=HF_TOKEN)\n",
    "            if (m := ckpt_pattern.match(f))\n",
    "        ]\n",
    "        if remote_ckpts:\n",
    "            latest_phase, latest_epoch = max(remote_ckpts)\n",
    "            ckpt_name = f\"phase_{latest_phase:02d}_epoch_{latest_epoch:02d}\"\n",
    "            local_ckpt = OUT / \"checkpoints\" / ckpt_name\n",
    "            if not local_ckpt.exists():\n",
    "                print(f\"Downloading checkpoint {ckpt_name} from Hub...\")\n",
    "                snapshot_download(\n",
    "                    repo_id=HUB_MODEL_REPO,\n",
    "                    local_dir=str(OUT / \"checkpoints\"),\n",
    "                    allow_patterns=f\"{ckpt_name}/*\",\n",
    "                    repo_type=\"model\",\n",
    "                    token=HF_TOKEN,\n",
    "                )\n",
    "            model = GPT2LMHeadModel.from_pretrained(str(local_ckpt))\n",
    "            # advance resume point past the completed epoch\n",
    "            n_epochs_in_phase = PHASE_EPOCHS[latest_phase]\n",
    "            if latest_epoch + 1 < n_epochs_in_phase:\n",
    "                start_phase, start_epoch = latest_phase, latest_epoch + 1\n",
    "            else:\n",
    "                start_phase, start_epoch = latest_phase + 1, 0\n",
    "            print(f\"Resuming from {ckpt_name} → start_phase={start_phase}, start_epoch={start_epoch}\")\n",
    "        else:\n",
    "            print(\"No checkpoints found on Hub — starting from scratch.\")\n",
    "    except Exception as e:\n",
    "        print(f\"Hub resume check failed ({e}) — starting from scratch.\")\n",
    "else:\n",
    "    print(\"HUB_MODEL_REPO not set — starting from scratch, no Hub push.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b007",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Cell 6: Train ──\n",
    "\n",
    "train_cfg = TrainingConfig(\n",
    "    per_device_batch_size=PASS_BATCH_SIZE,\n",
    "    effective_batch_size=EFFECTIVE_BATCH_SIZE,\n",
    "    learning_rate=LEARNING_RATE,\n",
    "    max_seq_len=MAX_SEQ_LEN,\n",
    "    fp16=(DEVICE == \"cuda\"),\n",
    ")\n",
    "\n",
    "checkpoint_paths, history = train_curriculum(\n",
    "    model, tokenizer,\n",
    "    phases=phases,\n",
    "    output_dir=str(OUT),\n",
    "    config=train_cfg,\n",
    "    seed=SEED,\n",
    "    device=DEVICE,\n",
    "    start_phase=start_phase,\n",
    "    start_epoch=start_epoch,\n",
    "    hub_repo=HUB_MODEL_REPO,\n",
    "    hub_token=HF_TOKEN,\n",
    ")\n",
    "print(f\"\\nTraining complete. {len(checkpoint_paths)} checkpoints saved.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b008",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Cell 7: Push final model to Hub ──\n",
    "\n",
    "if HUB_MODEL_REPO and checkpoint_paths:\n",
    "    from huggingface_hub import HfApi\n",
    "    api = HfApi()\n",
    "    api.create_repo(repo_id=HUB_MODEL_REPO, repo_type=\"model\", exist_ok=True, token=HF_TOKEN)\n",
    "    final_ckpt = checkpoint_paths[-1]\n",
    "    print(f\"Pushing final model ({Path(final_ckpt).name}) to {HUB_MODEL_REPO} ...\")\n",
    "    api.upload_folder(\n",
    "        folder_path=final_ckpt,\n",
    "        repo_id=HUB_MODEL_REPO,\n",
    "        path_in_repo=\"final\",\n",
    "        repo_type=\"model\",\n",
    "        token=HF_TOKEN,\n",
    "        commit_message=\"final trained model\",\n",
    "    )\n",
    "    print(\"Done.\")\n",
    "else:\n",
    "    print(\"HUB_MODEL_REPO not set or no checkpoints — skipping Hub push.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b009",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Cell 8: Training curves ──\n",
    "import math\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "epochs_x    = list(range(len(history)))\n",
    "losses      = [h[\"loss\"] for h in history]\n",
    "perplexities = [math.exp(min(l, 20)) for l in losses]  # cap at exp(20) to avoid overflow\n",
    "lrs         = [h[\"lr\"] for h in history]\n",
    "boundary    = PHASE_EPOCHS[0] - 0.5   # vertical line between phase 0 and phase 1\n",
    "\n",
    "fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)\n",
    "\n",
    "axes[0].plot(epochs_x, losses, marker=\"o\", color=\"steelblue\")\n",
    "axes[0].axvline(boundary, color=\"red\", linestyle=\"--\", alpha=0.6, label=\"phase boundary\")\n",
    "axes[0].set_ylabel(\"Loss\")\n",
    "axes[0].set_title(\"Training Loss\")\n",
    "axes[0].legend()\n",
    "\n",
    "axes[1].plot(epochs_x, perplexities, marker=\"o\", color=\"darkorange\")\n",
    "axes[1].axvline(boundary, color=\"red\", linestyle=\"--\", alpha=0.6)\n",
    "axes[1].set_ylabel(\"Perplexity\")\n",
    "axes[1].set_title(\"Perplexity (exp(loss))\")\n",
    "\n",
    "axes[2].plot(epochs_x, lrs, marker=\"o\", color=\"green\")\n",
    "axes[2].axvline(boundary, color=\"red\", linestyle=\"--\", alpha=0.6)\n",
    "axes[2].set_ylabel(\"Learning Rate\")\n",
    "axes[2].set_title(\"LR Schedule\")\n",
    "axes[2].set_xlabel(\"Epoch (global)\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(str(OUT / \"training_curves.png\"), dpi=150)\n",
    "plt.show()\n",
    "print(f\"Curves saved to {OUT / 'training_curves.png'}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify the notebook is valid JSON**

```bash
python3 -c "import json; json.load(open('notebooks/05_curriculum_training.ipynb')); print('valid JSON')"
```

Expected: `valid JSON`

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_curriculum_training.ipynb
git commit -m "feat: add curriculum training notebook (05)"
```
