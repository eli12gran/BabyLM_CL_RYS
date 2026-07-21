# Influence Curriculum Sorter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `sort_by_influence()` — a three-phase pipeline that trains a surrogate CLM, scores every document by training-data influence, and writes a per-epoch curriculum as JSONL files.

**Architecture:** Four focused modules (`data`, `train`, `score`, `curriculum`) behind a thin public API (`sort`). Each module maps to one phase and can be imported and tested independently. Notebooks debug each phase in isolation using small synthetic or real data.

**Tech Stack:** Python ≥3.11, UV, PyTorch ≥2.3, Transformers ≥4.40, NumPy ≥1.26, JupyterLab (dev).

## Global Constraints

- Device-agnostic: `cpu` / `mps` / `cuda` all work; never hardcode `.cuda()` or `cuda:0`
- `fp16=False` always (MPS unreliability)
- CLM only — no MLM, no masking machinery
- Grad target: input-embedding matrix (`model.get_input_embeddings().weight`), dense, cosine-normalized
- Default curriculum: `C~` — per-epoch sort + 1000-doc segment shuffle, ascending
- `src/` layout; package name `influence_curriculum`
- All tests runnable with no GPU (cpu only, tiny synthetic data)

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | UV project, deps, dev extras |
| `src/influence_curriculum/__init__.py` | re-export `sort_by_influence` |
| `src/influence_curriculum/data.py` | `DataConfig`, `load_documents()` |
| `src/influence_curriculum/train.py` | `TrainingConfig`, `train_surrogate()` |
| `src/influence_curriculum/score.py` | `InfluenceConfig`, `compute_influence_matrix()` |
| `src/influence_curriculum/curriculum.py` | `CurriculumConfig`, `build_curriculum()` |
| `src/influence_curriculum/sort.py` | `sort_by_influence()` public API |
| `tests/test_correctness.py` | §12.1 fast checks (no training) |
| `notebooks/01_data_inspect.ipynb` | Inspect BabyLM segmentation |
| `notebooks/02_influence_debug.ipynb` | Verify Φ math on synthetic data |
| `notebooks/03_curriculum_check.ipynb` | Permutation validity + source-mix |

---

## Task 1: UV project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/influence_curriculum/__init__.py`
- Create: `src/influence_curriculum/data.py` (stub)
- Create: `src/influence_curriculum/train.py` (stub)
- Create: `src/influence_curriculum/score.py` (stub)
- Create: `src/influence_curriculum/curriculum.py` (stub)
- Create: `src/influence_curriculum/sort.py` (stub)

**Interfaces:**
- Produces: installable package `influence_curriculum` importable as `from influence_curriculum import sort_by_influence`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "influence-curriculum"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3",
    "transformers>=4.40",
    "numpy>=1.26",
    "scipy>=1.13",
]

[project.optional-dependencies]
dev = ["jupyterlab", "pytest", "ipykernel"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/influence_curriculum"]
```

- [ ] **Step 2: Create package stubs**

```bash
mkdir -p src/influence_curriculum tests notebooks
```

`src/influence_curriculum/__init__.py`:
```python
from .sort import sort_by_influence

__all__ = ["sort_by_influence"]
```

Each of `data.py`, `train.py`, `score.py`, `curriculum.py`, `sort.py` — create empty files for now (filled in subsequent tasks).

- [ ] **Step 3: Install with UV**

```bash
uv venv
uv pip install -e ".[dev]"
```

Expected: no errors; `python -c "import influence_curriculum"` succeeds.

- [ ] **Step 4: Commit**

```bash
git init  # if not already a repo
git add pyproject.toml src/ tests/ notebooks/
git commit -m "feat: UV project scaffold"
```

---

## Task 2: data.py — segmentation and document loading

**Files:**
- Create: `src/influence_curriculum/data.py`
- Create: `tests/test_correctness.py` (segmentation test only)

**Interfaces:**
- Produces:
  - `DataConfig` dataclass (fields: `doc_boundary`, `default_boundary`, `min_doc_tokens`, `chunking`)
  - `load_documents(dataset_dir: str, config: DataConfig, tokenizer=None) -> tuple[list[str], list[str]]` — returns `(texts, doc_ids)` where `doc_id = "<file_stem>#<segment_index>"`

- [ ] **Step 1: Write failing segmentation test**

`tests/test_correctness.py`:
```python
import textwrap
from pathlib import Path
from influence_curriculum.data import DataConfig, load_documents


def test_line_segmentation(tmp_path):
    (tmp_path / "src.txt").write_text("hello\nworld\n\nskip me\n")
    texts, ids = load_documents(str(tmp_path), DataConfig(doc_boundary="line", min_doc_tokens=0))
    assert texts == ["hello", "world", "skip me"]
    assert ids == ["src#0", "src#1", "src#2"]


def test_blank_line_segmentation(tmp_path):
    (tmp_path / "src.txt").write_text("para one\nstill one\n\npara two\n")
    texts, ids = load_documents(str(tmp_path), DataConfig(doc_boundary="blank_line", min_doc_tokens=0))
    assert texts == ["para one\nstill one", "para two"]
    assert ids == ["src#0", "src#1"]


def test_doc_id_stable(tmp_path):
    (tmp_path / "a.txt").write_text("x\ny\n")
    (tmp_path / "b.txt").write_text("z\n")
    _, ids = load_documents(str(tmp_path), DataConfig(min_doc_tokens=0))
    assert "a#0" in ids and "a#1" in ids and "b#0" in ids
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_correctness.py::test_line_segmentation -v
```
Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement data.py**

```python
from __future__ import annotations
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class DataConfig:
    doc_boundary: str | dict[str, str | Callable] = "line"
    default_boundary: str = "line"
    min_doc_tokens: int = 1
    chunking: str = "truncate"


def _segment(text: str, rule: str | Callable) -> list[str]:
    if callable(rule):
        return [s for s in rule(text) if s.strip()]
    if rule == "line":
        return [l.strip() for l in text.splitlines() if l.strip()]
    if rule == "blank_line":
        return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if rule == "whole_file":
        return [text.strip()] if text.strip() else []
    raise ValueError(f"Unknown doc_boundary: {rule!r}")


def load_documents(
    dataset_dir: str,
    config: DataConfig,
    tokenizer=None,
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    doc_ids: list[str] = []

    for path in sorted(Path(dataset_dir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        rule = config.doc_boundary
        if isinstance(rule, dict):
            rule = rule.get(path.name, config.default_boundary)
        segs = _segment(raw, rule)
        for i, seg in enumerate(segs):
            if tokenizer is not None:
                n = len(tokenizer.encode(seg, add_special_tokens=False))
                if n < config.min_doc_tokens:
                    continue
            elif config.min_doc_tokens > 1 and len(seg.split()) < config.min_doc_tokens:
                continue
            texts.append(seg)
            doc_ids.append(f"{path.stem}#{i}")

    if len(texts) < 2000:
        warnings.warn(
            f"Only {len(texts)} documents loaded from {dataset_dir}. "
            "Segmentation may not have fired correctly.",
            stacklevel=2,
        )
    return texts, doc_ids
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_correctness.py -k "segmentation or doc_id" -v
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/influence_curriculum/data.py tests/test_correctness.py
git commit -m "feat: data loading and segmentation (DataConfig, load_documents)"
```

---

## Task 3: train.py — surrogate training

**Files:**
- Create: `src/influence_curriculum/train.py`

**Interfaces:**
- Consumes: `texts: list[str]`, `tokenizer`, `model` (nn.Module), `device: str`
- Produces:
  - `TrainingConfig` dataclass
  - `train_surrogate(model, tokenizer, texts, output_dir, config, seed, device) -> list[str]` — returns list of checkpoint dir paths, length `config.epochs`

- [ ] **Step 1: Implement train.py**

```python
from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler


@dataclass
class TrainingConfig:
    epochs: int = 10
    effective_batch_size: int = 2048
    per_device_batch_size: int = 32
    learning_rate: float = 7e-4
    lr_scheduler: str = "cosine"
    adam_betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-6
    weight_decay: float = 0.01
    max_seq_len: int = 256
    fp16: bool = False


class _TokenDataset(Dataset):
    def __init__(self, encodings: list[dict]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, i: int) -> dict:
        return {k: v.squeeze(0) for k, v in self.encodings[i].items()}


def _collate(batch: list[dict]) -> dict:
    import torch
    keys = batch[0].keys()
    return {k: torch.nn.utils.rnn.pad_sequence(
        [b[k] for b in batch], batch_first=True, padding_value=0
    ) for k in keys}


def train_surrogate(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    output_dir: str,
    config: TrainingConfig,
    seed: int,
    device: str,
) -> list[str]:
    torch.manual_seed(seed)
    random.seed(seed)

    encodings = [
        tokenizer(t, truncation=True, max_length=config.max_seq_len, return_tensors="pt")
        for t in texts
    ]
    dataset = _TokenDataset(encodings)

    model = model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    grad_accum = max(1, config.effective_batch_size // config.per_device_batch_size)
    steps_per_epoch = max(1, len(dataset) // config.per_device_batch_size)
    total_steps = config.epochs * steps_per_epoch // grad_accum
    scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=total_steps,
    )

    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[str] = []

    indices = list(range(len(dataset)))
    rng = random.Random(seed)

    for epoch in range(config.epochs):
        rng.shuffle(indices)
        loader = DataLoader(
            dataset, batch_size=config.per_device_batch_size,
            sampler=indices, collate_fn=_collate,
        )
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            outputs = model(**batch, labels=labels)
            (outputs.loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        path = str(ckpt_dir / f"epoch_{epoch:02d}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        checkpoint_paths.append(path)

    return checkpoint_paths
```

- [ ] **Step 2: Smoke test (no pytest, just import)**

```bash
uv run python -c "from influence_curriculum.train import TrainingConfig, train_surrogate; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/influence_curriculum/train.py
git commit -m "feat: surrogate training (TrainingConfig, train_surrogate)"
```

---

## Task 4: score.py — influence matrix

**Files:**
- Create: `src/influence_curriculum/score.py`
- Modify: `tests/test_correctness.py` (add mean-gradient identity test)

**Interfaces:**
- Consumes: `checkpoint_paths: list[str]`, `encodings: list[dict]`, `device: str`
- Produces:
  - `InfluenceConfig` dataclass
  - `compute_influence_matrix(checkpoint_paths, encodings, config, device) -> np.ndarray` — shape `(|D|, T)`

- [ ] **Step 1: Add mean-gradient identity test**

Append to `tests/test_correctness.py`:
```python
import numpy as np


def test_mean_gradient_identity():
    """dot-with-mean == (1/D)*sum_of_pairwise for any unit-normalized gradient matrix."""
    rng = np.random.default_rng(0)
    D, V = 8, 50
    raw = rng.random((D, V)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    grads = raw / norms  # unit-normalized

    efficient = grads @ grads.mean(axis=0)                              # O(D)
    pairwise = np.array([(1 / D) * (grads[i] @ grads.T).sum()          # O(D^2)
                         for i in range(D)])
    np.testing.assert_allclose(efficient, pairwise, rtol=1e-5)
```

- [ ] **Step 2: Run to verify pass (pure numpy, no model needed)**

```bash
uv run pytest tests/test_correctness.py::test_mean_gradient_identity -v
```
Expected: PASSED.

- [ ] **Step 3: Implement score.py**

```python
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM


@dataclass
class InfluenceConfig:
    grad_target: str = "input_embeddings"
    grad_path: str = "full"
    normalize: bool = True
    memory_route: str = "recompute"
    projection_dim: int = 0
    grad_batch_size: int = 16


def _doc_gradient(model: torch.nn.Module, input_ids: torch.Tensor, device: str) -> np.ndarray:
    model.zero_grad()
    ids = input_ids.unsqueeze(0).to(device)
    outputs = model(input_ids=ids, labels=ids)
    outputs.loss.backward()
    emb = model.get_input_embeddings()
    grad = emb.weight.grad.detach().cpu().float().numpy().ravel().copy()
    return grad


def _unit(g: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(g)
    return g / n if n > 1e-10 else g


def compute_influence_matrix(
    checkpoint_paths: list[str],
    encodings: list[dict],
    config: InfluenceConfig,
    device: str,
) -> np.ndarray:
    D = len(encodings)
    T = len(checkpoint_paths)
    Phi = np.zeros((D, T), dtype=np.float32)

    for t, ckpt in enumerate(checkpoint_paths):
        model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
        model.eval()
        V = model.get_input_embeddings().weight.numel()

        # Pass 1: mean gradient
        mean_g = np.zeros(V, dtype=np.float64)
        for enc in encodings:
            g = _doc_gradient(model, enc["input_ids"], device)
            if config.normalize:
                g = _unit(g)
            mean_g += g
        mean_g /= D

        # Pass 2: per-doc score
        for i, enc in enumerate(encodings):
            g = _doc_gradient(model, enc["input_ids"], device)
            if config.normalize:
                g = _unit(g)
            Phi[i, t] = float(np.dot(g, mean_g))

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return Phi
```

- [ ] **Step 4: Smoke test**

```bash
uv run python -c "from influence_curriculum.score import InfluenceConfig, compute_influence_matrix; print('ok')"
```
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/influence_curriculum/score.py tests/test_correctness.py
git commit -m "feat: influence matrix scoring (InfluenceConfig, compute_influence_matrix)"
```

---

## Task 5: curriculum.py — sort and write epoch files

**Files:**
- Create: `src/influence_curriculum/curriculum.py`
- Modify: `tests/test_correctness.py` (add permutation + determinism tests)

**Interfaces:**
- Consumes: `Phi: np.ndarray (D×T)`, `texts: list[str]`, `doc_ids: list[str]`
- Produces:
  - `CurriculumConfig` dataclass
  - `build_curriculum(Phi, texts, doc_ids, config, output_dir, seed) -> None` — writes `output_dir/curriculum/epoch_NN.jsonl`

- [ ] **Step 1: Add permutation + determinism tests**

Append to `tests/test_correctness.py`:
```python
import json
from pathlib import Path
from influence_curriculum.curriculum import CurriculumConfig, build_curriculum


def test_permutation_validity(tmp_path):
    D, T = 20, 3
    rng = np.random.default_rng(42)
    Phi = rng.random((D, T)).astype(np.float32)
    texts = [f"doc {i}" for i in range(D)]
    doc_ids = [f"fake#{i}" for i in range(D)]
    build_curriculum(Phi, texts, doc_ids, CurriculumConfig(segment_size=5), str(tmp_path), seed=0)
    for e in range(T):
        path = tmp_path / "curriculum" / f"epoch_{e:02d}.jsonl"
        assert path.exists()
        ids = [json.loads(l)["id"] for l in path.read_text().splitlines()]
        assert sorted(ids) == sorted(doc_ids), f"epoch {e} is not a permutation"


def test_determinism(tmp_path):
    D, T = 10, 2
    rng = np.random.default_rng(7)
    Phi = rng.random((D, T)).astype(np.float32)
    texts = [f"doc {i}" for i in range(D)]
    doc_ids = [f"fake#{i}" for i in range(D)]
    cfg = CurriculumConfig()
    build_curriculum(Phi, texts, doc_ids, cfg, str(tmp_path / "r1"), seed=42)
    build_curriculum(Phi, texts, doc_ids, cfg, str(tmp_path / "r2"), seed=42)
    for e in range(T):
        f1 = (tmp_path / "r1" / "curriculum" / f"epoch_{e:02d}.jsonl").read_text()
        f2 = (tmp_path / "r2" / "curriculum" / f"epoch_{e:02d}.jsonl").read_text()
        assert f1 == f2, f"epoch {e} differs between identical runs"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_correctness.py::test_permutation_validity -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement curriculum.py**

```python
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CurriculumConfig:
    aggregation: str = "per_epoch_raw"   # "per_epoch_raw" | "lognormal"
    direction: str = "asc"               # "asc" | "desc"
    segment_size: int = 1000
    lognormal_window: int = 10
    lognormal_mu: float = 0.0
    lognormal_sigma: float = 1.0


def _lognormal_kernel(window: int, mu: float, sigma: float) -> np.ndarray:
    from scipy.stats import lognorm
    k = np.arange(1, window + 1, dtype=float)
    h = lognorm.pdf(k, s=sigma, scale=np.exp(mu))
    return h / h.sum()


def build_curriculum(
    Phi: np.ndarray,
    texts: list[str],
    doc_ids: list[str],
    config: CurriculumConfig,
    output_dir: str,
    seed: int,
) -> None:
    D, T = Phi.shape
    rng = random.Random(seed)
    out = Path(output_dir) / "curriculum"
    out.mkdir(parents=True, exist_ok=True)

    if config.aggregation == "lognormal":
        h = _lognormal_kernel(config.lognormal_window, config.lognormal_mu, config.lognormal_sigma)
        scores = np.zeros_like(Phi)
        for t in range(T):
            for k, hk in enumerate(h):
                if t - k >= 0:
                    scores[:, t] += Phi[:, t - k] * hk
    else:
        scores = Phi

    for e in range(T):
        col = scores[:, e]
        order = np.argsort(col)
        if config.direction == "desc":
            order = order[::-1]

        shuffled: list[int] = []
        for start in range(0, len(order), config.segment_size):
            seg = order[start : start + config.segment_size].tolist()
            rng.shuffle(seg)
            shuffled.extend(seg)

        with open(out / f"epoch_{e:02d}.jsonl", "w") as f:
            for idx in shuffled:
                f.write(json.dumps({"id": doc_ids[idx], "text": texts[idx]}) + "\n")
```

- [ ] **Step 4: Run all correctness tests**

```bash
uv run pytest tests/test_correctness.py -v
```
Expected: all PASSED (no training required).

- [ ] **Step 5: Commit**

```bash
git add src/influence_curriculum/curriculum.py tests/test_correctness.py
git commit -m "feat: curriculum construction (CurriculumConfig, build_curriculum)"
```

---

## Task 6: sort.py — public API

**Files:**
- Create: `src/influence_curriculum/sort.py`

**Interfaces:**
- Consumes: all four modules
- Produces: `sort_by_influence(model, dataset_dir, output_dir, *, checkpoints, tokenizer, data_config, training_config, influence_config, curriculum_config, seed, device) -> str`

- [ ] **Step 1: Implement sort.py**

```python
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .curriculum import CurriculumConfig, build_curriculum
from .data import DataConfig, load_documents
from .score import InfluenceConfig, compute_influence_matrix
from .train import TrainingConfig, train_surrogate


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sort_by_influence(
    model,
    dataset_dir: str,
    output_dir: str,
    *,
    checkpoints: list[str] | None = None,
    tokenizer=None,
    data_config: DataConfig | None = None,
    training_config: TrainingConfig | None = None,
    influence_config: InfluenceConfig | None = None,
    curriculum_config: CurriculumConfig | None = None,
    seed: int = 0,
    device: str = "auto",
) -> str:
    data_config = data_config or DataConfig()
    training_config = training_config or TrainingConfig()
    influence_config = influence_config or InfluenceConfig()
    curriculum_config = curriculum_config or CurriculumConfig()
    device = _resolve_device(device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(model, str):
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)
    elif tokenizer is None:
        raise ValueError("tokenizer is required when model is an nn.Module")

    # Phase 0: load + tokenize
    texts, doc_ids = load_documents(dataset_dir, data_config, tokenizer)
    encodings = [
        tokenizer(t, truncation=True, max_length=training_config.max_seq_len, return_tensors="pt")
        for t in texts
    ]

    # Phase 1: surrogate
    if checkpoints is None:
        checkpoints = train_surrogate(model, tokenizer, texts, str(out), training_config, seed, device)
    else:
        if len(checkpoints) == 1:
            warnings.warn("T=1: per-epoch curriculum collapses to a single ordering.", stacklevel=2)
        if len(checkpoints) != training_config.epochs:
            warnings.warn(
                "Supplied checkpoint count doesn't match training_config.epochs. "
                "If checkpoints were not produced by a random-order run on this data, "
                "results will diverge from the paper.",
                stacklevel=2,
            )

    # Phase 2: influence matrix
    Phi = compute_influence_matrix(checkpoints, encodings, influence_config, device)
    np.save(str(out / "influence_matrix.npy"), Phi)
    (out / "doc_ids.json").write_text(json.dumps(doc_ids))

    # Phase 3: curriculum
    build_curriculum(Phi, texts, doc_ids, curriculum_config, str(out), seed)

    # Persist resolved config
    (out / "config.json").write_text(json.dumps({
        "data_config": {k: str(v) for k, v in data_config.__dict__.items()},
        "training_config": training_config.__dict__,
        "influence_config": influence_config.__dict__,
        "curriculum_config": curriculum_config.__dict__,
        "seed": seed,
        "device": device,
        "num_docs": len(texts),
        "T": len(checkpoints),
    }, indent=2))

    return str(out)
```

- [ ] **Step 2: Verify import**

```bash
uv run python -c "from influence_curriculum import sort_by_influence; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/influence_curriculum/sort.py
git commit -m "feat: public sort_by_influence API"
```

---

## Task 7: Notebooks

**Files:**
- Create: `notebooks/01_data_inspect.ipynb`
- Create: `notebooks/02_influence_debug.ipynb`
- Create: `notebooks/03_curriculum_check.ipynb`

**Interfaces:**
- Consumes: installed `influence_curriculum` package + BabyLM data dir
- Produces: interactive debug environment for each phase

- [ ] **Step 1: Create 01_data_inspect.ipynb**

Create a notebook with these cells (in order):

Cell 1 (markdown): `# 01 — Data Inspection`

Cell 2 (code):
```python
from pathlib import Path
from influence_curriculum.data import DataConfig, load_documents

DATA_DIR = "../BabyLM-2026-Strict-Small"
```

Cell 3 (code) — raw file peek:
```python
for p in sorted(Path(DATA_DIR).glob("*.txt")):
    lines = p.read_text(errors="replace").splitlines()
    blank_lines = sum(1 for l in lines if not l.strip())
    print(f"{p.name:35s}  total_lines={len(lines):7d}  blank_lines={blank_lines:6d}")
    print("  first 3 non-empty:", [l[:80] for l in lines if l.strip()][:3])
    print()
```

Cell 4 (code) — test default segmentation:
```python
cfg = DataConfig(doc_boundary="line", min_doc_tokens=1)
texts, ids = load_documents(DATA_DIR, cfg)
from collections import Counter
source_counts = Counter(i.split("#")[0] for i in ids)
print(f"Total docs: {len(texts)}")
for src, cnt in sorted(source_counts.items()):
    print(f"  {src:25s}: {cnt:6d} docs")
```

Cell 5 (code) — per-source length distribution:
```python
import numpy as np
for src in source_counts:
    src_texts = [t for t, i in zip(texts, ids) if i.startswith(src)]
    lengths = [len(t.split()) for t in src_texts]
    print(f"{src:25s}: median={np.median(lengths):.0f} words, max={max(lengths)}")
```

- [ ] **Step 2: Create 02_influence_debug.ipynb**

Create a notebook with these cells:

Cell 1 (markdown): `# 02 — Influence Math Debug (synthetic data)`

Cell 2 (code) — mean-gradient identity:
```python
import numpy as np

rng = np.random.default_rng(0)
D, V = 10, 200
raw = rng.random((D, V)).astype(np.float32)
grads = raw / np.linalg.norm(raw, axis=1, keepdims=True)

efficient = grads @ grads.mean(axis=0)
pairwise  = np.array([(1/D)*(grads[i] @ grads.T).sum() for i in range(D)])

print("max abs diff:", np.abs(efficient - pairwise).max())
assert np.allclose(efficient, pairwise, rtol=1e-5), "identity FAILED"
print("identity OK")
```

Cell 3 (code) — tiny end-to-end Phi with a real (tiny) model:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from influence_curriculum.score import InfluenceConfig, compute_influence_matrix

MODEL = "sshleifer/tiny-gpt2"  # 2-layer toy model, downloads fast
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL)

synthetic_texts = ["The cat sat on the mat.", "Dogs are great pets.", "I love machine learning."]
encodings = [tokenizer(t, return_tensors="pt") for t in synthetic_texts]

# Save a fake checkpoint by just saving the untrained model
import tempfile, os
tmp = tempfile.mkdtemp()
model.save_pretrained(tmp)

Phi = compute_influence_matrix([tmp], encodings, InfluenceConfig(), device="cpu")
print("Phi shape:", Phi.shape)   # (3, 1)
print("Phi values:", Phi)
```

Cell 4 (code) — score distribution:
```python
import matplotlib
matplotlib.use("inline")
import matplotlib.pyplot as plt

plt.hist(Phi[:, 0], bins=20)
plt.xlabel("influence score"); plt.ylabel("count")
plt.title("Influence distribution (synthetic)")
plt.show()
```

- [ ] **Step 3: Create 03_curriculum_check.ipynb**

Create a notebook with these cells:

Cell 1 (markdown): `# 03 — Curriculum Validity Check`

Cell 2 (code) — load a saved Phi or generate synthetic:
```python
import numpy as np, json
from pathlib import Path

# Point to a real output_dir if you have one, otherwise use synthetic
PHI_PATH = None  # e.g. "output/run1/influence_matrix.npy"
IDS_PATH = None  # e.g. "output/run1/doc_ids.json"

if PHI_PATH and Path(PHI_PATH).exists():
    Phi = np.load(PHI_PATH)
    doc_ids = json.loads(Path(IDS_PATH).read_text())
else:
    rng = np.random.default_rng(99)
    D, T = 500, 3
    Phi = rng.random((D, T)).astype(np.float32)
    doc_ids = [f"fake#{i}" for i in range(D)]
    print(f"Using synthetic Phi {Phi.shape}")

print("Phi shape:", Phi.shape)
```

Cell 3 (code) — build curriculum and check permutation:
```python
import json, tempfile
from influence_curriculum.curriculum import CurriculumConfig, build_curriculum

tmp = tempfile.mkdtemp()
texts = [f"placeholder text {i}" for i in range(len(doc_ids))]
build_curriculum(Phi, texts, doc_ids, CurriculumConfig(), tmp, seed=0)

D, T = Phi.shape
for e in range(T):
    path = Path(tmp) / "curriculum" / f"epoch_{e:02d}.jsonl"
    ids = [json.loads(l)["id"] for l in path.read_text().splitlines()]
    ok = sorted(ids) == sorted(doc_ids)
    print(f"epoch {e:02d}: {len(ids)} docs, is_permutation={ok}")
```

Cell 4 (code) — source mix per epoch:
```python
from collections import Counter
import matplotlib.pyplot as plt

source_mix = []
for e in range(T):
    path = Path(tmp) / "curriculum" / f"epoch_{e:02d}.jsonl"
    ids = [json.loads(l)["id"] for l in path.read_text().splitlines()]
    counts = Counter(i.split("#")[0] for i in ids)
    source_mix.append(counts)

sources = sorted(source_mix[0].keys())
x = np.arange(T)
bottom = np.zeros(T)
for src in sources:
    vals = np.array([m.get(src, 0) for m in source_mix])
    plt.bar(x, vals / len(doc_ids), bottom=bottom, label=src)
    bottom += vals / len(doc_ids)

plt.xlabel("epoch"); plt.ylabel("fraction"); plt.legend(loc="upper right")
plt.title("Source mix per epoch"); plt.show()
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/
git commit -m "feat: debug notebooks for data, influence, and curriculum phases"
```

---

## Self-Review

**Spec coverage:**
- §4 signature: `sort_by_influence` in `sort.py` ✓; all config dataclasses present ✓
- §5 data contract: `load_documents`, `DataConfig`, doc IDs, min_doc_tokens, warning on low count ✓
- §6 Phase 1 Route A + Route B + warnings ✓
- §7 Phase 2 recompute route, mean-grad pass, per-doc pass, `Phi` saved ✓; `doc_ids.json` ✓
- §8 Phase 3 `C~` default, lognormal optional ✓; `epoch_NN.jsonl` with `id`+`text` ✓
- §10 output layout: `influence_matrix.npy`, `doc_ids.json`, `config.json`, `curriculum/` ✓
- §11 determinism: single `seed` drives all RNG ✓
- §12.1 correctness checks: mean-grad identity, permutation validity, determinism ✓
- Device-agnostic constraint: `_resolve_device`, no `.cuda()` literals ✓
- `fp16=False` default ✓

**Missing:** `README.md` in output_dir (§10) — intentionally deferred; the `config.json` covers reproducibility.
