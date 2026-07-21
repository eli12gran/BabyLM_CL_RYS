# Design: Influence-driven Curriculum Sorter

**Date:** 2026-06-17
**Status:** Approved
**Ref spec:** `docs/spec/influence_curriculum_spec.md`

## Layout

```
src/influence_curriculum/
    data.py        # segmentation, DataConfig, doc IDs
    train.py       # Phase 1: surrogate training, checkpoint saving
    score.py       # Phase 2: influence matrix Φ (mean-grad + per-doc dot)
    curriculum.py  # Phase 3: sort + segment-shuffle + write epoch_NN.jsonl
    sort.py        # public API: sort_by_influence()
notebooks/
    01_data_inspect.ipynb      # validate segmentation rules per source
    02_influence_debug.ipynb   # verify Φ math on synthetic data
    03_curriculum_check.ipynb  # permutation validity + source-mix diagnostics
tests/
    test_correctness.py        # §12.1 fast checks (no training)
```

## Dependencies (UV)

- `torch` — training + gradient computation
- `transformers` — model/tokenizer
- `numpy` — Φ storage
- `jupyterlab` — notebooks (dev group)

## Data flow

```
dataset_dir (.txt files)
  → data.py: segment → doc list + doc_ids
  → train.py: random-order surrogate → T checkpoints
  → score.py: per-checkpoint mean-grad + dot → Φ [|D|×T]
  → curriculum.py: sort + segment-shuffle → epoch_NN.jsonl × T
  → output_dir/
```

## Key decisions (from spec §2)

- CLM only (Llama-style), device-agnostic (cpu/mps/cuda)
- Default curriculum: `C~` (per-epoch influence sort + 1000-doc segment shuffle, ascending)
- Influence: dense input-embedding gradient, cosine-normalized
- Lognormal re-weighting: optional knob, off by default

## Segmentation (open item)

Per-source `doc_boundary` rules to be finalized after `git lfs pull` resolves childes/gutenberg/open_subtitles. Known so far:
- `switchboard`, `bnc_spoken`: `"line"` (one utterance per line)
- `simple_wiki`: custom callable splitting on `= = = Title = = =` headers
- `childes`, `gutenberg`, `open_subtitles`: TBD after LFS pull

## Notebooks purpose

| Notebook | Phase tested | No training needed? |
|---|---|---|
| 01_data_inspect | data.py | yes |
| 02_influence_debug | score.py | yes (synthetic D) |
| 03_curriculum_check | curriculum.py | yes (load saved Φ) |

## Out of scope (v1)

- MLM/RoBERTa, cumulative curricula, data-discarding curricula
- Downstream eval harness (second iteration)
- Multi-node distributed training
