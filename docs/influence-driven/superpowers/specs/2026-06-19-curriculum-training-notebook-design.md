# Curriculum Training Notebook Design

**Date:** 2026-06-19  
**Status:** Approved

## Goal

A new notebook (`notebooks/05_curriculum_training.ipynb`) that pulls curriculum JSONL files from HF Hub, trains a fresh GPT-2 model using curriculum learning (multi-phase), saves per-epoch checkpoints to HF Hub for resume support, and plots training curves at the end.

## Context

The influence pipeline (`notebooks/04_colab_pipeline.ipynb`) produces curriculum JSONL files via `build_curriculum` with the `cumulative` strategy. With `n_groups=2` this yields:

- `epoch_00.jsonl` — easy docs only (~50% of corpus)
- `epoch_01.jsonl` — all docs (~100% of corpus)

The curriculum-training notebook consumes these files, trains a model in two phases (e.g. 5 epochs on `epoch_00`, 5 epochs on `epoch_01`), and pushes the result to HF Hub.

## Architecture

### New function: `train_curriculum` in `src/influence_curriculum/train.py`

```python
def train_curriculum(
    model: torch.nn.Module,
    tokenizer,
    phases: list[tuple[str, int]],   # [(jsonl_path, n_epochs), ...]
    output_dir: str,
    config: TrainingConfig,
    seed: int,
    device: str,
    start_phase: int = 0,            # resume: skip completed phases
    start_epoch: int = 0,            # resume: skip completed epochs in start_phase
    hub_repo: str | None = None,
    hub_token: str | None = None,
) -> tuple[list[str], list[dict]]    # (checkpoint_paths, history)
```

**Behavior:**
- Builds one `AdamW` optimizer and one cosine LR scheduler over the **total step count** across all phases — smooth LR curve across the phase boundary.
- Iterates phases in order; within each phase iterates epochs.
- Skips phases/epochs before `start_phase`/`start_epoch` (resume fast-forward).
- After each epoch: saves checkpoint locally as `phase_{p:02d}_epoch_{e:02d}`, pushes to `hub_repo` if set.
- Appends `{"phase": p, "epoch": e, "loss": float, "lr": float}` to `history` each epoch.
- Returns `(checkpoint_paths, history)`.

**Optimizer/scheduler on resume:** model weights are restored from the latest Hub checkpoint before calling `train_curriculum`. The optimizer and scheduler are re-initialized but fast-forwarded to the correct step count so the LR position is accurate. Optimizer momentum is not restored (acceptable trade-off for simplicity).

### Checkpoint naming

`phase_{p:02d}_epoch_{e:02d}` — e.g. `phase_00_epoch_04`, `phase_01_epoch_02`.

Stored under `{output_dir}/checkpoints/` locally and mirrored to `hub_repo` after each epoch.

## Notebook Structure (`notebooks/05_curriculum_training.ipynb`)

| Cell | Label | Purpose |
|------|-------|---------|
| 1 | Setup | RunPod clone/install/path setup (commented, same pattern as nb 04) |
| 2 | Config | `HUB_CURRICULUM_REPO`, `HUB_MODEL_REPO`, `HF_TOKEN`, `PHASE_EPOCHS=[5,5]`, model config, batch sizes |
| 3 | Imports + device | Imports, device detection, path setup |
| 4 | Pull curriculum | Download JSONL files from `HUB_CURRICULUM_REPO` to `/workspace/curriculum/` |
| 5 | Resume check | Scan `HUB_MODEL_REPO` for latest `phase_XX_epoch_YY` checkpoint; set `start_phase`/`start_epoch`; load weights if resuming |
| 6 | Train | Build fresh `GPT2LMHeadModel(GPT2Config(...))`, call `train_curriculum(...)` |
| 7 | Push final model | Upload last checkpoint to Hub with a `final` tag/alias |
| 8 | Metrics | Loss, perplexity, LR curves with vertical line at phase boundary |

## Config (Cell 2 defaults)

```python
HUB_CURRICULUM_REPO = None   # "yourname/babylm-curriculum-run001"
HUB_MODEL_REPO      = None   # "yourname/babylm-trained-run001"
HF_TOKEN            = None

PHASE_EPOCHS  = [5, 5]       # epochs per curriculum phase
MAX_SEQ_LEN   = 128
SEED          = 0

PASS_BATCH_SIZE      = 64    # per-device batch size (A40 48 GB safe)
EFFECTIVE_BATCH_SIZE = 256
LEARNING_RATE        = 7e-4
```

Model config (same as surrogate):
```python
GPT2Config(n_embd=384, n_layer=8, n_head=6, n_inner=1536, vocab_size=16384, n_positions=1024)
```

## Metrics Cell (Cell 8)

Three subplots sharing the x-axis (epoch number 0–9):
1. Training loss per epoch
2. Perplexity (exp(loss)) per epoch
3. Learning rate per epoch

A vertical dashed line marks the phase boundary (after epoch `PHASE_EPOCHS[0] - 1`).

## Scope

- No validation set / held-out perplexity (future work)
- No per-step loss logging (epoch-level only)
- No multi-GPU / DDP (single A40)
- Optimizer state not saved on Hub (model weights only)
