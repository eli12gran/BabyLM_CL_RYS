# Pass-2 JVP Fix & vmap Restore Design

**Date:** 2026-06-19  
**Status:** Approved

## Problem

Two bugs compound to cause a crash + slow runtime in `compute_influence_matrix`:

1. **Wrong `attn_impl` condition** (`score.py:175`): `attn_impl` is set to `"sdpa"` whenever `pass2_batch_size=1`. But the per-doc `jvp` call also uses forward-mode AD, which is incompatible with sdpa's efficient attention kernel. Result: `NotImplementedError` at pass-2 start.

2. **`pass2_batch_size` regressed to 1**: Previous OOM debugging (vmap(fgrad) create_graph issue) led to setting `pass2_batch_size=1` as the safe default. The current `vmap(jvp)` implementation does not have that issue — it uses forward-mode AD with no backward graph. Keeping B=1 abandons the batched path and triggers bug #1.

## Fix

### Change 1 — `score.py:175` (1 token)

```python
# Before
attn_impl = "eager" if use_vmap_jvp else "sdpa"

# After
attn_impl = "eager" if use_jvp else "sdpa"
```

Forward-mode AD (jvp) requires `"eager"` attention regardless of whether vmap is used. SDPA's efficient kernel has no forward AD derivative registered.

### Change 2 — `InfluenceConfig.pass2_batch_size` default (`score.py:42`)

```python
# Before
pass2_batch_size: int = 1

# After
pass2_batch_size: int = 8
```

B=8 is a conservative T4-safe default (~315 MB peak for vmap(jvp)). The notebook overrides to B=32 (~705 MB peak), which also fits on T4 (15.6 GB) and gives ~10x speedup over per-doc loop.

## Memory Analysis (T4, 15.6 GB VRAM, vmap(jvp) with eager)

JVP forward-mode AD peak ≈ 2× one forward pass at batch size B (no backward graph, no activation storage for backprop):

| B | Approx peak | Fits T4? |
|---|-------------|----------|
| 1 | ~115 MB | ✓ |
| 8 | ~315 MB | ✓ |
| 32 | ~705 MB | ✓ |
| 64 | ~1.24 GB | ✓ |

## Expected Runtime (T4, 363k docs, 2 checkpoints)

| Pass | Before fix | After fix (B=32) |
|------|-----------|-----------------|
| Pass 1 | ~6 min | ~6 min (unchanged) |
| Pass 2 | crash | ~12 min |
| **Total** | **crash** | **~18 min** |

## Scope

No changes to: pass-1 logic, vmap(jvp) implementation, model loading, cleanup, notebook batch config.
