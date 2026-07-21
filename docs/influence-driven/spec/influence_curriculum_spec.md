# Spec: Influence-driven Curriculum Sorter

**Status:** Draft for implementation handoff
**Source method:** Schoenegger, Thoma, Blevins & Roth (2025), *Influence-driven Curriculum Learning for Pre-training on Limited Data* (arXiv:2508.15475v2 / BabyLM 2025).
**Goal:** A single function that takes a model + a directory of documents and produces an output directory containing a per-epoch training curriculum, sorted by training-data influence.

---

## 1. Purpose & scope

Implement `sort_by_influence(...)`. It runs the full three-phase pipeline from the paper:

1. **Train** a fresh surrogate model on the data in *random* order, saving one checkpoint per epoch.
2. **Score** every document with its average training-data influence at each checkpoint, producing an influence matrix `Φ ∈ R^{|D|×T}`.
3. **Sort** the data into a per-epoch curriculum using one default strategy and write it out.

### v1 decisions (locked)

| Decision | Choice |
|---|---|
| Surrogate boundary | Full pipeline by default; **optionally accept caller-supplied checkpoints** (custom training rule / BYO model) |
| Output granularity | Per-epoch orderings (one ordering per training epoch); requires `T ≥ 2` |
| Curriculum strategies | One default: `C~` — per-epoch influence sort + within-segment shuffle (see §8). Lognormal re-weighting `(C*h)~` is an optional knob, **off by default** |
| Objective | **CLM only** (causal LM, Llama-style). No MLM, no masking-reproducibility machinery |
| Input layout | One or more `.txt` files; **documents are segmented *within* files (file ≠ document)** |
| Default sort direction | **Ascending** influence |
| Influence gradient | Dense standard-backprop gradient of the **input-embedding matrix**, cosine-normalized (see §3) |

### Non-goals for v1

- MLM / dynamic-masking support.
- Cumulative-coverage curricula (`C^E`, `C_A`), data-discarding curricula (`C^{50}`), or human-centered baselines (`Csource`, `CMATTR`, `CPPL`).
- Distributed multi-node training. Single-node, one-or-more-GPU is enough.
- Consuming the curriculum / running the downstream training. We only **emit** the ordering (plus a documented consumer contract).

> **Important — how the model is used:** by default the weights of the passed-in `model` are **not** used as a starting point. The surrogate is trained from initialization on *this* dataset in random order, because the influence scores are derived from that random-order checkpoint trajectory. `model` then supplies architecture, config, and tokenizer.
>
> Callers who have a **custom training rule** (or want to bring their own model) may instead pass an ordered list of `checkpoints` (see §4). The function then skips Phase 1 and scores those checkpoints directly. Caveats:
> - The method sums influence over a **checkpoint trajectory**, so a single checkpoint means `T = 1`: cross-epoch aggregation degenerates and the per-epoch curriculum collapses to one global ordering. The headline per-epoch behavior needs `T ≥ 2`.
> - If the supplied checkpoints did **not** come from a random-order run on this data (e.g. a generic pretrained base), the scores measure something different from the paper and results will not match it. Emit a warning in this case.

### Implementation constraint — device-agnostic

Write the code to run on a **single device** of any kind: accept a `device` arg resolving to `cpu` / `mps` (Apple Silicon) / `cuda`. Do **not** hardcode CUDA (`.cuda()`, `cuda:0` literals) and do **not** *assume* multiple GPUs — single-device must be a correct, complete code path; multi-GPU parallelism (e.g. scoring checkpoints concurrently) is an optional optimization layered on top, never a requirement. For Apple MPS specifically: keep `fp16=False` (already the default; fp16/bf16 on MPS is unreliable), rely on the plain backward-pass-per-document loop in §7 rather than `torch.func`/`vmap` per-sample-gradient tricks (MPS coverage gaps), and expect `PYTORCH_ENABLE_MPS_FALLBACK=1` to be set for any unsupported ops. This keeps "develop on a laptop, run for real on a GPU" a single code path.

---

## 2. Resolved defaults (source code unavailable)

The authors' released code (Zenodo DOI `10.5281/zenodo.16919045`; HF collection `loris3/ticl-68a6fd8bcc3093f239439e42`) could not be retrieved, so the three implicit details below are **decided here** rather than verified against their implementation. Each is exposed as a knob so it can be A/B-tested (see §12), and correctness is established empirically (curriculum beats random order) rather than by byte-matching their `Φ`.

1. **Influence gradient target — DECIDED: dense, standard backprop, input-embedding matrix.** Take the gradient of the CLM loss w.r.t. the input-embedding matrix as produced by a normal backward pass. Because the paper's models tie input/output embeddings, that gradient is **dense over the whole vocabulary** (the tied output head contributes to every vocab row). This is the most literal reading of "the model's input embeddings," and the paper's own footnote notes the score "incorporates information about the full model, as the gradient chains through higher layers" — i.e. the kept embedding gradient already reflects the whole network. For Llama this is unambiguous: positions are rotary (RoPE), so the input embedding is exactly the token-embedding matrix and nothing else. *Alternative to test: input-lookup path only (sparse over the doc's tokens), per the "first is better than last" purity argument.*
2. **Cross-epoch aggregation — DECIDED: none by default (`C~`).** Each epoch sorts on its own checkpoint's raw influence column; no lognormal filter. The paper found the lognormal re-weighting had a marginal, non-significant effect, so it is an optional knob (§8), not the default. This removes the unknown filter parameters from the critical path.
3. **Normalization — DECIDED: cosine.** Unit-normalize each gradient *before* the dot product (paper's stated fix for length bias). The dataset mean is the mean of unit gradients; the per-doc score is `unit_grad · mean(unit_grads)`.

---

## 3. The influence score (math)

Per checkpoint `t`, the average influence of document `z` over the dataset `D` (paper Eq. 2→3) is:

```
φ_t(z, D) = (1/|D|) · Σ_{z'∈D}  ∇ℓ(w_t, z) · ∇ℓ(w_t, z')
          = ∇ℓ(w_t, z) · E_{z'∼D}[ ∇ℓ(w_t, z') ]
```

i.e. **each document's loss-gradient dotted with the dataset's mean gradient.** This is the key efficiency lever: O(|D|) per checkpoint, not O(|D|²).

- `w_t` = the model's **input-embedding matrix** at checkpoint `t`. The gradient is the one a standard backward pass produces for that matrix; with tied embeddings it is dense over the vocabulary (see §2). The "chains through higher layers" point (paper footnote) means this single matrix's gradient already reflects the full network via the chain rule — it does **not** mean using the gradient of all parameters; we still restrict to the input-embedding matrix.
- `ℓ` = standard CLM next-token cross-entropy loss for the document.
- **Normalization (cosine):** each gradient is unit-normalized before use, to remove the length bias the paper observed: `ĝ = ∇ℓ / ||∇ℓ||`. Then `φ_t(z,D) = ĝ_z · mean_{z'}(ĝ_{z'})`.

Stacking over `T` checkpoints gives the matrix `Φ[i, t]`.

---

## 4. Function signature

```python
def sort_by_influence(
    model,                      # nn.Module OR model id/dir. Architecture + config only; weights are NOT reused by default.
    dataset_dir: str,           # directory containing one or more .txt files (segmented into documents)
    output_dir: str,            # destination for curriculum + artifacts
    *,
    checkpoints: list[str] | None = None,  # optional ordered checkpoint paths; if given, Phase 1 is skipped
    tokenizer=None,             # defaults to the model's own tokenizer
    data_config: DataConfig = DataConfig(),
    training_config: TrainingConfig = TrainingConfig(),
    influence_config: InfluenceConfig = InfluenceConfig(),
    curriculum_config: CurriculumConfig = CurriculumConfig(),
    seed: int = 0,
    device: str = "auto",       # "auto" | "cpu" | "mps" | "cuda[:N]"; single-device must work
) -> str:                       # returns output_dir
    ...
```

### Config objects (with defaults from the paper's Table 3 where applicable)

```python
@dataclass
class DataConfig:
    # Single rule applied to all files, OR a dict {filename_or_glob: rule} with a default fallback.
    # Each rule is "line" | "blank_line" | "whole_file" | <callable str -> list[str]>.
    doc_boundary: "str | dict[str, str | Callable]" = "line"
    default_boundary: str = "line"   # fallback for files not matched by the dict
    min_doc_tokens: int = 1          # drop segments shorter than this
    chunking: str = "truncate"       # "truncate" | "chunk" — applied AFTER segmentation (see §5)
    # Document IDs are "<file_stem>#<segment_index>"

@dataclass
class TrainingConfig:
    epochs: int = 10                 # == T, the number of checkpoints
    effective_batch_size: int = 2048 # paper: per-device 32 × grad-accum 16 × 4 GPUs
    per_device_batch_size: int = 32
    learning_rate: float = 7e-4      # CLM / Llama default
    lr_scheduler: str = "cosine"
    optimizer: str = "adamw"
    adam_betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-6
    weight_decay: float = 0.01
    max_seq_len: int = 256           # paper Llama max position embeddings
    fp16: bool = False

@dataclass
class InfluenceConfig:
    grad_target: str = "input_embeddings"  # model.get_input_embeddings().weight gradient (dense w/ tied embeds)
    grad_path: str = "full"                # "full" (standard backprop, default) | "input_lookup_only" (sparse, to test)
    normalize: bool = True                 # cosine similarity (unit-normalize each gradient before the dot)
    memory_route: str = "recompute"        # "recompute" (low mem, 2× compute) | "cache" (store ĝ)
    projection_dim: int = 0                # >0 enables random projection of gradients (TRAK/LESS-style)
    grad_batch_size: int = 16              # docs per backward pass during scoring

@dataclass
class CurriculumConfig:
    aggregation: str = "per_epoch_raw"  # "per_epoch_raw" (C~, default) | "lognormal" ((C*h)~)
    direction: str = "asc"              # "asc" | "desc"
    segment_size: int = 1000            # within-segment shuffle window
    # Used only when aggregation == "lognormal" (optional, untested params — see §8):
    lognormal_window: int = 10          # causal filter length (≤ T)
    lognormal_mu: float = 0.0           # placeholder; tune empirically
    lognormal_sigma: float = 1.0        # placeholder; tune empirically
```

---

## 5. Data contract (loading + segmentation)

- **Input:** `dataset_dir` contains one or more `.txt` files. **A file is a *source*, not a document.** BabyLM-style corpora ship as a handful of large files (e.g. ~6: `childes.txt`, `bnc_spoken.txt`, `gutenberg.txt`, `open_subtitles.txt`, `simple_wiki.txt`, `switchboard.txt`), each containing thousands of documents.
- **Segmentation** (`data_config.doc_boundary`): each file is split into documents:
  - `"line"` — one document per non-empty line (good default for utterance/sentence-style corpora).
  - `"blank_line"` — one document per blank-line-separated block (good for paragraph/long-form corpora).
  - `"whole_file"` — the whole file is one document (only sensible if files are already per-document).
  - a custom callable `str -> list[str]` for source-specific rules.
  - The right rule is often **source-specific**, so `doc_boundary` may be a **dict keyed by filename/glob** (e.g. `{"gutenberg.txt": "blank_line", "childes.txt": "line"}`) with `default_boundary` covering the rest. This makes segmentation variations a config change, which the testing strategy (§12) sweeps over.
- **Document ID:** `"<file_stem>#<segment_index>"`, stable across runs. Must survive into the output manifest and `doc_ids.json`.
- **Expected counts:** the algorithm scales with **document count**, not file count. At 10M words this is typically tens of thousands of documents. Sanity-check at load time: if the total document count is below, say, a few thousand, warn — the 1000-doc segments (§8) and per-document averaging assume many units.
- **Length handling** (`data_config.chunking`, applied *after* segmentation):
  - `"truncate"` — each document is one unit, truncated to `max_seq_len`. Default; matches the paper's per-document framing.
  - `"chunk"` — documents longer than `max_seq_len` are split into windows, each a separate unit with ID `<stem>#<seg>#<k>`. Changes the influence units and the manifest; document clearly.
- Empty / all-whitespace segments and segments below `min_doc_tokens` are dropped with a logged count.

---

## 6. Phase 1 — obtain the checkpoint trajectory

Influence summing needs `T` checkpoints. Two routes:

**Route A — default trainer (no `checkpoints` arg).**
- Initialize a fresh model from the passed architecture/config (ignore any provided weights).
- Train for `epochs` (= `T`) full passes over **all** documents in **random** order (seeded).
- Save a checkpoint **after each epoch** → `T` checkpoints total.
- Use the `TrainingConfig` recipe; everything is overridable. Stochasticity is driven by `seed`.

**Route B — caller-supplied checkpoints (`checkpoints=[...]`).**
- Skip training entirely; use the provided ordered list as the trajectory (`T = len(checkpoints)`).
- This is the path for a **custom training rule** or a BYO model.
- Validate: each checkpoint is loadable into the given architecture and shares the tokenizer/vocab. If `T == 1`, log the degraded-mode warning (per-epoch curriculum collapses to one ordering; `(C*h)` filter is identity). If the checkpoints were not produced by a random-order run on this data, warn that results will diverge from the paper.

Checkpoints (Route A) are intermediates kept on disk, optionally cleaned after Phase 2 unless `--keep-checkpoints` is set. Route-B checkpoints are never modified or deleted.

---

## 7. Phase 2 — influence estimation

For each checkpoint `t ∈ {1..T}`, compute `Φ[:, t]` using the **recompute** route by default:

1. **Pass 1 — mean gradient.** For every document, compute the CLM loss gradient w.r.t. `grad_target`, unit-normalize it, and accumulate into a running sum. Divide by `|D|` to get the dataset mean gradient `ḡ_t`.
2. **Pass 2 — scores.** For every document, recompute its unit-normalized gradient `ĝ_i` and set `Φ[i, t] = ĝ_i · ḡ_t`.

Implementation notes:

- **Tied embeddings (critical):** if the input embedding is tied to the LM head, the gradient w.r.t. that parameter is **dense over the full vocab** (every vocab row participates in every position's softmax). The recompute route keeps memory bounded regardless: you hold one dense `vocab × hidden` mean-gradient buffer plus the current document's gradient. Do **not** attempt to store every document's dense gradient unless `memory_route="cache"` with `projection_dim>0`.
- **`memory_route="cache"`** stores each `ĝ_i` (sparse if untied; otherwise use `projection_dim>0` to random-project to a fixed small dimension à la TRAK/LESS) so Pass 2 becomes a single dot rather than a second backward. Faster, more memory/disk.
- **Parallelism (optional):** checkpoints are independent, so scoring them concurrently across multiple devices is a valid optimization where hardware allows — but single-device must work on its own (see the device-agnostic constraint in §1). This is the expensive phase (paper reports tens of GPU-hours per dataset), so make the device/batch knobs first-class.
- **`grad_batch_size`** controls docs per backward pass during scoring (per-example gradients, so accumulate carefully — do not average them into a single batch gradient).

Output of this phase: `Φ` (shape `|D| × T`), saved as `influence_matrix.npy`, with a sidecar `doc_ids.json` mapping row index → document ID.

> **Caching note (used by §12):** `Φ` depends only on `(model, data, segmentation, gradient target, training recipe, seed)` — **not** on the curriculum knobs (aggregation, direction, segment size). Compute `Φ` once and reuse it across all curriculum-knob variations. Key the cached `Φ` by a hash of those inputs.

---

## 8. Phase 3 — curriculum construction (`C~` default, ascending)

Per-epoch, full-coverage strategy. The ordering for an epoch comes entirely from **sorting by influence + a local shuffle** — the (optional) cross-epoch filter only changes *which score* you sort on, it does not create the order. For each epoch `e ∈ {1..T}`:

1. **Pick the score column.**
   - `aggregation = "per_epoch_raw"` (default, `C~`): use that epoch's own checkpoint column, `Φ[e, :]`.
   - `aggregation = "lognormal"` (optional, `(C*h)~`): use a causal lognormal-smoothed column `Ch[e,i] = Σ_{k=0}^{window-1} Φ[e−k, i]·h[k]`, where `h` is a lognormal kernel. This upweights documents that stay influential across epochs. Kernel params are untested placeholders (see §2) and the paper found the effect marginal/insignificant — treat as experimental.
2. **Sort** all documents by the chosen score in `direction` order (default ascending).
3. **Segment + shuffle:** split the sorted list into contiguous segments of `segment_size` (1000) documents, then shuffle documents *within* each segment (seeded). This local shuffle — not the sort direction — is what the paper credits for the gains (it groups similar-influence docs into a batch while adding diversity).
4. The result is epoch `e`'s ordering: a full pass over every document.

This yields `T` distinct epoch orderings. They differ across epochs because each epoch sorts on a different checkpoint's scores — **not** because of the filter. Disabling the (optional) lognormal aggregation therefore yields the well-tested `C~` curriculum, **not** random order. Direction (`asc`/`desc`) has, per the paper, no consistent effect; it is exposed but defaults to `asc`.

---

## 9. Consumer contract (document prominently)

Each materialized output directory is a **self-contained dataset version**. The downstream training loop:
- Reads `curriculum/epoch_00.jsonl` … `epoch_{T-1}.jsonl` in order, one file per epoch.
- Within a file, consumes documents **in the given line order** at a fixed batch size.
- Does **not** reshuffle within an epoch.

Reshuffling destroys the influence-based batch grouping, which is the mechanism behind the benefit. Because the text is inline (one document per line), consumption needs no ID resolution or access to the source corpus. The README in `output_dir` restates this contract.

---

## 10. Output layout (v1: materialized, self-contained)

Each call writes one self-contained version directory to `output_dir`. Multiple versions are just multiple output dirs the dev places wherever they like.

```
output_dir/                        # one self-contained dataset version
├── README.md                      # consumer contract + run summary
├── config.json                    # fully-resolved config + seed + library versions
├── influence_matrix.npy           # Φ, shape |D| × T  (kept for analysis)
├── doc_ids.json                   # row index → document ID
├── curriculum/
│   ├── epoch_00.jsonl             # documents IN ORDER, text inline: {"id": "...", "text": "..."} per line
│   ├── epoch_01.jsonl
│   └── ... epoch_{T-1}.jsonl
└── checkpoints/                   # optional; only if --keep-checkpoints (Route A)
```

- **Materialization is the default and only mode in v1.** Each epoch file is self-contained (text inline), so the content is duplicated ~`T`× across the epoch files. This is accepted: at BabyLM scale the data is small and devs materialize only a handful of versions. (See §14 for the deferred no-duplication design.)
- Each `epoch_NN.jsonl` contains **every document exactly once**, in that epoch's curriculum order. The `id` field preserves traceability back to `<file_stem>#<segment_index>`.
- `config.json` must capture everything needed to reproduce the run given the same data and seed.

---

## 11. Determinism

A single `seed` drives: surrogate initialization, random-order training shuffles, Phase-2 batching order, and Phase-3 within-segment shuffles. Same `(data, seed, config)` → identical `Φ` and identical curriculum. Under **Route B** (caller-supplied checkpoints) the training determinism is the caller's responsibility; given fixed checkpoints, Phases 2–3 remain deterministic under `seed`.

---

## 12. Testing & evaluation strategy

The real objective is **downstream gain over random order**, so the comparison framework matters as much as the implementation. We can't byte-match the authors' `Φ` (code unavailable), so correctness is established by internal consistency + empirical benefit. Structure it in three tiers.

### 12.1 Correctness checks (fast, no training)
- **Mean-gradient identity:** on a tiny synthetic `D`, the explicit `(1/|D|)Σ pairwise` dot-product equals `ĝ · mean(ĝ)`. Guards the core efficiency trick.
- **Permutation validity:** each `epoch_NN.jsonl` is a permutation of the full document set (every doc exactly once); `T` files present; ids resolve to `doc_ids.json`.
- **Determinism:** same `(data, seed, config)` → byte-identical `Φ` and `curriculum/`.
- **Segmentation sanity:** document counts per source are logged; warn if total is implausibly low (segmentation likely didn't fire).

### 12.2 Intrinsic curriculum diagnostics (cheap, no training)
Characterize a variation without paying for a full training run — useful for fast iteration over knobs:
- **Source mix over epochs** (reproduce the paper's Fig. 4 idea): proportion of each source per epoch. Surfaces the known **bias toward document-dense sources** so you can see if a segmentation choice skews the mix.
- **Documents-per-source** vs words-per-source, to expose count/word imbalance.
- **Kendall-τ** of the ordering against simple heuristics (length, unigram perplexity) and against random — sanity that the ordering is structured, not noise.
- **Jensen–Shannon divergence** between a variation's per-epoch source distribution and the random baseline's.

### 12.3 Downstream evaluation (decisive, expensive)
The acceptance bar. Train a model on the curriculum and measure benchmark macro-accuracy **relative to a random-order baseline trained on the same data, recipe, and seed**.
- **Baseline is mandatory and matched:** the metric is *gain over random*, so every variation is paired with a random-order run that differs in nothing but the ordering.
- **Multiple seeds + significance:** the paper reports many *non-significant* results, so single-run deltas are misleading. Run ≥3 seeds per variation and report mean ± spread with a significance test (the paper used p-values; mirror that).
- **Benchmarks:** the BabyLM eval suite the paper used (BLiMP, BLiMP-supplement, EWOK, (Super)GLUE, entity tracking, adjective nominalization). Start with a fast subset for iteration, full suite for final numbers.

### 12.4 Sweeping knobs efficiently
- **Reuse `Φ`:** per §7, `Φ` depends on `(model, data, segmentation, gradient target, recipe, seed)` but not on aggregation/direction/segment_size. Compute `Φ` once, then generate and evaluate many curriculum-knob variations on top of it. Only segmentation- or gradient-target changes force a rescore; only recipe/model/data changes force retraining the surrogate.
- **One knob at a time:** vary a single knob from a fixed reference config; don't co-vary, or you can't attribute the effect.
- **Bookkeeping:** append each run (resolved config + metrics) to a simple `results.csv`/`results.jsonl` so variations are comparable apples-to-apples. (This is the lightweight precursor to the deferred run registry in §14.)

### Knobs under test
gradient `grad_path` (`full` vs `input_lookup_only`) · `normalize` on/off · `aggregation` (`per_epoch_raw` vs `lognormal`, incl. kernel params) · `direction` · `segment_size` · `doc_boundary` per source · `chunking`/`max_seq_len`. The reference config is all defaults from §4.

---

## 13. Open items to resolve during implementation

- **Per-source `doc_boundary` rules** for the first real dataset (e.g. `blank_line` for Gutenberg, `line` for CHILDES/Switchboard/OpenSubtitles) — set from inspecting the actual files; this is the highest-impact knob and is driven by your data, not the missing code.
- Whether `chunk` mode is needed for very long sources, or `truncate` suffices.
- Lognormal kernel params **if** the optional `(C*h)~` aggregation is ever turned on — tune empirically (default is off).
- Whether to validate the `input_lookup_only` (sparse) gradient path against the default dense path on a small downstream run.

---

## 14. Deferred to later (explicitly NOT in v1)

Park these until sweeps or dataset size make them worthwhile:

- **No-duplication two-tier layout.** Write each document's content once into a content store keyed by a dataset+segmentation fingerprint; make per-run/per-epoch outputs lightweight pointer lists into it. Removes the ~`T`× duplication and lets variations that share a dataset share content.
- **Run registry + config-hash caching.** A `runs/registry.jsonl` plus `run_id = hash(model, dataset_ref, configs, seed)` so identical re-runs are cache hits and variations are easy to enumerate/compare.
- **`materialize` mode switch** (`"linked"` vs `"standalone"`) once the two-tier store exists, so a single version can still be exported as a portable self-contained directory on demand.

v1 deliberately uses the simple materialized layout in §10 instead.
