# How `flakoash/babylm-curriculum-tiered-4bands` Was Built

This document reconstructs, step by step, the exact pipeline that produced the 4-band
tiered curriculum dataset used to train **run002** (see `docs/results.md`, "Run 002 —
Influence-Driven Curriculum (4-band tiered)").

**Why this document exists as a reconstruction rather than a copy of a run log:** the
notebook that actually produced this specific dataset was executed outside of this
repo's committed history — no notebook cell in this repo shows saved output for the
exact run that built `tiered-4bands`. `notebooks/06_sequential_bands_curriculum.ipynb`
(the code that implements the `sequential` band-building algorithm used here) was run
to produce this dataset first, then committed to git afterward — the commit
(`b119622`, 2026-06-26) postdates the dataset's creation on HF (2026-06-23) simply
because of that commit lag, not because the code changed in between. This was
confirmed directly by the project author. Every fact below is either (a) read directly
from this repo's code, (b) read directly from the produced Hugging Face Hub artifacts
(treated as ground truth), (c) inferred from consistency between the two, or (d)
confirmed directly by the author, and each claim below is labeled accordingly. Section
11 lists everything that still could not be confirmed by any of these.

Confidence labels used throughout: **[CONFIRMED — code]**, **[CONFIRMED — Hub data]**,
**[CONFIRMED — author]**, **[INFERRED]**, **[UNKNOWN]**.

---

## 0. Paper and overall approach

**[CONFIRMED — code/docs]** The pipeline implements Schoenegger, Thoma, Blevins & Roth
(2025), *"Influence-driven Curriculum Learning for Pre-training on Limited Data"*
(arXiv:2508.15475v2), a BabyLM 2025 workshop paper. Reference digest:
`docs/influence_curriculum_paper_reference.md`; implementation spec:
`docs/spec/influence_curriculum_spec.md`.

Overall approach (paper, adapted here):
1. Train a small "surrogate" model on the corpus in random order, saving checkpoints.
2. Score every document by its **influence** — how much it pulls the model's gradient
   toward the dataset's mean gradient — using those checkpoints (TracInCP-style,
   Pruthi et al. 2020).
3. Reorder the corpus by that score into a curriculum (here: N discrete difficulty
   bands) and train a fresh model on the bands in easy → hard order.

The five pipeline stages, and the code/artifact responsible for each:

| Stage | Code | Output artifact |
|---|---|---|
| 1. Load + clean corpus | `src/influence_curriculum/data.py` (`load_documents`), driven by `notebooks/04_colab_pipeline.ipynb` cell 4 | in-memory `texts`, `doc_ids` |
| 2. Train surrogate model | `src/influence_curriculum/train.py` (`train_surrogate`), driven by `notebooks/04_colab_pipeline.ipynb` cell 6 | `flakoash/babylm-surrogate-run07` (2 checkpoints) |
| 3. Tokenize + score documents | notebook 04 cell 7 (tokenize) → `src/influence_curriculum/score.py` (`compute_influence_matrix`) | `flakoash/babylm-curriculum-run07`: `influence_matrix.npy`, `doc_ids.json` |
| 4. Build bands | `notebooks/06_sequential_bands_curriculum.ipynb`, cell 5 (`STRATEGY="sequential"`) | `flakoash/babylm-curriculum-tiered-4bands`: `curriculum/epoch_00..03.jsonl` |
| 5. Train on bands | `train_word_aware` (`train.py:307-552`), driven by `notebooks/07_train.ipynb` | `flakoash/babylm-gpt2-small-run002` |

`notebooks/04_colab_pipeline.ipynb` is written as a resumable, re-runnable "one notebook,
one `RUN_ID`" script (cell 2's config sets `RUN_ID`, and derives
`HUB_MODEL_REPO=f"flakoash/babylm-surrogate-run{RUN_ID}"` /
`HUB_DATASET_REPO=f"flakoash/babylm-curriculum-run{RUN_ID}"`). Re-running it with
`RUN_ID` bumped from `"05"` → `"06"` → `"07"` (and `SMOKE_TEST` eventually turned off)
is what produced the `run05`/`run06`/`run07` family on the Hub — see §3 for the direct
evidence. Its **committed state** in this repo, however, still shows `RUN_ID="05"`,
`SAMPLE_FRAC=1.0`, `SMOKE_TEST=True`, `SMOKE_N_DOCS=300` (cell 2) with `execution_count:
null` on every cell — i.e. it was reset to its smoke-test defaults and never
saved-with-output in git after (or between) real runs. Nothing here is a hidden/other
script: the notebook is the real pipeline source, just not committed in the exact state
it was run in for run07.

---

## 1. Source data

**[CONFIRMED — code/docs]** BabyLM 2026 Strict-Small track, the 10M-word corpus
(`BabyLM-2026-Strict-Small/README.md:26-38`), local mirror at
`BabyLM-2026-Strict-Small/` and `datasets/BabyLM-2026-Strict-Small/`. Six source files,
each already toxic-content-decontaminated upstream by the BabyLM organizers (this
decontamination is **not** something this repo's code does — it's inherited metadata
on the source corpus):

- `bnc_spoken.train.txt`
- `childes.train.txt`
- `gutenberg.train.txt`
- `open_subtitles.train.txt`
- `simple_wiki.train.txt`
- `switchboard.train.txt`

---

## 2. Cleaning / document segmentation

**[CONFIRMED — code]** `load_documents()` in `src/influence_curriculum/data.py:29-59`:

- Iterates the 6 `.txt` files (sorted by filename).
- Segments each file's raw text into "documents" per a `doc_boundary` rule (`_segment`,
  `data.py:17-26`): `"line"` (one doc per non-empty line), `"blank_line"` (split on
  blank-line-separated paragraphs), `"whole_file"` (one doc per file), or a custom
  per-file callable.
- Drops any segment that is empty after stripping whitespace.
- Drops any segment shorter than `min_doc_tokens` (word count if no tokenizer is
  passed, else token count).
- Assigns each surviving document an ID of the form `f"{path.stem}#{i}"` — e.g.
  `open_subtitles.train#92698`, `childes.train#401183` (`data.py:51`).
- Warns if fewer than 2000 documents load from a directory (a segmentation sanity
  check), but does not error.

**[CONFIRMED — reproduced from source]** The corpus was built with
`notebooks/04_colab_pipeline.ipynb` cell 4's exact `DataConfig`, **not** the class
defaults:

```python
data_cfg = DataConfig(
    doc_boundary={
        "childes.train.txt": segment_childes,
        "simple_wiki.train.txt": segment_simple_wiki,
    },
    default_boundary="line",
    min_doc_tokens=3,
)
```

where `segment_childes` strips `*SPEAKER:` tags and bracketed annotations line-by-line,
and `segment_simple_wiki` splits on `= = =` section headers into per-section documents
(both defined in the same cell). All other files fall back to `default_boundary="line"`
(one document per non-empty line).

This was verified by directly rebuilding the corpus from the real local
`BabyLM-2026-Strict-Small/*.train.txt` files using this exact configuration and
comparing against the real `tiered-4bands` data pulled from the Hub: the rebuild
produces **exactly 726,009 documents**, with per-source counts and **every single
`id → text` pair byte-identical** to the real dataset (726,009/726,009 exact matches,
zero mismatches). Two corroborating checks on the real data: no `childes` document
contains the raw `*SPEAKER:` tag pattern (confirming `segment_childes` fired), and no
`simple_wiki` document contains a literal `"= = ="` (confirming `segment_simple_wiki`
fired); the shortest document in the whole corpus is exactly 3 words, matching
`min_doc_tokens=3` (the class default is 1, which does not reproduce this floor). The
earlier hypothesis that this used `data.py`'s plain defaults is ruled out — it does not
reproduce D=726,009 or the observed per-document text at all.

**Result: D = 726,009 documents total**, confirmed by the influence matrix shape for
`flakoash/babylm-curriculum-run07` (`(726009, 2)`, read directly from the `.npy` file
header on the Hub), the final `tiered-4bands` dataset's total row count (726,009, from
the HF datasets-server API), and the from-scratch local rebuild above — four
independent measurements agree exactly. Average document length is ~12.0 words (median
6) — shorter than the ~13.8-word estimate a naive corpus-total/doc-count division gives,
because word count is not evenly distributed across documents.

---

## 3. Surrogate model training

**[CONFIRMED — code + Hub data]** A GPT-2-architecture surrogate model was trained on
the full 726,009-document corpus in **random order** and checkpointed once per epoch,
via `train_surrogate()` in `src/influence_curriculum/train.py:33-133` (called from
notebook 04 cell 6). Each "checkpoint" here is a full model-weights snapshot saved
after one complete pass over the (freshly reshuffled) corpus — used purely to extract
gradients in the next stage, not as a resumable training state.

Both checkpoints are stored as `flakoash/babylm-surrogate-run07/epoch_00/` and
`epoch_01/`. Architecture, hardcoded in notebook 04 cell 6 and matching
`epoch_00/config.json` exactly:

```python
GPT2Config(n_embd=384, n_layer=8, n_head=6, n_inner=384 * 4,
           vocab_size=16384, n_positions=1024, ...)
```

```json
{
  "architectures": ["GPT2LMHeadModel"],
  "model_type": "gpt2",
  "n_embd": 384, "n_head": 6, "n_inner": 1536, "n_layer": 8,
  "n_positions": 1024, "vocab_size": 16384,
  "activation_function": "gelu_new",
  "tie_word_embeddings": true
}
```

A custom tokenizer is bundled with the surrogate (`tokenizer_config.json`:
`TokenizersBackend`, `bos="<s>"`, `eos="</s>"`, `pad="</s>"`, `model_max_length=1024`).
Note this is a **different tokenizer** from the one used later to train the actual
run002 model (`BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small`, per run002's
`run_config.json`) — the surrogate's tokenizer only needs to support gradient scoring,
not the final trained model.

**[CONFIRMED — code]** Surrogate training hyperparameters, from notebook 04 cell 6's
`TrainingConfig` (with unset fields falling back to the dataclass defaults in
`train.py:19-30`):

| Param | Value |
|---|---|
| epochs (`SURROGATE_EPOCHS`) | 2 |
| per-device batch size | 64 |
| effective batch size | 256 |
| learning rate | 7e-4 *(class default — not overridden)* |
| lr scheduler | cosine |
| max sequence length | 128 |
| seed | 0 |
| fp16 | True (on CUDA) |

**How this was recovered:** although no `training_args.json`/model card exists on the
Hub for any surrogate repo, notebook 04 cell 2's config (`RUN_ID`, `SMOKE_TEST`,
`SMOKE_N_DOCS=300`) directly derives `HUB_MODEL_REPO=f"flakoash/babylm-surrogate-run
{RUN_ID}"` / `HUB_DATASET_REPO=f"flakoash/babylm-curriculum-run{RUN_ID}"` — the exact
naming scheme seen for run05/06/07. Re-running this same notebook with `RUN_ID` bumped
"05" → "06" → "07" (and eventually `SMOKE_TEST=False`, `SAMPLE_FRAC=1.0`) fully
explains that whole Hub family, including `run05`'s **exact** D=300, which matches
`SMOKE_N_DOCS=300` precisely.

The **2 checkpoints** (`T=2`) directly determine the shape of the influence matrix in
the next step.

---

## 4. Tokenization, then influence scoring

**[CONFIRMED — code]** Before scoring, notebook 04 cell 7 tokenizes all 726,009 texts:
`tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN=128, padding=False,
return_tensors=None)`, producing one `encodings` dict per document — this is the step
that bridges §2's `(texts, doc_ids)` to `compute_influence_matrix`'s `encodings`
argument. Document order is preserved end-to-end: `load_documents` → `texts`/`doc_ids`
→ `encodings` (same order) → `Phi` rows (same order, saved alongside `doc_ids.json`) →
band assignment (§6, indexes into this same order). This single shared ordering is what
lets every later stage look up a document by row index.

`compute_influence_matrix()` (`src/influence_curriculum/score.py:150-266`). For each
checkpoint `t` and document `i`:

1. **Pass 1** (per checkpoint): compute the mean input-embedding gradient over the
   *entire* corpus via mini-batch backward passes (fp16), then L2-normalize it
   (`normalize=True` default) → `ḡ_t`.
2. **Pass 2** (per checkpoint): compute each document's own input-embedding gradient
   and dot it with `ḡ_t`. On the production GPU/JVP path (`_jvp_score`/
   `_vmap_jvp_scores`, `score.py:93-129`, used whenever `torch.func` and CUDA are
   available), the per-document gradient is used **raw, not normalized** — only `ḡ_t`
   is a unit vector, so `Φ[i, t] = g_i · ĝ_t`. Only the CPU fallback path
   (`_fallback_grad`, `score.py:258`) additionally normalizes the per-document
   gradient before the dot product. Since the JVP path is what runs at production
   scale, the paper's "cosine similarity of two unit vectors" framing is only exact for
   the CPU fallback, not for the actual run.

Default `InfluenceConfig` (`score.py:32-42`): `grad_target="input_embeddings"`,
`grad_path="full"`, `normalize=True`, `pass1_batch_size=64`, `pass2_batch_size=8`.
Notebook 04 cell 2 overrides these to `PASS1_BATCH_SIZE=256`, `PASS2_BATCH_SIZE=64` (a
comment in `score.py` claiming "notebook sets 32" is stale and does not match any
notebook actually in this repo). `fp16=True` for pass-1 backward (pass-2 JVP always
runs in fp32 to avoid a LayerNorm/GELU dtype clash).

**[CONFIRMED — Hub data]** Output: `influence_matrix.npy` of shape `(726009, 2)` and a
matching `doc_ids.json` (726,009 entries), uploaded to
`flakoash/babylm-curriculum-run07`. This dataset repo also carries `curriculum/epoch_*`
JSONL files pairing each `doc_id` back to its raw text (used later to reconstruct text
without re-running segmentation — see §6).

**[INFERRED — timeline]** Hub creation timestamps place `babylm-surrogate-run07` at
2026-06-19 21:28:40 UTC and `babylm-curriculum-run07` at 2026-06-19 23:55:15 UTC — a
~2.5 hour gap consistent with scoring a 726k-document corpus. Two earlier, smaller
iterations exist on the same profile — `run05` (D=300, likely a tiny smoke-test) and
`run06` (D=363,003, roughly half-scale) — created earlier the same day, showing this
was an iterative scale-up (run05 → run06 → run07) before the final full-corpus run.
`run07` is the only one of the three whose document count (726,009) matches
`tiered-4bands`, which rules out `run05`/`run06` as its ancestor.

---

## 5. Aggregation and difficulty ranking

**[CONFIRMED — code]** In `notebooks/06_sequential_bands_curriculum.ipynb`, cell 5:

```python
agg   = Phi.mean(axis=1)                   # mean influence across all T checkpoints
order = np.argsort(agg)[::-1].tolist()     # descending: highest influence = easiest
```

Each document's per-checkpoint influence scores are averaged into one scalar
"difficulty" score. Documents are sorted **descending** by that score. Per the paper's
finding (and `docs/influence_curriculum_paper_reference.md:127`), **higher mean
influence corresponds to easier documents** — so `order[0]` is the single easiest
document in the corpus and `order[D-1]` is the single hardest.

This descending sort is the **only** sort ever applied to the full corpus. It happens
once, before band assignment.

---

## 6. Splitting into 4 bands (the actual "tiering" step)

**[CONFIRMED — code + Hub data]** `notebooks/06_sequential_bands_curriculum.ipynb`,
cell 5, `STRATEGY="sequential"` branch:

```python
boundaries = [round(D * k / N_SEGMENTS) for k in range(N_SEGMENTS + 1)]
bands = [order[boundaries[k] : boundaries[k + 1]] for k in range(N_SEGMENTS)]
```

With `D = 726,009` and `N_SEGMENTS = 4`, this produces four **contiguous, disjoint,
near-equal-size** slices of the descending-sorted order:

| Band | Doc count | Content |
|---|---|---|
| `epoch_00` | 181,502 | easiest quartile (highest mean influence) |
| `epoch_01` | 181,502 | second-easiest quartile |
| `epoch_02` | 181,503 | second-hardest quartile |
| `epoch_03` | 181,502 | hardest quartile (lowest mean influence) |

(Counts computed from the `boundaries` formula above with the confirmed `D` and
`N_SEGMENTS` — off by at most 1 document per band due to integer rounding — **and**
independently verified by directly downloading all 4 real `epoch_00..03.jsonl` files
from `flakoash/babylm-curriculum-tiered-4bands` and counting lines: exactly
181,502 / 181,502 / 181,503 / 181,502, matching the formula precisely. This line-count
match is stronger, more direct evidence for `STRATEGY="sequential"` than the
no-duplicates/non-monotonic-size argument below — it confirms the exact partition
formula, not just the general shape of the algorithm.)

Each document appears in **exactly one** band — no overlap, no duplication. Band
assignment is by **rank/position** in the sorted order (equal document counts per
band), **not** by any fixed score threshold or equal-size-in-bytes/words split — bands
of near-identical row count can (and do) differ substantially in total text volume,
because average document length isn't uniform across the difficulty spectrum (e.g.
`epoch_02`'s underlying `.jsonl` file is roughly double the size of `epoch_00`'s, at
equal document counts — consistent with harder/mid-difficulty text skewing longer on
average).

**How this was determined empirically (why "sequential", not the other two candidate
algorithms in this codebase):** Three different band-building implementations exist in
this repo (see §11.1 for the other two: `_build_cumulative` in
`src/influence_curriculum/curriculum.py`, and `STRATEGY="sliding_window"` in the same
notebook cell). They produce structurally distinguishable outputs:

- Cumulative/nested bands would produce strictly **increasing** file sizes
  (`epoch_03` ⊇ `epoch_02` ⊇ `epoch_01` ⊇ `epoch_00`) — ruled out because
  `tiered-4bands`'s actual band files are non-monotonic in size (`epoch_02` is larger
  than `epoch_03`).
- Sliding-window bands would produce **overlap** — each document appearing ~2× on
  average (confirmed on the sibling `sliding-window-4bands` dataset, which has exactly
  2× `tiered-4bands`'s total row count: 1,452,016 vs 726,009).
- Sequential (disjoint) bands produce **exactly D total rows, zero duplicates** — this
  matches `tiered-4bands` exactly (726,009 rows, no duplicate `id`s across its 4
  files), which is what the code's own sanity check (cell 7 of notebook 06) asserts for
  this strategy.

So the empirical structure of the actual Hub dataset — not any saved notebook output —
is what identifies `STRATEGY="sequential"` as the algorithm actually used. **[CONFIRMED
— author]** The project author has directly confirmed `sequential` is indeed the
algorithm used, and that `notebooks/06_sequential_bands_curriculum.ipynb` is the actual
script that was run — it was simply committed to git after the fact, which is why its
commit date (2026-06-26) postdates `tiered-4bands`'s creation on HF (2026-06-23). This
independently corroborates the empirical (line-count) evidence above.

**[CONFIRMED — Hub data + author]** `N_SEGMENTS=4, SEED=0` — the Hub commit message on
`babylm-curriculum-tiered-4bands` reads `"sequential-bands curriculum  N=4 seed=0"`.
Note: the leading text `"sequential-bands curriculum"` is a **hardcoded literal** in the
notebook's push cell (`commit_message=f"sequential-bands curriculum  N={N_SEGMENTS}
seed={SEED}"`, cell 9) — it does not change based on which `STRATEGY` actually ran
(the identical string also appears on the `sliding-window-4bands` repo's commit, which
is why it can't be trusted *on its own* to identify the algorithm — the line-count
match and author confirmation above are what actually establish that). The `N=4` and
`seed=0` **values** are live f-string substitutions of the real `N_SEGMENTS`/`SEED`
variables at push time, so those two specific numbers are trustworthy for this run.

---

## 7. Ordering *within* each band — the direct answer to "are docs sorted or re-sorted inside the groups?"

**[CONFIRMED — code]** No secondary sort happens within a band. Immediately after
slicing, each band is shuffled with a **seeded** Python `random.Random(SEED)`
(`notebooks/06_...ipynb`, cell 5):

```python
rng = random.Random(SEED)   # SEED = 0
...
shuffled = band[:]
rng.shuffle(shuffled)
```

So the pipeline's only ordering signal is the single corpus-wide descending sort by
influence score (§5), used purely to **assign** documents to bands. Once a document is
inside a band, its position is randomized (deterministically, via the fixed seed) —
there is no re-sorting by score, length, source, or any other criterion inside a band.
The written JSONL/TXT files reflect this shuffled order, not the original influence
rank.

---

## 8. Output artifacts and schema

**[CONFIRMED — Hub data]** `flakoash/babylm-curriculum-tiered-4bands` on Hugging Face
Hub (dataset repo, sha `82a1f92...`, pushed 2026-06-23):

- Single HF `default` config, single `train` split, **726,009 rows total**, columns
  `id` (string, e.g. `open_subtitles.train#92698`) and `text` (string).
- Underlying files, one JSONL + one TXT mirror per band (`curriculum/epoch_00.jsonl`
  through `epoch_03.jsonl`, plus matching `.txt`): each JSONL line is
  `{"id": ..., "text": ...}`; each TXT file has one document per line with a blank
  line between documents (same convention as notebook 04's format).
- No `influence_matrix.npy` or `doc_ids.json` on this repo — those artifacts stay on
  the source repo (`babylm-curriculum-run07`); `tiered-4bands` only re-uploads the
  final banded text.
- **No dataset card** exists for this repo (or for run05/run06/run07, or the surrogate
  model repos) — none of the generation parameters above are documented on the Hub
  itself; everything had to be reconstructed from this repo's code plus the empirical
  shape of the produced data.

---