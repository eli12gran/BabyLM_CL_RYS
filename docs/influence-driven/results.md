# BabyLM 2026 — Experiment Results

## Run 001 — Influence-Driven Curriculum (2-band)

**Repo:** `flakoash/babylm-gpt2-small-run001`  
**Model:** GPT-2 Small  
**Track:** Strict-Small (10M word corpus)  
**Strategy:** 2-phase curriculum — easy band (phase 0) → hard band (phase 1), extended to 100M total words  
**Completed:** 2026-06-29

### Scores by checkpoint

| Revision  | BLiMP  | BLiMP+Supp | Eye Tracking | SPR  |
|-----------|--------|------------|--------------|------|
| chck_1M   | 0.555  | 0.535      | 9.46         | 2.77 |
| chck_2M   | 0.566  | 0.553      | 8.80         | 2.05 |
| chck_3M   | 0.585  | 0.577      | 9.01         | 2.06 |
| chck_4M   | 0.610  | 0.607      | 9.71         | 2.03 |
| chck_5M   | 0.606  | 0.591      | 9.67         | 1.98 |
| chck_6M   | 0.605  | 0.596      | 9.97         | 2.05 |
| chck_7M   | 0.627  | 0.603      | 8.64         | 1.90 |
| chck_8M   | 0.617  | 0.609      | 8.23         | 1.55 |
| chck_9M   | 0.624  | 0.608      | 9.53         | 1.87 |
| chck_10M  | 0.634  | 0.619      | 8.96         | 1.41 |
| chck_20M  | 0.665  | 0.631      | 8.42         | 1.45 |
| chck_30M  | 0.667  | 0.624      | 8.37         | 1.52 |
| chck_40M  | 0.674  | 0.627      | 8.53         | 1.53 |
| chck_50M  | 0.675  | 0.628      | 8.50         | 1.46 |
| chck_60M  | 0.676  | 0.628      | 8.63         | 1.50 |
| chck_70M  | 0.676  | 0.626      | 8.58         | 1.45 |
| chck_80M  | 0.677  | 0.624      | 8.68         | 1.48 |
| chck_90M  | 0.680  | 0.626      | 8.64         | 1.46 |
| chck_100M | 0.678  | 0.623      | 8.59         | 1.45 |

**Final model (main branch, full zero-shot):**
- Eye Tracking Score: 8.60
- Self-Paced Reading Score: 1.45

Fine-tuning (GLUE) predictions submitted: boolq, mnli, mrpc, multirc, qqp, rte, wsc. Scores computed server-side.

### Observations

- **BLiMP**: clean monotonic improvement 0.555 → 0.678. Grammatical knowledge scales consistently with word count. Plateaus slightly after 70M.
- **Eye Tracking**: peaks early at ~10% (chck_6M: 9.97), drops and plateaus around 8.5–8.7% for 20M–100M. The sweet spot for psycholinguistic fit is the **4–9M checkpoint range**, not the final model.
- **SPR**: drops from 2.77 → ~1.45 and flattens similarly to Eye Tracking.
- The early-training psycholinguistic peak is a known phenomenon: early in training, surprisal correlates well with human reading difficulty, but as the model improves at language modeling its surprisal distribution shifts in ways that are less predictive of human RT.

### Score interpretation

| Metric | Range | Notes |
|--------|-------|-------|
| BLiMP / BLiMP+Supp | 0–1 (0.5 = chance) | >0.65 solid, >0.75 competitive |
| Eye Tracking | 0–100% residual variance explained | 8–15% typical for small LMs |
| SPR | 0–100% residual variance explained | Lower than ET due to noisier paradigm |

---

## Run 002 — Influence-Driven Curriculum (4-band tiered)

**Repo:** `flakoash/babylm-gpt2-small-run002`  
**Model:** GPT-2 Small  
**Track:** Strict-Small (10M word corpus)  
**Curriculum:** `flakoash/babylm-curriculum-tiered-4bands` — 4-band tiered curriculum, extended to 100M total words  
**Completed:** 2026-06-29

### Scores by checkpoint

| Revision  | BLiMP  | BLiMP+Supp | Eye Tracking | SPR  |
|-----------|--------|------------|--------------|------|
| chck_1M   | 0.513  | 0.514      | 7.43         | 2.34 |
| chck_2M   | 0.550  | 0.535      | 7.47         | 1.51 |
| chck_3M   | 0.563  | 0.547      | 7.76         | 1.60 |
| chck_4M   | 0.591  | 0.558      | 5.79         | 1.35 |
| chck_5M   | 0.594  | 0.553      | 8.22         | 1.72 |
| chck_6M   | 0.598  | 0.587      | 6.92         | 1.98 |
| chck_7M   | 0.613  | 0.585      | 7.54         | 2.02 |
| chck_8M   | 0.613  | 0.587      | 6.19         | 2.03 |
| chck_9M   | 0.626  | 0.593      | 6.25         | 1.98 |
| chck_10M  | 0.618  | 0.583      | 6.15         | 1.89 |
| chck_20M  | 0.639  | 0.583      | 5.01         | 1.70 |
| chck_30M  | 0.638  | 0.577      | 5.90         | 1.58 |
| chck_40M  | 0.667  | 0.632      | 3.99         | 0.96 |
| chck_50M  | 0.660  | 0.626      | 4.39         | 1.09 |
| chck_60M  | 0.652  | 0.602      | 3.90         | 1.14 |
| chck_70M  | 0.644  | 0.592      | 3.87         | 1.22 |
| chck_80M  | 0.648  | 0.594      | 3.72         | 1.18 |
| chck_90M  | 0.647  | 0.610      | 3.47         | 1.08 |
| chck_100M | 0.646  | 0.607      | 3.53         | 1.09 |

**Final model (main branch, full zero-shot):**
- Eye Tracking Score: 3.53
- Self-Paced Reading Score: 1.09

Fine-tuning (GLUE) predictions submitted: boolq, mnli, mrpc, multirc, qqp, rte, wsc. Scores computed server-side.

### Observations

- **BLiMP**: starts lower than run001 (0.513 vs 0.555 at 1M) and ends lower (0.646 vs 0.678 at 100M). More bands did not help grammatical learning.
- **Eye Tracking**: much weaker than run001. Peaks at 8.22 (chck_5M) but collapses after 40M to the 3–4% range. The 4-band curriculum appears to hurt psycholinguistic fit significantly in the second half of training.
- **SPR**: similarly weaker; drops to ~1.0 after 40M vs run001's stable ~1.45.
- The sharp drop after chck_40M in both ET and SPR coincides with the model entering the harder bands — the harder data may shift surprisal distributions in ways that are increasingly misaligned with human reading times.

---

## Run vs Run Comparison (final model)

| Run | Curriculum | BLiMP@100M | BLiMP+Supp@100M | ET@100M | SPR@100M |
|-----|-----------|-----------|-----------------|---------|----------|
| run001 | 2-band | **0.678** | **0.623** | **8.59** | **1.45** |
| run002 | 4-band tiered | 0.646 | 0.607 | 3.53 | 1.09 |

**Takeaway:** 2-band curriculum (run001) wins on all metrics. More granular banding hurts psycholinguistic alignment without improving BLiMP.

---

## Run 003 — (next experiment)

<!-- Fill in after run003 completes -->
