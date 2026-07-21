# Paper Reference Digest — Influence-driven Curriculum Learning

> **What this is:** a paraphrased technical companion to `influence_curriculum_spec.md`, summarizing Schoenegger, Thoma, Blevins & Roth (2025), *Influence-driven Curriculum Learning for Pre-training on Limited Data* (arXiv:2508.15475v2). It is written in our own words for implementation reference — not a verbatim copy of the paper. Equations and hyperparameter values are reproduced as facts; the prose is rewritten; the reference list is omitted; appendix details relevant to implementation are kept. For exhaustive per-model result tables, consult the original PDF.

---

## 1. Problem and approach

Curriculum learning (presenting training data in a non-random, usually easy→hard order) has historically *not* helped low-resource LM pretraining when difficulty is defined by human-centered heuristics (sentence length, lexical diversity, source difficulty). The paper's thesis: replace the human-centered difficulty signal with a **model-centric** one — **training-data influence** — and curriculum learning becomes competitive.

Pipeline:
1. Train a surrogate model on the data in **random order**, saving a checkpoint each epoch.
2. From those checkpoints, score each document by its **average influence** on the rest of the data.
3. Re-order the data by that score into a curriculum; train fresh models on it and compare to random order.

---

## 2. Influence estimation

They adapt **TracInCP** (Pruthi et al. 2020). Point-wise influence of training example `z` on the prediction of `z'`, summed over checkpoints `t ∈ T`:

```
φ_TracInCP(z, z') = Σ_{t∈T} η_t · ∇ℓ(w_t, z) · ∇ℓ(w_t, z')
```

Following Yeh et al. (2022), `w_t` is the model's **input embeddings** at checkpoint `t` (the gradient still chains back through all higher layers, so it reflects the whole model). `η_t` is the learning rate, dropped in practice.

To get a single per-document difficulty signal, they define the **average influence** a document `z` exerts over the whole dataset `D`:

```
φ_t(z, D) = (1/|D|) · Σ_{z'∈D} ∇ℓ(w_t, z) · ∇ℓ(w_t, z')
          = ∇ℓ(w_t, z) · E_{z'∼D}[ ∇ℓ(w_t, z') ]
```

The second form is the efficiency lever: average influence = the document's gradient dotted with the **dataset mean gradient**, i.e. O(|D|) not O(|D|²). Intuitively this is high for *prototypical* examples (gradient close to the average) and low for outliers — unlike perplexity/surprisal.

**Normalization:** the raw dot product is biased toward longer examples, so each gradient is unit-normalized before the dot product, yielding cosine similarity.

Computing this for all documents at all `T` checkpoints gives an influence matrix `Φ ∈ R^{|D|×T}`.

---

## 3. Curriculum designs

Two families, distinguished by **coverage strategy**:

- **Epoch-wise coverage** — every epoch is a full pass over the whole dataset (just re-ordered).
- **Cumulative coverage** — difficulty increases across epochs; early-epoch examples are not revisited later.

### Epoch-wise variants
- **C↘ / C↗** — each epoch, sort documents by descending / ascending influence, using the surrogate checkpoint saved after that epoch.
- **C̃↘ / C̃↗** — same, but split the sorted order into contiguous 1000-document segments and shuffle *within* each segment (adds local diversity).
- **(C∗h)̃↘ / (C∗h)̃↗** — before sorting, convolve `Φ` along the epoch axis with a causal **lognormal** filter `h`, then segment-shuffle as above. This upweights documents that stay influential across epochs:
  ```
  (C∗h)(t,i) = Σ_{k=0}^{T} Φ(t−k, i) · h(k)
  ```
- **C^{50}** — discard the 50% least-influential documents each epoch (keeping total words constant), shuffle once per epoch. A data-cleaning style variant.

### Cumulative variants
- **C^E↘ / C^E↗** — aggregate influence across all epochs `φ_T(z,D) = Σ_t φ_t(z,D)`, sort ascending/descending, split into `m=10` segments, then sample to form `m` equal-length epochs of increasing/decreasing difficulty.
- **C_A** — sort by the aggregate `φ_T`, form `m=10` segments, then build the curriculum by **alternating** highest- and lowest-influence segments; train 10 epochs, reshuffling within each segment each pass. A compromise between epoch-wise and cumulative.

### Baselines
- **C_rand** — random order, 10 full passes. Doubles as the surrogate (source of checkpoints) and the comparison baseline. `T = 10` checkpoints.
- **C_source** — handcrafted source-difficulty: datasets presented as ordered blocks assigned to 5 difficulty stages (C1–C5), 2 epochs per stage.
- **C_MATTR** — sort by increasing moving-average type–token ratio (window 5).
- **C_PPL** — sort by increasing perplexity under a static unigram model.

---

## 4. Datasets

- **D2024** — the BabyLM 10M-word text-only dataset (2024/2025 challenge), a mix of sources of varying difficulty.
- **D_stratified** — same sources rebalanced to equal words per stage.
- **D_equitoken** — stratified *and* length-controlled: synthetic documents of exactly 100 words (by concatenation), to remove document-length as a confound.
- **Eval set** — sampled from the 100M-word BabyLM dataset, `|D_eval| = 0.05 · |D2024|`.

Stage taxonomy (used only by `C_source` / the source-difficulty curricula). Two schemes:
- D2024: C1 child-directed speech (CHILDES) · C2 unscripted dialogue (Switchboard, BNC dialogue) · C3 scripted dialogue (OpenSubtitles) · C4 wiki (Simple Wiki) · C5 written English (Gutenberg).
- D_stratified / D_equitoken: C1 child-directed · C2 children's books · C3 dialogue · C4 educational · C5 written English.

Note a key imbalance: in D2024, child-directed speech is ~54% of *documents* but only ~28% of *words*, while written English is ~25% of words in only ~6% of documents.

---

## 5. Models and training (factual, from Table 3)

Two architectures, both randomly initialized: **RoBERTa** (126M params, MLM) and **Llama** (97.2M params, CLM). 84 models total = 2 architectures × 3 datasets × 14 curricula. Trained on 4× H100, effective batch size 2048. Each curriculum shows at most 100M words (10 passes over 10M).

| Hyperparameter | RoBERTa | Llama |
|---|---|---|
| Vocabulary size | 52k | 52k |
| Hidden size | 768 | 768 |
| Layers | 12 | 12 |
| Attention heads | 12 | 12 |
| Initializer range | 0.02 | 0.02 |
| Tie word embeddings | True | True |
| Max position embeddings | 514 | 256 |
| Intermediate (FFN) size | 3072 | 2048 |
| Norm epsilon | 1e-5 | 1e-6 |
| Attention dropout | 0.1 | 0 |
| Activation | gelu | silu |
| Hidden dropout | 0.1 | — |
| FP16 | False | False |
| Per-device batch size | 32 | 32 |
| Grad accumulation steps | 16 | 16 |
| GPUs | 4 | 4 |
| Adam β1 / β2 / ε | 0.9 / 0.98 / 1e-6 | 0.9 / 0.98 / 1e-6 |
| Weight decay | 0.01 | 0.01 |
| Learning rate | 5e-4 | 7e-4 |
| LR scheduler | polynomial | cosine |

(Our spec is **CLM-only**, so the Llama column is the relevant one.)

---

## 6. Key findings (relevant to implementation and expectations)

- **Curriculum learning helps with this signal.** Best gains over random order: up to **+12.42 pp** for RoBERTa (`C^{50}`, D2024) and **+4.62 pp** for Llama (`C↗`, D_stratified). Headline: >10 pp for RoBERTa and >4 pp for Llama on D2024.
- **Best raw models:** RoBERTa `(C∗h)̃↗` on D_stratified = 0.592 macro-acc (+7.96 pp); Llama `(C∗h)̃↘` on D2024 = 0.584 (+4.34 pp).
- **RoBERTa gains exceed Llama's**, partly because RoBERTa's random-order baseline is lower.
- **Coverage strategy matters most.** Epoch-wise (full pass per epoch) beats cumulative coverage. The source-difficulty curricula (handcrafted `C_source` and synthetic `C^E`) are the weak performers; influence-based sorting could not rescue that scheduling.
- **Sort direction is a wash.** Ascending vs descending influence does not consistently beat the other; the within-segment shuffle and the lognormal re-weighting also do not reliably change results. The authors' interpretation: the benefit comes from **grouping similar-influence documents into the same batch**, not from the global order. (This is why our spec defaults to the simpler `C~` and treats the lognormal filter as optional.)
- **Source composition stays stable.** Influence curricula keep a source distribution close to random order's (it doesn't drift across epochs), so performance is not explained by source mix alone.
- **Ranking is biased toward document-dense sources.** Because the score is similarity to the dataset *mean* gradient, sources with many documents (e.g. child-directed speech) dominate. `C^{50}` ends up >90% child-directed speech, which explains its poor cross-dataset behavior despite a strong single D2024 RoBERTa number. (Implication for our `doc_boundary` choice: segmentation sets per-source document counts and therefore skews influence.)
- **Training-loss spikes are not predictive.** Influence curricula produce severe training-loss spikes, but the loss-ratio instability metric shows no significant negative correlation with downstream benchmark gain — i.e. spikes don't reliably indicate worse models here.
- **Influence is inversely related to heuristic difficulty.** Decreasing-influence curricula correlate more with MATTR/PPL orderings than increasing-influence ones, suggesting higher influence ≈ lower difficulty.
- **Benchmarks used:** BLiMP, BLiMP-supplement, EWOK, (Super)GLUE, an entity-tracking task, and an adjective-nominalization task — the BabyLM evaluation pipeline. (This is the eval harness our §12.3 depends on.)

---

## 7. Appendix details relevant to implementation

- **MLM masking reproducibility (RoBERTa only).** For dynamic-masking models, influence depends on which tokens are masked, so they made masking deterministic via a hash of `(document, epoch)` in a custom data collator. **Not needed for our CLM-only scope.**
- **Runtime (for expectation-setting).** All 84 models took 195 H100-hours (~2h20m each). Influence estimation is run once per dataset and is the heavy step: roughly 44.3 h sequential or under 5 h parallelized across checkpoints; per-dataset sequential figures ranged from ~7.75 h (D_equitoken) to ~149.5 h (D_stratified, on slower V100s), ~266 GPU-hours total. Confirms that **influence estimation, not training, dominates cost**.
- **Loss-ratio instability metric (Appendix B).** `lr(s) = ℓ(s) / min_{s'<s} ℓ(s')` — current step loss over the best prior loss. No significant negative Spearman correlation with benchmark gain (D2024 0.177; D_equitoken 0.096; D_stratified 0.197). They hypothesize that sorting reduces in-batch diversity, causing the loss spikes.
- **Released artifacts (could not be retrieved at spec time).** Code on Zenodo (DOI 10.5281/zenodo.16919045); datasets and models on the HF collection `loris3/ticl-...`. Datasets released under CC BY 4.0.
