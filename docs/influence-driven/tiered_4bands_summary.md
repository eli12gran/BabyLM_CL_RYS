# Summary: Influence-Driven 4-Band Curriculum

This summarizes how the training corpus was reordered into a four-band,
easy-to-hard curriculum, and how a language model was trained on it. The method
follows Schoenegger, Thoma, Blevins & Roth (2025), *"Influence-driven Curriculum
Learning for Pre-training on Limited Data"*, which adapts TracInCP (Pruthi et al.,
2020) to rank training documents by their estimated influence on a language model,
rather than by human-judged difficulty.

## 1. Corpus

The base corpus is the BabyLM 2026 Strict-Small track, approximately 10 million
words drawn from six sources: British National Corpus transcripts (spoken
register), CHILDES (child-directed speech transcripts), Project Gutenberg (books),
OpenSubtitles (film/TV dialogue), Simple Wikipedia, and Switchboard (telephone
conversation transcripts). The corpus had already been filtered for toxic content
by its original creators.

## 2. Document segmentation

Each source was split into individual documents. For most sources, a document is
one non-empty line of text (e.g., one spoken utterance or subtitle line). Two
sources used custom segmentation: CHILDES transcripts had speaker-identifier tags
and bracketed annotations stripped from each line; Simple Wikipedia articles were
split into one document per section, at section-header boundaries. Documents
shorter than three words after segmentation were discarded. This produced a
working corpus of **726,009 documents**.

## 3. Surrogate model and influence estimation

To estimate each document's training influence without the cost of training the
full target model, a small surrogate language model was trained from scratch on
the complete 726,009-document corpus, presented in random order, for two epochs,
saving one checkpoint after each epoch (the surrogate used a GPT-2-style
architecture: 8 layers, 6 attention heads, 384-dimensional embeddings, ~16,000-token
vocabulary).

Using the two checkpoints, an influence score was computed for every document via
the TracInCP method: for each checkpoint, the average input-embedding gradient
across the entire corpus was computed (the "mean gradient" direction the model was
moving in as a whole), and each document's own input-embedding gradient was
projected onto that mean-gradient direction. This produces one influence value per
document per checkpoint; the two values were averaged into a single difficulty
score per document. Per the source paper's finding, **higher influence corresponds
to an easier document, and lower influence to a harder one**.

## 4. Curriculum construction — the four bands

All 726,009 documents were ranked in one ordered list, from highest influence
(easiest) to lowest influence (hardest). This single ranked list was then divided
into four contiguous, non-overlapping quarters of approximately equal size
(~181,500 documents each):

- **Band 1** — the easiest quarter of the corpus (highest influence)
- **Band 2** — the second-easiest quarter
- **Band 3** — the second-hardest quarter
- **Band 4** — the hardest quarter of the corpus (lowest influence)

Every document belongs to exactly one band; none is duplicated or omitted. Band
membership is determined purely by a document's rank in the influence ordering —
bands are equal in document *count*, not in total word count, since document
length is not uniform across the difficulty spectrum (the middle bands, in
practice, contained noticeably longer documents on average than the easiest band).

Within each band, after assignment, document order was randomized (with a fixed
seed, for reproducibility); no further sorting by difficulty, length, or source was
applied within a band. The single sorting operation in the entire pipeline is the
initial corpus-wide ranking by influence score, used only to decide which band a
document belongs to — nothing is re-sorted afterward.
