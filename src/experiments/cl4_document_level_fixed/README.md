# CL4 document-level influence curriculum

This package rewrites the CL4 pipeline so the influence matrix is document-level, matching the document/example unit used in the influence-driven curriculum method.

## Files

- `cl4_common.py`: shared helpers for corpus loading, filtering, tokenization, packing, model creation, checkpoint callbacks, and training arguments.
- `training_CL4_random_document_level.py`: trains the random-order surrogate. The surrogate is still trained on packed causal-LM blocks, but the corpus definition is document-level and shared with the influence script.
- `influence_matrix_document_level.py`: computes `Phi` with shape `[n_documents, n_checkpoints]`; each row is one raw document after filtering, not a packed sequence.
- `training_CL4_descending_document_level.py`: sorts raw documents by mean influence, locally shuffles within bins, then packs the ordered documents into LM blocks and trains the final model sequentially.

## Run order

### 1. Train the random surrogate

```bash
python training_CL4_random_document_level.py \
  --dataset_name BabyLM-community/BabyLM-2026-Strict-Small \
  --output_dir ./model_surrogate \
  --babylm_checkpoint_dir ./babylm_checkpoints_surrogate \
  --surrogate_epoch_checkpoint_dir ./surrogate_epoch_checkpoints \
  --min_document_words 3 \
  --max_seq_length 128
```

Push the epoch checkpoints to HF branches as you already do, e.g. `epoch_01` ... `epoch_10`.

### 2. Compute the document-level influence matrix

```bash
python influence_matrix_document_level.py \
  --surrogate_repo_id eligran12/babylm_CL4_surrogate \
  --surrogate_branches epoch_01 epoch_02 epoch_03 epoch_04 epoch_05 epoch_06 epoch_07 epoch_08 epoch_09 epoch_10 \
  --output_dir ./influence_output_document_level \
  --min_document_words 3 \
  --max_seq_length 128 \
  --normalize_mean_gradient
```

Outputs:

```text
./influence_output_document_level/influence_matrix.npy
./influence_output_document_level/influence_metadata.json
./influence_output_document_level/document_ids.npy
./influence_output_document_level/document_word_counts.npy
```

### 3. Train the final descending curriculum model

```bash
python training_CL4_descending_document_level.py \
  --influence_matrix ./influence_output_document_level/influence_matrix.npy \
  --influence_metadata ./influence_output_document_level/influence_metadata.json \
  --output_dir ./model_ticl_descending_document_level \
  --babylm_checkpoint_dir ./babylm_checkpoints_ticl_descending_document_level \
  --min_document_words 3 \
  --max_seq_length 128 \
  --bin_size 1000
```

## Important methodological change

Old pipeline:

```text
raw documents -> shuffle -> pack into 128-token blocks -> score blocks -> sort blocks
```

New pipeline:

```text
raw documents -> score documents -> sort documents -> pack sorted documents into 128-token blocks
```

That is the relevant change for matching the authors' document/example-level curriculum construction.
