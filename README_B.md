# README_B — Option B Semantic Retrieval

Option B is optional and runs after Option A.

It reads:

```text
outputs_A/final_labels.jsonl
outputs_A/chunks_for_embeddings.jsonl
```

The upsert script also falls back to `chunk_evaluations.jsonl` if the
compatibility file is absent.

## Supabase structure

### `trace_eval_summaries`

One row per transcript: normally 26 rows.

### `trace_eval_chunks`

One row per evaluated chunk: normally 104 rows.

Each chunk row stores:

```text
chunk_id
run_id
chunk_index
chunk_count
previous_chunk_id
next_chunk_id
chunk_text
chunk_summary
chunk_evaluation
embedding
metadata
```

AB_11 boundary fields are retained inside `metadata`.

## Run

1. Execute `sql/01_create_vector_store.sql` in Supabase SQL Editor.
2. Run:

```powershell
python b1_create_embeddings_upsert.py
python b1_retrieve_example.py
```

## Important distinction

Option B does not need to cut each Option A chunk again. It creates one vector
for each existing `chunk_text`, so the expected count is 104 embeddings for
26 traces split into four chunks.
