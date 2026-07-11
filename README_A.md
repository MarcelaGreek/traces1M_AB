# README_A — Option A Transcript Evaluation

Option A evaluates long transcripts using a map-reduce pattern and does **not**
require embeddings.

## Flow

```text
one transcript row
      ↓
four balanced core partitions
      ↓
small overlap added at boundaries
      ↓
one structured JSON evaluation per chunk
      ↓
all four chunk JSON evaluations grouped by run_id
      ↓
one final transcript-level evaluation
```

## Run

```powershell
python evaluate_option_a.py
```

## Outputs

```text
outputs_A/final_labels.csv
outputs_A/final_labels.jsonl
outputs_A/chunk_evaluations.jsonl
outputs_A/chunks_for_embeddings.jsonl
```

There should be:

```text
26 final transcript records
104 chunk records
```

`chunk_evaluations.jsonl` is the descriptive Option A name.
`chunks_for_embeddings.jsonl` is an identical compatibility copy for Option B.

## Chunk integrity

Every word belongs to exactly one `core_text` partition. `chunk_text` includes
the core plus optional boundary overlap. This means:

- no core truncation;
- no missing core words;
- no accidental core duplication;
- deliberate overlap only for context.

Configure:

```env
ROW_CHUNKS_PER_ROW=4
CHUNK_OVERLAP_WORDS=40
```

## Production note

Equal word partitioning is suitable for this controlled demonstration, but real
agent traces should use event-aware token-budget chunking. See
`docs/PRODUCTION_CHUNKING.md`.
