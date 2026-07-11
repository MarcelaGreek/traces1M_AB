# AB_11 — Transcript Evaluation and Semantic Retrieval

AB_11 contains both workflows:

- **Option A:** map-reduce transcript evaluation without embeddings.
- **Option B:** optional chunk embeddings, Supabase storage, and semantic retrieval.

## Important AB_11 corrections

1. The 26 traces keep the same scenarios, but each trace has distinct neutral filler.
2. Each trace is divided into exactly four balanced **core** partitions.
3. A configurable overlap supplies boundary context without dropping any core text.
4. Chunk records expose core and context boundaries for inspection.
5. Option A writes both:
   - `outputs_A/chunk_evaluations.jsonl`
   - `outputs_A/chunks_for_embeddings.jsonl`

   The second filename preserves Option B compatibility.
6. `traces.csv` is the stable input filename. `traces_AB_11.csv` is included as a versioned copy.

## Start

Follow `STEPS.md`.

## Folder guide

- `README_A.md` — Option A
- `README_B.md` — Option B
- `docs/PRODUCTION_CHUNKING.md` — production recommendation
- `sql/01_create_vector_store.sql` — Supabase schema
- `CHANGELOG.md` — changes from AB_10
- `VERSION.txt` — release identifier
