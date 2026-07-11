# Changelog

## AB_11

- Preserved the complete Option A + Option B project structure from AB_10.
- Replaced repetitive neutral filler with distinct filler for each trace.
- Added exactly four balanced core partitions per transcript.
- Added configurable boundary overlap (`CHUNK_OVERLAP_WORDS`).
- Added explicit core/context word boundaries and integrity checks.
- Preserved `chunks_for_embeddings.jsonl` for Option B compatibility.
- Added `chunk_evaluations.jsonl` as the clearer Option A filename.
- Added robust `STEPS.md`, `.gitignore`, version file, tests, SQL, and
  production chunking guidance.
- Updated Option B metadata to AB_11.
