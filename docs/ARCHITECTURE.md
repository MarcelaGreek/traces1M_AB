# Architecture

## Option A

Map each of four chunks to a strict evaluation JSON, then reduce the four JSON
objects into one transcript-level result.

## Option B

Embed each already-created `chunk_text` once, store the vector and Option A
labels in Supabase, retrieve nearest chunks, and optionally expand to immediate
neighbors.

Expected toy counts:

```text
26 transcripts
104 chunks
104 vectors
26 transcript summaries
```
