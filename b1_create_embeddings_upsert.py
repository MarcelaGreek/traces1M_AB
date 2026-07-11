"""
AB_11 - Option B step 1: create embeddings and upsert to Supabase.

This script assumes Option A already ran and created:

    outputs_A/final_labels.jsonl
    outputs_A/chunks_for_embeddings.jsonl

What this script does
---------------------
1. Read final transcript summaries from Option A.
2. Read chunk records from Option A.
3. Create one OpenAI embedding for each chunk_text.
4. Upsert transcript summaries into Supabase table trace_eval_summaries.
5. Upsert chunk rows + embeddings into Supabase table trace_eval_chunks.

Important idea
--------------
The vector store is NOT only embeddings.
Each row stores:
- the text being embedded
- the embedding vector
- the labels from Option A
- model / system prompt information
- links to previous and next chunks

That is why a retrieval result can later show enough information to understand
what the agent did.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client


# -----------------------------------------------------------------------------
# 1. LOAD SETTINGS
# -----------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
OUTPUT_A_DIR = Path(os.getenv("OUTPUT_A_DIR", "outputs_A"))
UPSERT_BATCH_SIZE = 50

SUMMARY_TABLE = "trace_eval_summaries"
CHUNK_TABLE = "trace_eval_chunks"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file.

    JSONL means one JSON object per line.
    Option A uses JSONL because it is easy to stream and inspect.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def yes_no_to_bool(value: Any) -> bool | None:
    """
    Convert evaluator strings into database booleans.

    Option A writes yes/no because those values are easy to read in CSV.
    Supabase stores booleans because that is easier to query.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    value = str(value).strip().lower()
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def make_embedding(client: OpenAI, text: str) -> List[float]:
    """
    Create an embedding vector for one chunk of text.

    Input
    -----
    text: chunk_text from Option A.

    Output
    ------
    A list of floats, usually 3072 numbers for text-embedding-3-large.

    Why
    ---
    The embedding allows semantic search. For example, a search for
    "fictional creature" may retrieve chunks mentioning "Kaitus" even if the
    exact words differ.
    """
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text or "")
    return response.data[0].embedding


def build_summary_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one Option A final transcript result into one Supabase summary row.

    This table is the transcript-level store.
    One row = one original transcript/run_id.
    """
    return {
        "run_id": summary.get("run_id"),
        "model": summary.get("model"),
        "system_prompt_version": summary.get("system_prompt_version"),
        "user_input": summary.get("user_input"),
        "final_response": summary.get("final_response"),
        "user_request_intention": summary.get("user_request_intention"),
        "animal_harm_level": summary.get("animal_harm_level"),
        "eco_issue": yes_no_to_bool(summary.get("eco_issue")),
        "model_refused": yes_no_to_bool(summary.get("model_refused")),
        "safety_evaluation": summary.get("safety_evaluation"),
        "reason": summary.get("reason"),
        "final_summary": summary.get("final_summary"),
        "metadata": summary.get("metadata", {}),
    }


def build_chunk_record(client: OpenAI, chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one Option A chunk into one Supabase vector row.

    This table is the vector store.
    One row = one evaluated chunk.

    Key columns:
    - chunk_text: the text that is embedded
    - embedding: semantic vector for chunk_text
    - chunk_evaluation: JSON labels from Option A
    - chunk_summary: short evidence sentence from Option A
    - previous_chunk_id / next_chunk_id: used for neighborhood retrieval
    """
    chunk_text = chunk.get("chunk_text", "")
    embedding = make_embedding(client, chunk_text)

    return {
        "chunk_id": chunk.get("chunk_id"),
        "run_id": chunk.get("run_id"),
        "chunk_index": chunk.get("chunk_index"),
        "chunk_count": chunk.get("chunk_count"),
        "previous_chunk_id": chunk.get("previous_chunk_id"),
        "next_chunk_id": chunk.get("next_chunk_id"),
        "chunk_text": chunk_text,
        "chunk_summary": chunk.get("chunk_summary") or chunk.get("chunk_evaluation", {}).get("evidence", ""),
        "chunk_evaluation": chunk.get("chunk_evaluation", {}),
        "embedding": embedding,
        "metadata": {
            **chunk.get("metadata", {}),
            "embedding_model": EMBEDDING_MODEL,
            "option_b_version": "AB_11",
            "core_word_start": chunk.get("core_word_start"),
            "core_word_end_exclusive": chunk.get("core_word_end_exclusive"),
            "context_word_start": chunk.get("context_word_start"),
            "context_word_end_exclusive": chunk.get("context_word_end_exclusive"),
            "core_word_count": chunk.get("core_word_count"),
            "context_word_count": chunk.get("context_word_count"),
            "overlap_words_requested": chunk.get("overlap_words_requested"),
        },
    }


def upsert_in_batches(supabase: Client, table_name: str, records: List[Dict[str, Any]], batch_size: int) -> None:
    """
    Upsert records into Supabase in small batches.

    Upsert means:
    - insert if the row is new
    - update if the primary key already exists

    This makes it safe to rerun the script after an error or quota issue.
    """
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        supabase.table(table_name).upsert(batch).execute()


def main() -> None:
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("paste_"):
        raise RuntimeError("Set OPENAI_API_KEY in .env")
    if not SUPABASE_URL or "/rest/v1" in SUPABASE_URL:
        raise RuntimeError("Set SUPABASE_URL to the base URL only, e.g. https://xxxx.supabase.co")
    if not SUPABASE_ANON_KEY or SUPABASE_ANON_KEY.startswith("paste_"):
        raise RuntimeError("Set SUPABASE_ANON_KEY in .env")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    summary_path = OUTPUT_A_DIR / "final_labels.jsonl"
    chunk_path = OUTPUT_A_DIR / "chunks_for_embeddings.jsonl"
    if not chunk_path.exists():
        chunk_path = OUTPUT_A_DIR / "chunk_evaluations.jsonl"

    summaries = read_jsonl(summary_path)
    chunks = read_jsonl(chunk_path)

    print(f"Loaded {len(summaries)} transcript summaries")
    print(f"Loaded {len(chunks)} chunk records")

    summary_records = [build_summary_record(s) for s in summaries]
    print(f"Upserting {len(summary_records)} transcript summaries...")
    upsert_in_batches(supabase, SUMMARY_TABLE, summary_records, UPSERT_BATCH_SIZE)

    chunk_records: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {idx}/{len(chunks)}: {chunk.get('chunk_id')}")
        chunk_records.append(build_chunk_record(openai_client, chunk))

    print(f"Upserting {len(chunk_records)} embedded chunks...")
    upsert_in_batches(supabase, CHUNK_TABLE, chunk_records, UPSERT_BATCH_SIZE)

    print("\nDone. Supabase now contains transcript summaries and embedded chunks.")


if __name__ == "__main__":
    main()
