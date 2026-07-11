"""
AB_11 - Option B step 2: semantic retrieval example.

This script reads questions from semantic_questions.txt and searches Supabase.

It writes both:
- outputs_B/retrieval_results.csv  -> easy to open in Excel
- outputs_B/retrieval_answer.txt   -> readable text answer
- outputs_B/retrieval_results.jsonl -> machine-readable details

Why CSV matters
---------------
The CSV output includes the columns needed to understand what happened:
- question
- user_input
- final_response
- model
- system_prompt_version
- safety_evaluation
- reason
- final_summary
- chunk_text
- chunk_summary

So someone can inspect the retrieval result without opening JSON.
"""

from __future__ import annotations

import csv
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
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-5.5")
QUESTIONS_FILE = Path(os.getenv("QUESTIONS_FILE", "semantic_questions.txt"))
OUTPUT_B_DIR = Path(os.getenv("OUTPUT_B_DIR", "outputs_B"))
TOP_K = int(os.getenv("TOP_K", "5"))


def read_questions(path: Path) -> List[str]:
    """
    Read semantic retrieval questions from semantic_questions.txt.

    Rules:
    - blank lines are ignored
    - lines starting with # are ignored
    - every other line is treated as one question

    Why use a separate file?
    ------------------------
    .env is private and usually not committed.
    semantic_questions.txt can be committed, so other people can see and reuse
    the test questions.
    """
    questions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)
    return questions


def make_embedding(client: OpenAI, text: str) -> List[float]:
    """
    Create an embedding for the search question.

    This embedding is compared against the chunk embeddings stored in Supabase.
    """
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def vector_search(supabase: Client, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    """
    Ask Supabase for the nearest chunks.

    This calls the SQL function match_trace_eval_chunks.
    The result is a list of chunks ranked by semantic similarity.
    """
    result = supabase.rpc(
        "match_trace_eval_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
        },
    ).execute()
    return result.data or []


def fetch_chunk_by_id(supabase: Client, chunk_id: str | None) -> Dict[str, Any] | None:
    """
    Fetch one chunk row by chunk_id.

    Used to get the immediate previous and next chunks.
    This demonstrates neighborhood retrieval:

        previous chunk + matched chunk + next chunk

    not the entire transcript.
    """
    if not chunk_id:
        return None
    result = supabase.table("trace_eval_chunks").select("*").eq("chunk_id", chunk_id).execute()
    if result.data:
        return result.data[0]
    return None


def fetch_summary_by_run_id(supabase: Client, run_id: str) -> Dict[str, Any] | None:
    """
    Fetch the transcript-level final evaluation for a retrieved chunk.

    This is how the B result can show:
    - user_input
    - final_response
    - model
    - system_prompt_version
    - final safety_evaluation
    - reason
    """
    result = supabase.table("trace_eval_summaries").select("*").eq("run_id", run_id).execute()
    if result.data:
        return result.data[0]
    return None


def expand_with_neighbors(supabase: Client, match: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return previous, matched, and next chunks.

    We do this because a semantic match may land on a middle chunk.
    The immediate neighbor chunks often provide enough context without loading
    the whole transcript.
    """
    rows: List[Dict[str, Any]] = []

    previous_row = fetch_chunk_by_id(supabase, match.get("previous_chunk_id"))
    if previous_row:
        previous_row["neighborhood_role"] = "previous"
        previous_row["similarity"] = match.get("similarity")
        rows.append(previous_row)

    matched_row = dict(match)
    matched_row["neighborhood_role"] = "matched"
    rows.append(matched_row)

    next_row = fetch_chunk_by_id(supabase, match.get("next_chunk_id"))
    if next_row:
        next_row["neighborhood_role"] = "next"
        next_row["similarity"] = match.get("similarity")
        rows.append(next_row)

    return rows


def flatten_result(question: str, rank: int, row: Dict[str, Any], summary: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Flatten Supabase JSON into one CSV row.

    This is the key function for readability.
    It combines:
    - retrieval information
    - chunk text
    - chunk evaluation
    - transcript summary
    - final safety evaluation
    """
    summary = summary or {}
    return {
        "question": question,
        "rank": rank,
        "neighborhood_role": row.get("neighborhood_role"),
        "similarity": row.get("similarity"),
        "run_id": row.get("run_id"),
        "chunk_id": row.get("chunk_id"),
        "chunk_index": row.get("chunk_index"),
        "chunk_count": row.get("chunk_count"),
        "model": summary.get("model"),
        "system_prompt_version": summary.get("system_prompt_version"),
        "user_input": summary.get("user_input"),
        "final_response": summary.get("final_response"),
        "user_request_intention": summary.get("user_request_intention"),
        "animal_harm_level": summary.get("animal_harm_level"),
        "eco_issue": summary.get("eco_issue"),
        "model_refused": summary.get("model_refused"),
        "safety_evaluation": summary.get("safety_evaluation"),
        "reason": summary.get("reason"),
        "final_summary": summary.get("final_summary"),
        "chunk_summary": row.get("chunk_summary"),
        "chunk_text": row.get("chunk_text"),
    }


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write retrieval results to CSV for Excel inspection."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write retrieval results to JSONL for debugging or later automation."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ask_answer_model(client: OpenAI, question: str, context_rows: List[Dict[str, Any]]) -> str:
    """
    Ask GPT to produce a readable answer from retrieved context only.

    This is optional but helpful for the demo.
    The CSV remains the main inspection artifact.
    """
    compact_context = []
    for r in context_rows:
        compact_context.append(
            {
                "run_id": r.get("run_id"),
                "neighborhood_role": r.get("neighborhood_role"),
                "user_input": r.get("user_input"),
                "final_response": r.get("final_response"),
                "safety_evaluation": r.get("safety_evaluation"),
                "reason": r.get("reason"),
                "chunk_summary": r.get("chunk_summary"),
                "chunk_text_snippet": (r.get("chunk_text") or "")[:900],
            }
        )

    prompt = f"""
Answer the question using ONLY the retrieved context below.

Question:
{question}

Retrieved context:
{json.dumps(compact_context, ensure_ascii=False, indent=2)}

Write a concise answer that mentions run_id, user_input, model response summary, and safety_evaluation.
"""
    response = client.responses.create(
        model=ANSWER_MODEL,
        input=prompt,
        max_output_tokens=1500,
    )
    return response.output_text


def main() -> None:
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("paste_"):
        raise RuntimeError("Set OPENAI_API_KEY in .env")
    if not SUPABASE_URL or "/rest/v1" in SUPABASE_URL:
        raise RuntimeError("Set SUPABASE_URL to the base URL only, e.g. https://xxxx.supabase.co")
    if not SUPABASE_ANON_KEY or SUPABASE_ANON_KEY.startswith("paste_"):
        raise RuntimeError("Set SUPABASE_ANON_KEY in .env")

    OUTPUT_B_DIR.mkdir(exist_ok=True)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    questions = read_questions(QUESTIONS_FILE)

    all_csv_rows: List[Dict[str, Any]] = []
    answer_sections: List[str] = []

    for question in questions:
        print(f"\nQuestion: {question}")
        query_embedding = make_embedding(openai_client, question)
        matches = vector_search(supabase, query_embedding, TOP_K)

        question_rows: List[Dict[str, Any]] = []

        for rank, match in enumerate(matches, start=1):
            neighbor_rows = expand_with_neighbors(supabase, match)
            for row in neighbor_rows:
                summary = fetch_summary_by_run_id(supabase, row.get("run_id"))
                flat = flatten_result(question, rank, row, summary)
                all_csv_rows.append(flat)
                question_rows.append(flat)

        answer = ask_answer_model(openai_client, question, question_rows)
        answer_sections.append(f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n" + "=" * 80)

        print(answer)

    csv_path = OUTPUT_B_DIR / "retrieval_results.csv"
    jsonl_path = OUTPUT_B_DIR / "retrieval_results.jsonl"
    txt_path = OUTPUT_B_DIR / "retrieval_answer.txt"

    write_csv(csv_path, all_csv_rows)
    write_jsonl(jsonl_path, all_csv_rows)
    txt_path.write_text("\n\n".join(answer_sections), encoding="utf-8")

    print("\nDone. Retrieval outputs:")
    print(f"  {csv_path}")
    print(f"  {jsonl_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
