from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "evaluate_option_a",
    ROOT / "evaluate_option_a.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

with open(ROOT / "traces.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 26, f"Expected 26 traces, found {len(rows)}"

total_chunks = 0
for row in rows:
    text = module.transcript_text_from_row(row)
    chunks = module.split_text_into_exact_parts(
        text,
        number_of_chunks=4,
        overlap_words=40,
    )
    assert len(chunks) == 4
    original_words = text.split()
    reconstructed = []
    for chunk in chunks:
        reconstructed.extend(chunk["core_text"].split())
    assert reconstructed == original_words
    total_chunks += len(chunks)

assert total_chunks == 104
print("26 traces validated")
print("104 chunks validated")
print("No missing or duplicated core words")
