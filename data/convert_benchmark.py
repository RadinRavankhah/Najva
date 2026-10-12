#!/usr/bin/env python3
"""Convert Jev-Persian-Benchmark data/scenarios.jsonl + gold.jsonl to benchmark.jsonl."""
import json
from pathlib import Path

SCENARIOS = Path("E:/Github/Personal/Temp/Jev-Persian-Benchmark/data/scenarios.jsonl")
GOLD = Path("E:/Github/Personal/Temp/Jev-Persian-Benchmark/data/gold.jsonl")
OUT = Path("data/benchmark.jsonl")

# Build gold lookup by question id
answers = {}
with open(GOLD, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            answers[row["id"]] = row.get("expected")

def map_answer(qid, kind, criteria, expected):
    if kind == "choice":
        # expected is option letter like A/B/C/D -> need to match criteria key
        # The criteria in scenarios uses A/B/C/D as keys directly
        return expected if isinstance(expected, str) else list(criteria.keys())[0]
    elif kind == "score":
        # expected is integer index (0/1/2)
        return int(expected) if isinstance(expected, int) else 0
    else:  # noul
        return bool(expected) if isinstance(expected, bool) else (expected == "true" or expected == 1 or expected == "1")

with open(SCENARIOS, "r", encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue
        scenario = json.loads(line)
        state = scenario.get("state", {}).get("text", "")
        cat = scenario.get("category", "unknown")
        sid = scenario.get("id", "unknown")
        for q in scenario.get("questions", []):
            qid = q.get("id")
            kind = q.get("type")
            criteria = q.get("criteria", {})
            instructions = q.get("instructions", "")
            expected = answers.get(qid)
            # Handle opposite mapping if criteria keys don't match expected format
            # For choice: criteria is dict with letter keys; expected is letter
            # For score: criteria is list; expected is int index
            # For noul: criteria is {"false":..., "true":...}; expected bool/1/0
            if kind == "choice" and isinstance(criteria, dict):
                # If expected is not directly a key, try to find via English mapping
                # But gold uses letter keys directly matching criteria for Persian
                answer = expected
            elif kind == "score":
                answer = int(expected) if expected is not None else 0
            elif kind == "noul":
                # Gold expected can be "true"/"false" or 0/1 — normalize to bool
                if isinstance(expected, bool):
                    answer = expected
                elif isinstance(expected, int):
                    answer = bool(expected)
                elif isinstance(expected, str):
                    answer = expected.lower() == "true" or expected == "1"
                else:
                    answer = False
            else:
                answer = expected
            
            row = {
                "state": state,
                "question_id": qid,
                "category": cat,
                "type": kind,
                "instructions": instructions,
                "criteria": criteria,
                "answer": answer,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote benchmark to {OUT}")
