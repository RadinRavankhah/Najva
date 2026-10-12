"""
Estimate the USD cost of translating a Jev-format JSONL with gpt-4o-mini.

Method:
  1. tiktoken (o200k_base) counts the EXACT English input tokens per API call,
     including the prompt prefix that is re-sent with every text.
  2. Persian output is longer in tokens than English input. Two ways to set the
     multiplier:
       --measure N   make N real API calls and read response.usage to get the
                     real output/input ratio (costs a fraction of a cent)
       --ratio R     assume a fixed output/input token ratio (default 2.0)

Usage:
    python data/estimate_translation_cost.py data/persian_train_sample.jsonl
    python data/estimate_translation_cost.py data/persian_train_sample.jsonl --measure 20
    python data/estimate_translation_cost.py data/persian_train_sample.jsonl --rows 520199
"""
import argparse
import json
import os
import sys
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# The prompt prefix sent with every text (must match translate_to_persian.py)
PROMPT = (
    "Translate the following English text to Persian (Farsi). "
    "Return ONLY the translation, no quotes, no explanation, no markdown."
    "\n\nTEXT:\n"
)

# gpt-4o-mini pricing (USD per token)
INPUT_PRICE = 0.15 / 1_000_000
OUTPUT_PRICE = 0.60 / 1_000_000


def collect_texts(rows):
    """Yield every text that is sent to the translator, in the same order."""
    for r in rows:
        yield r.get("state", "")
        yield r.get("instructions", "")
        crit = r.get("criteria") or {}
        if isinstance(crit, dict):
            for v in crit.values():
                yield v
        elif isinstance(crit, list):
            for v in crit:
                yield v


def load_rows(path, limit=0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def measure_ratio(texts, n, api_key, base_url):
    """Translate a RANDOM sample of n texts and return (avg_input, avg_output).

    Random sampling matters: the first rows of the file are not representative
    (states near the top are longer), which biases the estimate.
    """
    import random
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    sample = random.sample(texts, min(n, len(texts)))
    in_tot = out_tot = 0
    for t in sample:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT + t}],
            temperature=0.0,
        )
        in_tot += resp.usage.prompt_tokens
        out_tot += resp.usage.completion_tokens
    return in_tot / len(sample), out_tot / len(sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--rows", type=int, default=0, help="Extrapolate to this many rows (default: file's own count)")
    ap.add_argument("--measure", type=int, default=0, help="Make N real API calls to calibrate output tokens")
    ap.add_argument("--ratio", type=float, default=1.2, help="Output/input-text token ratio if not measuring")
    args = ap.parse_args()

    rows = load_rows(args.input)
    texts = list(collect_texts(rows))
    n_calls = len(texts)

    enc = tiktoken.get_encoding("o200k_base")
    prompt_tokens = len(enc.encode(PROMPT))
    text_tokens = sum(len(enc.encode(t)) for t in texts)
    input_tokens = text_tokens + prompt_tokens * n_calls  # prompt re-sent per call

    print(f"Rows: {len(rows):,}")
    print(f"API calls (one per text): {n_calls:,}")
    print(f"English text tokens (tiktoken): {text_tokens:,}")
    print(f"Prompt overhead: {prompt_tokens} x {n_calls:,} = {prompt_tokens * n_calls:,}")
    print(f"TOTAL input tokens: {input_tokens:,}")

    if args.measure:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        if not api_key:
            raise SystemExit("--measure needs OPENAI_API_KEY in .env")
        avg_in, avg_out = measure_ratio(texts, args.measure, api_key, base_url)
        output_tokens = int(avg_out * n_calls)
        print(f"[measured] avg input={avg_in:.1f}  avg output={avg_out:.1f} "
              f"(output/input={avg_out / avg_in:.2f})")
    else:
        output_tokens = int(text_tokens * args.ratio)
        print(f"Output ratio assumed: {args.ratio:.2f} (text-only)")

    cost = input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE
    per_row = cost / len(rows)

    print(f"\nOutput tokens: {output_tokens:,}")
    print(f"  input cost:  {input_tokens * INPUT_PRICE:.4f} USD")
    print(f"  output cost: {output_tokens * OUTPUT_PRICE:.4f} USD")
    print(f"Cost for {len(rows):,} rows: ${cost:.4f}  (${per_row * 1000:.4f} per 1,000 rows)")

    if args.rows and args.rows != len(rows):
        scale = args.rows / len(rows)
        print(f"Extrapolated to {args.rows:,} rows: ${cost * scale:.2f}")


if __name__ == "__main__":
    main()
