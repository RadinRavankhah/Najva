# Translation Cost Estimate — English → Persian (gpt-4o-mini)

**Measured on:** `data/persian_train_sample.jsonl` (5,200 rows, 1% of the full dataset)
**Method:** `data/estimate_translation_cost.py` — exact input tokens via `tiktoken` (`o200k_base`) + a random-sample API measurement to calibrate Persian output tokens.

## How the numbers were produced

1. **Input tokens (exact).** Every row sends `state`, `instructions`, and each `criteria` value to the translator — one API call per text field. tiktoken counts the English tokens, and the 27-token prompt prefix is added once per call.
2. **Output tokens (measured).** A random sample of 30 texts was translated for real and the API's `usage` field read back. Measured: **avg input = 103.5 tokens/call, avg output = 86.5 tokens/call** (Persian output ≈ 0.84× the input, prompt included).
3. **Pricing:** gpt-4o-mini — $0.15 / 1M input tokens, $0.60 / 1M output tokens.

## Results

| Quantity | Value |
|:---|---:|
| Rows | 5,200 |
| API calls (one per text field) | 27,530 |
| English text tokens | 2,274,325 |
| Prompt overhead (27 × 27,530) | 743,310 |
| **Total input tokens** | **3,017,635** |
| **Output tokens** | **2,380,427** |
| Input cost | $0.45 |
| Output cost | $1.43 |
| **Cost for 5,200 rows** | **$1.88** |
| Per 1,000 rows | $0.36 |
| **Extrapolated to 520,199 rows (full dataset)** | **$188.16** |

> Earlier rough estimate in STATUS.md said "$0.05–0.20 per 5K rows" — that was wrong by ~10×. It counted input only and ignored both the per-call prompt overhead and the output tokens. These measured numbers supersede it.

## Cost drivers & optimisations

- **Output tokens dominate** (~76% of cost). Persian output is generated token-by-token, so it is priced at 4× the input rate. This is the floor unless a cheaper output model is used.
- **Prompt overhead is ~25% of input.** Each text re-sends the 27-token instruction prefix. Batching several texts per call (or sending a whole row as one JSON object) would cut this, but risks parse errors — not worth it at this scale.
- **Free alternatives** (`--engine nllb`, `--engine argos`) cost $0 but produce lower-quality Persian; use them only if budget is a hard constraint.

## Reproduce

```powershell
python data/estimate_translation_cost.py data/persian_train_sample.jsonl --measure 30 --rows 520199
```
