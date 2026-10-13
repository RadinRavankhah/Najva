# Najva

**A System 1, Jev-like decision model for Persian.**

*Najva* (نجوا) — Persian for *whisper*: decisions made instantly and quietly, without generating a single token of reasoning.

Najva fine-tunes [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) — a 322M-parameter mmBERT-base encoder with a typed-decision head — into a fast, calibrated decision engine purpose-built for Persian text. Instead of autoregressively thinking out loud, it reads a `state` once and answers typed questions in a single parallel forward pass.

---

## What it does

Given a Persian text (a support ticket, a message, a log, a JSON payload), Najva answers three kinds of typed questions:

| Type | Question shape | Answer |
|:---|:---|:---|
| **choice** | Pick one of K options | Option key + full probability distribution |
| **score** | Position on an ordinal rubric | Expected level + per-level probabilities |
| **noul** | P(yes) for a yes/no question | Probability of *true* |

```python
from laya import Router

router = Router(models={"multilingual": "convaiinnovations/laya-multilingual"},
                default="multilingual")

result = router.predict(
    "مشتری می‌گوید دو بار بابت یک خرید شارژ شده است و رسید هر دو تراکنش را دارد.",
    {"team": {"type": "choice",
              "instructions": "کدام تیم باید این درخواست را بررسی کند؟",
              "criteria": {"billing": "صورت‌حساب و پرداخت", "technical": "مشکل فنی",
                           "sales": "فروش", "other": "سایر"}}},
    model="multilingual",
)
# → {'choice': 'billing', 'probabilities': {'billing': 0.9999, ...}, 'confidence': 0.999}
```

> **Note:** load the standalone multilingual checkpoint as shown above — not `Router(preload=True)`, which downloads all three checkpoints of the laya bundle (~1.16 GB extra) even though Persian text always routes to the multilingual one.

## Why System 1?

Large language models make decisions by generating reasoning — powerful, but slow, expensive, and hard to calibrate. A System 1 decision model instead behaves like trained intuition: it maps situations directly to typed decisions with calibrated confidence.

- **Fast** — one forward pass per request (single-digit ms on GPU, ~100 ms on CPU)
- **Small** — 322M parameters; runs on a laptop GPU or even CPU
- **Calibrated** — probabilities are real probabilities, not sampled tokens
- **Typed** — choice / score / noul questions with structured, machine-readable answers
- **Persian-native** — evaluated on authored Persian benchmarks, including colloquial register, تعارف, negation scope, and Finglish

## Repository layout

```
├── api.py                        # Jev-style FastAPI server (POST /v1/systemone)
├── benchmark_zero_shot.py        # Zero-shot baseline runner: base checkpoint vs. benchmark
├── train_laya_fa.py              # LoRA fine-tuning script targeting laya's real architecture
├── data/
│   ├── benchmark.jsonl           # 480-question Persian benchmark (derived from Jev-Persian-Benchmark)
│   ├── benchmark.example.jsonl   # Reference format for benchmark/training rows
│   ├── prepare_jev_dataset.py    # Download OpenAGILab/Jev-dataset → Najva JSONL format
│   ├── translate_to_persian.py   # English → Persian (gpt-4o-mini / nllb / argos engines)
│   ├── estimate_translation_cost.py  # gpt-4o-mini USD cost estimate for translation
│   └── persian_train_sample.jsonl    # 5,200-row sample for smoke tests & cost estimates
├── docs/
│   └── translation_cost.md       # Measured translation cost breakdown (~$0.36 / 1K rows)
├── results/                      # Baseline outputs & comparison (JSON gitignored; MD/CSV summary committed)
├── STATUS.md                     # Living checklist of project status
└── requirements.txt
```

## Setup

Requires Python 3.10+ and (ideally) an NVIDIA GPU with ≥4 GB VRAM.

```powershell
git clone https://github.com/RadinRavankhah/Najva
cd Najva

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The model checkpoint (~644 MB) downloads automatically on first use from the Hugging Face Hub.

## Usage

**Zero-shot baseline** — measure how the base model performs on Persian decisions before any fine-tuning:

```powershell
python benchmark_zero_shot.py --data data/benchmark.jsonl
```

Reports accuracy, Brier score, ECE, and option-flip invariance per question type; full per-question records are saved to `results/`.

**Fine-tuning** — LoRA (r=16) on the encoder's attention layers plus the decision head, trained with soft cross-entropy on the marker logits:

```powershell
python train_laya_fa.py --train data/persian_train.jsonl --val data/persian_val.jsonl --merge
```

The script consumes the same row format as the benchmark (`state` / `type` / `instructions` / `criteria` / `answer`; soft targets supported). With `--merge`, the output folder becomes a self-contained checkpoint loadable via `laya.load()`.

**Training data** — Najva trains on the [OpenAGILab/Jev-dataset](https://huggingface.co/datasets/OpenAGILab/Jev-dataset) (already in the Jev typed-decision format with soft targets), translated English → Persian:

```powershell
# 1. Download & convert to Najva JSONL (default: redistributable subset ~114K rows; --all-subsets for all 520K)
python data/prepare_jev_dataset.py

# 2. Estimate translation cost (optional; needs OPENAI_API_KEY in .env)
python data/estimate_translation_cost.py data/persian_train_sample.jsonl --measure 30 --rows 520199

# 3. Translate the English rows to Persian (resumable; failed rows land in a .failed.jsonl)
python data/translate_to_persian.py data/jev_dataset_train.jsonl --engine gpt4o-mini
```

`--engine nllb` (free, offline, ~1 GB) and `--engine argos` (free, offline) are alternatives to `gpt4o-mini`. Measured cost for the paid engine: **$1.88 per 5,200 rows** (~$0.36 / 1,000 rows) — see [`docs/translation_cost.md`](docs/translation_cost.md).

**Serve as an API** — [`api.py`](api.py) wraps the model in a Jev-shaped FastAPI endpoint:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"state":"متن نمونه","model":"najva-latest","questions":{"q1":{"type":"noul","instructions":"آیا این پیام فوری است؟"}}}'
```

Answers mirror the Jev response schema (`noul` / `choice` / `score` objects with `probabilities`, `confidence`, `usage`); `GET /v1/models` and `GET /health` are also exposed, and FastAPI serves interactive Swagger docs at `/`. Optional bearer-token auth: set `Najva_API_KEY`. Until fine-tuning lands, the server uses the base `laya-multilingual` checkpoint under aliases like `najva-latest` — to switch it to the trained model, point `CHECKPOINT_REPO` at your checkpoint.

## Evaluation

Najva is evaluated against the [Jev-Persian-Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark): 480 authored general-Persian questions (80 scenarios × 3 choice + 2 noul + 1 score) across 10 categories — intent, sentiment, negation, idioms, pragmatics/تعارف, scenario decisions, moderation, semantic matching, and extraction.

The zero-shot baseline over the base Laya-multilingual checkpoint has been run end-to-end on all 480 questions — these measured numbers are what Najva's fine-tuning aims to beat:

| Task | n | Accuracy | Brier (mean) | ECE |
|:---|---:|---:|---:|---:|
| Choice | 240 | 56.67% | 0.5867 | 0.2879 |
| Score (within ±0.5 rubric levels) | 80 | 38.75% | 0.7249 | 0.1879 |
| Noul (p ≥ 0.5) | 160 | 58.13% | 0.3014 | 0.2605 |
| Overall | 480 | 54.17% | 0.5146 | 0.2408 |

Choice option-flip rate: **27.5%** — above the <5% invariance target, so fine-tuning must bring it down. For reference, the [Jev-Persian-Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark) publishes official Laya-multilingual scores of choice 58.75%, noul 63.75%, and score (±0.5) 35.0% on the same set. Full side-by-side: [`results/BENCHMARK_COMPARISON.md`](results/BENCHMARK_COMPARISON.md); raw records: `results/zero_shot_baseline.json`.

## Project status

Development is in progress — see [STATUS.md](STATUS.md) for the live checklist. The environment, end-to-end inference check, benchmark runner, and the zero-shot baseline run are complete; the Jev-dataset preparation and translation scripts are written and verified, with the full English→Persian training-data run next, followed by fine-tuning and evaluation against the baseline above.

## Acknowledgements

- [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations — the base decision model and runtime
- [Jev-Persian-Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark) by Arman Jafarnezhad — the Persian evaluation suite
- The Jev ecosystem (typed decisions, choice/score/noul primitives)
