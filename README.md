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
├── benchmark_zero_shot.py       # Zero-shot baseline runner: base checkpoint vs. benchmark
├── train_laya_fa.py             # LoRA fine-tuning script targeting laya's real architecture
├── data/
│   └── benchmark.example.jsonl  # Reference format for benchmark/training rows
├── results/                     # Baseline & evaluation outputs (JSON)
├── plans/                       # Project planning documents (gitignored)
├── STATUS.md                    # Living checklist of project status
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

## Evaluation

Najva is evaluated against the [Jev-Persian-Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark): 480 authored general-Persian questions (80 scenarios × 3 choice + 2 noul + 1 score) across 10 categories — intent, sentiment, negation, idioms, pragmatics/تعارف, scenario decisions, moderation, semantic matching, and extraction.

The zero-shot Laya-multilingual reference scores on the general set — the numbers Najva's fine-tuning aims to beat:

| Metric | Laya-multilingual (zero-shot) |
|:---|:---|
| Choice accuracy (240 q) | 58.75% |
| Noul accuracy (160 q) | 63.75% |
| Score within ±0.5 rubric levels (80 q) | 35.0% |
| Choice Brier ↓ | 0.5995 |
| Noul Brier ↓ | 0.2823 |
| Score MAE ↓ | 0.7238 |

## Project status

Development is in progress — see [STATUS.md](STATUS.md) for the live checklist. Environment, inference verification, the benchmark runner, and the training script are complete; the zero-shot baseline run and training-data translation are next.

## Acknowledgements

- [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations — the base decision model and runtime
- [Jev-Persian-Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark) by Arman Jafarnezhad — the Persian evaluation suite
- The Jev ecosystem (typed decisions, choice/score/noul primitives)
