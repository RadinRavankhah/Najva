# Najva — Project Status

**Goal**: Fine-tune `convaiinnovations/laya-multilingual` (322M, mmBERT-base) for Persian decision tasks using QLoRA. Full plan: `plans/laya_main_plan.md` (refined) / `plans/main_plan.md` (original).

**Last verified**: 2026-10-06 — all checks below were actually executed against the local venv and HF cache.

---

## ✅ Checklist

### Phase 0 — Environment & Setup
- [x] Project plan written (`plans/main_plan.md`, `plans/laya_main_plan.md`)
- [x] `requirements.txt` defined
- [x] Virtual environment created (`venv/`, Python 3.12.10)
- [x] PyTorch with CUDA installed (`torch 2.7.1+cu118`) — **CUDA works**
- [x] GPU detected: **NVIDIA GeForce RTX 3070 Laptop (8 GB VRAM)** — enough for QLoRA
- [x] Fine-tuning stack installed: `transformers 5.17.0`, `peft 0.21.0`, `bitsandbytes 0.50.2`, `accelerate 1.15.0`, `datasets 5.0.1`, `scikit-learn 1.9.1`
- [x] Laya installed (`laya 0.3.4`) — imports cleanly
- [x] **Laya-multilingual checkpoint downloaded** (614 MB blob in HF cache, complete)
- [x] **Inference verified end-to-end**: Persian ticket routed to multilingual checkpoint → predicted `billing` @ 99.99% confidence ✅

### Phase 1 — Baseline (Week 1 of plan)
- [x] Benchmark located: **[github.com/ArmanJR/Jev-Persian-Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark)** — 480 general questions (80 scenarios × 6: 3 choice + 2 noul + 1 score), 10 categories; lives on **GitHub, not the HF Hub**
- [x] Benchmark runner created (`benchmark_zero_shot.py`) — smoke-tested end-to-end ✅
- [ ] Clone the benchmark repo and convert its `data/scenarios.jsonl` + `data/gold.jsonl` into the runner's JSONL format (see `data/benchmark.example.jsonl`)
- [ ] Run zero-shot baseline on the 480 questions: `python benchmark_zero_shot.py --data data/benchmark.jsonl`
- [ ] Record baseline numbers — official Laya-multilingual reference scores: choice **58.75%**, noul **63.75%**, score (±0.5 rubric) **35.0%**, choice Brier **0.5995**

### Phase 2 — Training Data
**Strategy: take an existing Jev-formatted decision dataset (expected English), translate it to Persian, train on it.** No adaptation of non-decision Persian datasets (FarsTail, Digikala, ArmanEmo, PerCoR, …) — they are not in the Jev `choice`/`score`/`noul` format and would need lossy conversion; dropped from the plan.

- [x] A Persian dataset downloaded (`mshojaei77/persian-gk`, 5,897 rows) — ⚠️ chat/instruction format, not decision format; not usable for this pipeline
- [ ] Source a Jev-formatted decision dataset (expected English — e.g. `LocalLLaMA/typed-decisions` or similar typed-decision corpora) with choice / score / noul rows
- [ ] Translate the natural-language fields (`state`, `instructions`, `criteria`) to Persian — **never** translate option keys, label indices, or soft targets (the chinese-laya "Strategy D" approach)
- [ ] Quality check 5–10% of translated rows: natural Persian, labels still correct
- [ ] Build `data/persian_train.jsonl` + `data/persian_val.jsonl` (format documented by `data/benchmark.example.jsonl`)

### Phase 3 — Training (Weeks 3–4 of plan)
- [x] Training script written (`train_laya_fa.py`) — targets laya's **real** DecisionModel architecture (marker logits + soft-CE), reuses `build_sequence`/`collate_items`; LoRA r=16 on encoder attention + trainable decision head; **NOT run yet**
- [ ] Build `data/persian_train.jsonl` + `data/persian_val.jsonl` (training script consumes the same format as the benchmark)
- [ ] Run first training experiment (3 epochs, small data)
- [ ] Full training run (5 epochs, full dataset)

### Phase 4 — Evaluation (Week 5 of plan)
- [ ] Evaluate fine-tuned model on the 480-question benchmark
- [ ] Compare vs zero-shot baseline (accuracy + calibration)
- [ ] Check option-flip invariance (<5% target)

### Phase 5 — Export & Deploy (Week 6 of plan)
- [ ] Merge LoRA adapters into base model
- [ ] Export (ONNX / GGUF) + simple inference API

### Housekeeping
- [ ] Commit `requirements.txt` (currently untracked; repo has only the initial commit)
- [ ] Decide whether `plans/` should stay gitignored (it currently is)

---

## Verification Results (run today)

| Check | Result |
|:---|:---|
| Python (venv) | 3.12.10 |
| `import laya` | ✅ 0.3.4 |
| `torch.cuda.is_available()` | ✅ True — RTX 3070 Laptop |
| Model blob on disk | ✅ 614 MB `model.safetensors` (complete; one 0-byte stale `.incomplete` leftover, harmless) |
| Live inference test | ✅ Routed to `multilingual`, correct prediction, 0.9999 confidence |
| Dataset cache | `mshojaei77/persian-gk` (5,897 chat rows) — wrong format for decision tuning |

---

## Caveats & Notes

1. **Use the multilingual checkpoint directly, not `Router(preload=True)`.**
   `Router(preload=True)` pulls **all three** checkpoints (english + typed-decisions) from the bundle repo `convaiinnovations/laya` (~1.16 GB extra). For Persian-only work, load just:
   ```python
   from laya import Router
   r = Router(models={"multilingual": "convaiinnovations/laya-multilingual"}, default="multilingual")
   ```
   (Verified working — this is what the test used.)

2. **The plan's training script (section 5.3) does NOT match laya's architecture — use `train_laya_fa.py` instead.**
   The plan uses `AutoModelForSequenceClassification`, but laya 0.3.4 checkpoints are **RL-agent decision models** (`encoder.` / `type_emb.` / `scorer.` / `act_head.` weight prefixes, `rl_agent_config.json`, custom `DecisionModel` in `laya/common.py`). `train_laya_fa.py` already targets the real architecture — the plan's script is kept only as background reference.

3. **`transformers 5.17.0` is very new** — untested against the training stack. If `Trainer`/PEFT hit breaking changes, pin `transformers<5`.

4. **bitsandbytes 4-bit on Windows** is installed but unverified. Test with a tiny load before committing to a full QLoRA run.

5. The `persian-gk` dataset in cache is Node.js/MySQL chat data — likely downloaded for a different experiment. It does **not** fit the Jev decision schema (`state` / `instructions` / `criteria` / `answer`).

---

## Translation cost estimate (English → Persian, before training)

**Workload:** translate the natural-language fields (`state`, `instructions`, `criteria`) of a Jev-formatted English dataset while keeping option keys, label indices, and soft targets untouched.

| Source dataset size | Avg words/row | Total English words | Total tokens (×1.3) |
|:---|:---:|---:|---:|
| POC (500 examples) | ~50 | 25,000 | 325K |
| Production (5,000) | ~50 | 250,000 | 3.25M |
| Research quality (10,000+) | ~50 | 500,000+ | 6.5M+ |

**Estimated cost to translate 5,000 examples (production):**

| Method | Cost | Quality | Notes |
|:---|:---|:---|:---|
| **GPT-4o mini / Claude Haiku** (API) | **$0.05–0.20** | Good — prompt to keep keys/labels untouched | Cheapest reliable option; batch with `messages` array |
| **GPT-4o / Claude Sonnet 4** (API) | **$0.80–2.00** | Better on nuance, sarcasm, تعارف | Use for a 5–10% sample instead of all rows |
| Google Cloud / DeepL API | **$25–30** | Good | Per-character pricing hurts; no structural awareness of Jev rows |
| Helsinki-NLP Opus-MT (en→fa) | **$0** | Fair | Offline; `opus-mt-en-fa`; may mangage Persian script/digits |
| Argos Translate | **$0** | Fair | Offline; smaller models; slower |
| LibreTranslate free tier | **$0** | Fair | Rate-limited; not reliable at scale |
| Human translation | **$50–200+** | Best | Only for the 5–10% review sample per the plan's quality-control step |

**Cheapest viable path (~$0–5 for 5K rows):**
1. Translate all rows with GPT-4o mini (or Opus-MT if you want $0) into a draft JSONL.
2. Review 5–10% via a cheap LLM consistency check (translate back → compare) — near-zero cost.
3. Human spot-check only the flagged rows (maybe 50 rows → negligible cost).

**Total expected translation spend for this project: $0–30**, dominated by the review method you pick. The model's training-data size (5K rows × ~50 words) is tiny by LLM standards — translation cost is not a real bottleneck here; data quality and label correctness are.

---

## Recommended Next Step

**Phase 1: zero-shot baseline.** It requires no training data, validates the whole pipeline, and gives you the number your fine-tuned model must beat:

```powershell
git clone https://github.com/ArmanJR/Jev-Persian-Benchmark ..\Jev-Persian-Benchmark
.\venv\Scripts\Activate.ps1
# 1. Convert the benchmark's data/scenarios.jsonl + data/gold.jsonl into the runner format
# 2. Run all 480 questions:
python benchmark_zero_shot.py --data data\benchmark.jsonl
```

