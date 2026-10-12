# Benchmark Results — Comparison

Official reference scores from [Jev-Persian-Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark):

| Model | Choice (240) | Noul (160) | Score ±0.5 (80) | Choice Brier ↓ | Noul Brier ↓ | Score MAE ↓ |
|---|---|---|---|---|---|---|
| **Laya multilingual (official)** | 58.75% | 63.75% | 35.0% | 0.5995 | 0.2823 | 0.7238 |

Our zero-shot baseline (this run):

| Task | n | Accuracy | Brier (mean) | ECE |
|---|---|---|---|---|
| Choice | 240 | 56.67% | 0.5867 | 0.2879 |
| Score (±0.5 rubric) | 80 | 38.75% | 0.7249 | 0.1879 |
| Noul (p ≥ 0.5) | 160 | 58.13% | 0.3014 | 0.2605 |
| Overall | 480 | 54.17% | 0.5146 | 0.2408 |

Flip rate (option-flip invariance, choice only): **27.5%** (240 questions).
This is above the <5% target. Fine-tuning (Phase 3–5) should reduce it.

---

Raw results: `results/zero_shot_baseline.json`
Benchmark data: `data/benchmark.jsonl` (derived from `scenarios.jsonl` + `gold.jsonl`)
Benchmark source: https://github.com/ArmanJR/Jev-Persian-Benchmark

---

To get results in CSV format from the JSON:
```python
import json
with open("results/zero_shot_baseline.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print("model,task,n,accuracy,brier_mean,ece")
for task in ["choice", "score", "noul", "overall"]:
    s = data["summary"].get(task)
    if s:
        print(f"{data['summary']['model']},{task},{s['n']},{s['accuracy']},{s['brier_mean']},{s['ece']}")
```
