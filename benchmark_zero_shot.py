"""
Zero-shot baseline: run the base Laya-multilingual checkpoint over a Persian decision
benchmark and record the numbers the fine-tuned model must beat.

Usage:
    python benchmark_zero_shot.py --data data/benchmark.jsonl
    python benchmark_zero_shot.py --data data/benchmark.example.jsonl --limit 3

Input format (one JSON object per line, matches train_laya_fa.py):
    {"state": "...", "type": "choice", "instructions": "...", "criteria": {"key": "desc", ...}, "answer": "key"}
    {"state": "...", "type": "score",  "instructions": "...", "criteria": ["lvl0", "lvl1", "lvl2"], "answer": 1}
    {"state": "...", "type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}, "answer": true}

    Note: `ArmanJR/Jev-Persian-Benchmark` is NOT on the Hugging Face Hub (verified
    2026-10-06 with a hub search). Until the benchmark file is obtained, point --data
    at a local JSONL in the format above. `data/benchmark.example.jsonl` documents it.

Outputs:
    results/zero_shot_baseline.json  - per-question records + aggregate metrics
    stdout summary table
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

BASE_MODEL = "convaiinnovations/laya-multilingual"


# --------------------------------------------------------------------------- data

def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: invalid JSON ({e})")
            t = row.get("type")
            if t not in ("choice", "score", "noul"):
                raise SystemExit(f"{path}:{ln}: 'type' must be choice|score|noul, got {t!r}")
            for key in ("state", "instructions", "answer"):
                if key not in row:
                    raise SystemExit(f"{path}:{ln}: missing required field {key!r}")
            if t == "choice" and not isinstance(row.get("criteria"), dict):
                raise SystemExit(f"{path}:{ln}: choice rows need 'criteria' as a dict of key -> description")
            if t == "score" and not isinstance(row.get("criteria"), list):
                raise SystemExit(f"{path}:{ln}: score rows need 'criteria' as a list of level descriptions")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows found")
    return rows


def question_def(row: dict) -> dict:
    """Convert a benchmark row into the question definition laya's Agent expects."""
    q = {"type": row["type"], "instructions": row["instructions"]}
    if "criteria" in row:
        q["criteria"] = row["criteria"]
    return q


# --------------------------------------------------------------------------- metrics

def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error. Same definition as laya.common.ece_score."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def brier(probs: list, answer_idx: int) -> float:
    """Multiclass Brier score (sum over classes). Lower is better; 2/K for a perfect
    one-hot prediction, ~2 for uniform guessing on 4 options."""
    p = np.asarray(probs, dtype=np.float64)
    target = np.zeros_like(p)
    target[answer_idx] = 1.0
    return float(((p - target) ** 2).sum())


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Zero-shot baseline for Laya-multilingual on a Persian decision benchmark")
    ap.add_argument("--data", default="data/benchmark.jsonl", help="Benchmark JSONL file (see module docstring)")
    ap.add_argument("--model", default=BASE_MODEL, help="Standalone checkpoint repo id (must be a laya RL-agent checkpoint)")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows (0 = all). For smoke tests.")
    ap.add_argument("--output", default="results/zero_shot_baseline.json")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--skip-flip", action="store_true", help="Skip the option-flip invariance test for choice questions")
    args = ap.parse_args()

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]

    # WHY NOT Router(preload=True):
    #   1. preload=True downloads and builds ALL THREE checkpoints (english, multilingual,
    #      typed-decisions) from the bundle repo `convaiinnovations/laya` - roughly 1.16B
    #      parameters / ~2.3 GB extra download. For a Persian-only benchmark the English
    #      and typed-decisions checkpoints are dead weight: the router's script detection
    #      sends Persian (non-Latin script) to the multilingual checkpoint 100% of the time.
    #   2. Instead, mount ONLY the multilingual standalone repo and pin it as the default,
    #      and force model="multilingual" on every call. One checkpoint (~644 MB), one
    #      resident model, deterministic routing, zero bundle traffic.
    #   (If you ever need mixed-language evaluation, Router(preload=["multilingual"])
    #   preloads selectively too - plain preload=True is the expensive one.)
    from laya import Router

    router = Router(models={"multilingual": args.model}, default="multilingual", device=args.device)

    records = []
    flip_records = []
    for i, row in enumerate(rows):
        qid = row.get("question_id", f"row-{i}")
        try:
            res = router.predict(row["state"], {"q": question_def(row)}, model="multilingual")
        except Exception as e:  # keep going on a bad row, but record it
            records.append({"question_id": qid, "type": row["type"], "error": str(e)})
            continue
        a = res["answers"]["q"]
        rec = {"question_id": qid, "type": row["type"], "category": row.get("category")}
        if row["type"] == "choice":
            keys = list(row["criteria"].keys())
            ans_idx = keys.index(row["answer"])
            probs = [a["probabilities"][k] for k in keys]
            pred_idx = int(np.argmax(probs))
            rec.update(answer=row["answer"], predicted=a["choice"], correct=pred_idx == ans_idx,
                       confidence=a["confidence"], brier=brier(probs, ans_idx))
        elif row["type"] == "score":
            ans = int(row["answer"])
            probs = [a["probabilities"][str(j)] for j in range(len(row["criteria"]))]
            pred_lvl = int(np.argmax(probs))
            rec.update(answer=ans, predicted_level=pred_lvl, expected_score=a["score"],
                       correct=pred_lvl == ans, confidence=a["confidence"], brier=brier(probs, ans))
        else:  # noul
            ans = bool(row["answer"])
            p_true = float(a["noul"])
            rec.update(answer=ans, predicted=p_true > 0.5, p_true=p_true,
                       correct=(p_true > 0.5) == ans, confidence=a["confidence"],
                       brier=float((p_true - float(ans)) ** 2))
        records.append(rec)

        # Option-flip invariance (choice only): re-ask with the options in reverse order.
        # A robust decision model should pick the same option regardless of position.
        # This goes through the same Agent path (and its calibration), so it measures
        # the deployed behaviour, not the raw encoder.
        if row["type"] == "choice" and not args.skip_flip:
            crit_rev = dict(reversed(list(row["criteria"].items())))
            res2 = router.predict(row["state"], {"q": {"type": "choice", "instructions": row["instructions"], "criteria": crit_rev}}, model="multilingual")
            flip_records.append({"question_id": qid, "original": a["choice"], "flipped": res2["answers"]["q"]["choice"],
                                 "flipped_result": res2["answers"]["q"]["choice"] != a["choice"]})

        done, total = i + 1, len(rows)
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    # ---------------------------------------------------------------- aggregate
    def agg(sub):
        if not sub:
            return None
        conf = np.array([r["confidence"] for r in sub], dtype=np.float64)
        cor = np.array([float(r["correct"]) for r in sub], dtype=np.float64)
        return {
            "n": len(sub),
            "accuracy": round(float(cor.mean()), 4),
            "brier_mean": round(float(np.mean([r["brier"] for r in sub])), 4),
            "ece": round(ece(conf, cor), 4),
        }

    summary = {"model": args.model, "data": args.data, "timestamp": datetime.now(timezone.utc).isoformat()}
    for t in ("choice", "score", "noul"):
        summary[t] = agg([r for r in records if r["type"] == t and "error" not in r])
    summary["overall"] = agg([r for r in records if "error" not in r])
    summary["errors"] = sum(1 for r in records if "error" in r)
    if flip_records:
        summary["option_flip_rate"] = round(sum(1 for f in flip_records if f["flipped_result"]) / len(flip_records), 4)
        summary["option_flip_n"] = len(flip_records)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records, "flip_test": flip_records}, f, ensure_ascii=False, indent=2)

    print("\n=== Zero-shot baseline summary ===")
    for t in ("choice", "score", "noul", "overall"):
        s = summary[t]
        if s:
            print(f"{t:8s} acc={s['accuracy']:.4f}  brier={s['brier_mean']:.4f}  ece={s['ece']:.4f}  n={s['n']}")
    if "option_flip_rate" in summary:
        print(f"flip     rate={summary['option_flip_rate']:.4f}  (target < 0.05)")
    print(f"\nSaved: {args.output}")
    return summary


if __name__ == "__main__":
    main()
