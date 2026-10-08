"""
Download OpenAGILab/Jev-dataset and convert to Najva's laya JSONL format.

The Jev dataset ships as 12 control-task subsets. This script loads one or
all of them, filters by question type / source task, and writes one JSONL
per split plus a combined file — the same format expected by
benchmark_zero_shot.py and train_laya_fa.py.

Default: the redistributable subset only (legally safe).  Use --all-subsets
for the full 520 K rows.

Usage:
    python data/prepare_jev_dataset.py                          # 114 K rows, redistributable
    python data/prepare_jev_dataset.py --all-subsets            # 520 K rows
    python data/prepare_jev_dataset.py --limit 500              # smoke test
    python data/prepare_jev_dataset.py --sources customer-control-v1   # single task
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

BASE_REPO = "OpenAGILab/Jev-dataset"
REDIST_SUBSET = "release-v2-redistributable"


def _flatten_target(target):
    """Some score rows nest lists inside the target vector; flatten by max."""
    out = []
    for v in target:
        if isinstance(v, (list, tuple)):
            out.append(max(v) if v else 0.0)
        else:
            out.append(float(v))
    return out


def convert_row(row):
    """One Jev row → one laya-format decision row."""
    kind = row["kind"]
    options = row["options"]
    target = _flatten_target(row["target"])

    if kind == "choice":
        criteria = {f"opt{i}": opt for i, opt in enumerate(options)}
        ans_key = f"opt{int(max(range(len(target)), key=lambda i: target[i]))}"
    elif kind == "score":
        criteria = list(options)
        ans_key = int(max(range(len(target)), key=lambda i: target[i]))
    else:  # noul
        criteria = {
            "false": options[0] if len(options) > 0 else "no",
            "true": options[1] if len(options) > 1 else "yes",
        }
        ans_key = bool(target[1] > 0.5)

    return {
        "state": row["state_json"],
        "question_id": row["id"],
        "category": row["source"],
        "type": kind,
        "instructions": row["question"],
        "criteria": criteria,
        "answer": ans_key,
        "soft_target": target,
        "split": row.get("split", "unknown"),
        "source": row["source"],
    }


def load_dataset_subset(subset, splits):
    from datasets import load_dataset

    print(f"[download] loading subset {subset!r} splits {splits} …")
    if not splits:
        return {"train": load_dataset(BASE_REPO, subset, split="train")}
    split_str = "+".join(splits)
    try:
        ds = load_dataset(BASE_REPO, subset, split=split_str)
        return {"train": ds} if not isinstance(ds, dict) else ds
    except Exception:
        result = {}
        for s in splits:
            try:
                result[s] = load_dataset(BASE_REPO, subset, split=s)
            except Exception:
                print(f"  [warn] split {s!r} not found, skipping")
        return result


def main():
    ap = argparse.ArgumentParser(description="Convert OpenAGILab/Jev-dataset → Najva JSONL")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--subset", default=REDIST_SUBSET,
                    help=f"HF subset name (default: {REDIST_SUBSET})")
    ap.add_argument("--all-subsets", action="store_true",
                    help="Load ALL 12 subsets (~520 K rows) instead of the redistributable one")
    ap.add_argument("--splits", nargs="+", default=["train", "calibration", "validation", "test", "ood"],
                    help="Splits to include")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Filter to these source task names (e.g. customer-control-v1)")
    ap.add_argument("--kinds", nargs="*", default=None, choices=["choice", "score", "noul"],
                    help="Filter to these question types")
    ap.add_argument("--limit", type=int, default=0, help="Max rows per split (0 = all)")
    ap.add_argument("--combined", default=None, help="Also write a combined file (default: <output-dir>/jev_dataset.jsonl)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all_subsets:
        subset = None  # load all configs
        print("[warn] Loading ALL 12 subsets — ~520 K rows, ~900 MB download")
    else:
        subset = args.subset

    raw = load_dataset_subset(subset, args.splits if args.splits else None)

    counters = Counter()
    combined = []
    per_split_files = {}

    # open per-split writers
    for split in args.splits:
        path = Path(args.output_dir) / f"jev_dataset_{split}.jsonl"
        per_split_files[split] = path.open("w", encoding="utf-8")

    for split_name, ds in raw.items():
        if split_name not in per_split_files:
            continue
        out = per_split_files[split_name]
        n = 0
        for row in ds:
            if args.sources and row["source"] not in args.sources:
                continue
            if args.kinds and row["kind"] not in args.kinds:
                continue
            if args.limit and n >= args.limit:
                break
            try:
                converted = convert_row(row)
            except Exception as e:
                print(f"  WARN: skipping row {row.get('id', '?')}: {e}", file=sys.stderr)
                counters["_errors"] += 1
                continue
            out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            combined.append(converted)
            counters[split_name] += 1
            counters[row["kind"]] += 1
            counters[row["source"]] += 1
            n += 1

    for f in per_split_files.values():
        f.close()

    # combined file
    comb_path = Path(args.output_dir) / (args.combined or "jev_dataset.jsonl")
    with comb_path.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n=== Conversion complete ===")
    print(f"Combined: {comb_path}  ({len(combined):,} rows)")
    for split in args.splits:
        print(f"  {split}: {counters.get(split, 0):,} rows")
    print(f"\nBy question type:")
    for k in ("choice", "score", "noul"):
        print(f"  {k}: {counters.get(k, 0):,}")
    print(f"\nBy source task (top 10):")
    for src, cnt in counters.most_common(10):
        if src in ("train", "calibration", "validation", "test", "ood"):
            continue
        print(f"  {src}: {cnt:,}")


if __name__ == "__main__":
    main()
