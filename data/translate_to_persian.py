"""
Translate a Jev-format JSONL from English to Persian (Farsi).

Only NATURAL-LANGUAGE fields are translated:
    state, instructions, criteria values (option texts)

EVERYTHING ELSE is copied untouched:
    question_id, category, type, answer, soft_target, split, source,
    criteria KEYS (opt0, opt1, …)

Engines (pick one):
    gpt4o-mini  — OpenAI-compatible API, cheapest reliable (~$0.36 per 1 K rows;
                  see docs/translation_cost.md for the measured estimate).
                  Reads OPENAI_API_KEY and OPENAI_BASE_URL from the local .env
                  file (see .env.example). CLI --api-key / --base-url override.
    nllb        — facebook/nllb-200-distilled-600M, FREE, offline (~1 GB download).
    argos       — Argos Translate, FREE, offline (pip install argostranslate).

Robustness (a failed run never loses completed work):
    * resume by default — rows already in the output are skipped (use --no-resume to start over)
    * every completed row is flushed to disk immediately
    * API calls retry with exponential backoff (--retries, default 5)
    * a row that still fails is written to <output>.failed.jsonl and skipped, not lost
    * Ctrl+C stops cleanly; re-run the same command to continue where it left off

Usage:
    python data/translate_to_persian.py data/persian_train_sample.jsonl
    python data/translate_to_persian.py data/jev_dataset.jsonl --engine nllb --limit 500
    python data/translate_to_persian.py data/persian_train_sample.jsonl --engine gpt4o-mini
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# .env is optional: if python-dotenv is missing, fall back to real env vars
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

PROMPT = (
    "Translate the following English text to Persian (Farsi). "
    "Return ONLY the translation, no quotes, no explanation, no markdown."
)


def _safe_print(msg, stream=sys.stdout):
    """Print without crashing on Windows consoles that can't encode Persian/arrows."""
    try:
        print(msg, file=stream, flush=True)
    except UnicodeEncodeError:
        enc = stream.encoding or "utf-8"
        print(msg.encode(enc, "replace").decode(enc, "replace"), file=stream, flush=True)


# ------------------------------------------------------------------ engines

def _translate_gpt4o_mini(texts, api_key=None, base_url=None, retries=5, timeout=60.0):
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    out = []
    for t in texts:
        last_err = None
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": PROMPT + "\n\nTEXT:\n" + t}],
                    temperature=0.0,
                )
                out.append(resp.choices[0].message.content.strip().strip("\"'"))
                last_err = None
                break
            except Exception as e:  # transient network / rate-limit / server error
                last_err = e
                time.sleep(min(30.0, 1.0 * (2 ** attempt)))  # 1s, 2s, 4s, 8s, 16s…
        if last_err is not None:
            raise last_err  # exhausted retries -> caller routes this row to the failed file
    return out


def _translate_nllb(texts):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    _safe_print("[translator] loading facebook/nllb-200-distilled-600M en->fa …")
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    fas_id = tok.convert_tokens_to_ids("fas_Arab")
    out = []
    for t in texts:
        inputs = tok(t[:1500], return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, forced_bos_token_id=fas_id, max_new_tokens=512)
        out.append(tok.decode(gen[0], skip_special_tokens=True))
    return out


def _translate_argos(texts):
    try:
        from argostranslate import translate as argos_translate
    except ImportError:
        raise SystemExit("argos: pip install argostranslate first")
    out = []
    for t in texts:
        out.append(argos_translate.translate(t, "en", "fa"))
    return out


ENGINES = {
    "gpt4o-mini": _translate_gpt4o_mini,
    "nllb": _translate_nllb,
    "argos": _translate_argos,
}


# ------------------------------------------------------------------ row logic

def translate_row(row, engine_fn, **kw):
    """Translate one row's text fields; keys/answer/soft_target are preserved.

    Only state / instructions / criteria VALUES go to the translator. The
    translations are placed back into the SAME nested structure — never as flat
    "criteria:key" top-level keys.
    """
    texts, fields = [], []

    texts.append(row["state"]); fields.append(("state",))
    texts.append(row["instructions"]); fields.append(("instructions",))
    crit = row.get("criteria") or {}
    if isinstance(crit, dict):
        for k, v in crit.items():
            texts.append(v); fields.append(("criteria", k))
    elif isinstance(crit, list):
        for j, v in enumerate(crit):
            texts.append(v); fields.append(("criteria", j))

    translated = engine_fn(texts, **kw)

    out = {**row}
    if isinstance(crit, dict):
        out["criteria"] = dict(crit)
    elif isinstance(crit, list):
        out["criteria"] = list(crit)

    for field, value in zip(fields, translated):
        if field[0] == "state":
            out["state"] = value
        elif field[0] == "instructions":
            out["instructions"] = value
        elif field[0] == "criteria":
            out["criteria"][field[1]] = value
    return out


def _load_done_ids(path):
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line).get("question_id"))
                    except json.JSONDecodeError:
                        pass  # tolerate a torn last line from a hard kill
    return done


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Translate a Jev JSONL to Persian")
    ap.add_argument("input", help="Input JSONL (English)")
    ap.add_argument("--output", default=None, help="Output JSONL (default: input with .fa.jsonl)")
    ap.add_argument("--engine", default="gpt4o-mini", choices=list(ENGINES))
    ap.add_argument("--api-key", default=None, help="API key (overrides OPENAI_API_KEY from .env)")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (overrides OPENAI_BASE_URL from .env)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retries", type=int, default=5, help="API retries per text with exponential backoff")
    ap.add_argument("--failed-file", default=None, help="Where failed rows go (default: <output>.failed.jsonl)")
    ap.add_argument("--no-resume", action="store_true", help="Ignore existing output and start from scratch")
    args = ap.parse_args()

    out_path = Path(args.output or str(args.input).replace(".jsonl", ".fa.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path = Path(args.failed_file) if args.failed_file else out_path.with_name(out_path.stem + ".failed.jsonl")

    # load input rows
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{args.input}:{ln}: {e}")
    if args.limit:
        rows = rows[:args.limit]
    _safe_print(f"[load] {len(rows):,} rows from {args.input}")

    # resume: skip rows already written (default behaviour)
    done_ids = set() if args.no_resume else _load_done_ids(out_path)
    if done_ids:
        _safe_print(f"[resume] {len(done_ids):,} rows already translated — skipping (use --no-resume to redo)")

    engine_fn = ENGINES[args.engine]
    kw = {}
    if args.engine == "gpt4o-mini":
        kw["api_key"] = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
        kw["base_url"] = base_url.strip() if base_url else None
        kw["retries"] = args.retries
        if not kw["api_key"]:
            raise SystemExit("gpt4o-mini: set OPENAI_API_KEY in .env (or pass --api-key)")
        _safe_print(f"[gpt4o-mini] base_url={kw['base_url'] or 'https://api.openai.com/v1 (default)'}  retries={args.retries}")

    translated = errors = skipped = 0
    t0 = time.time()
    interrupted = False

    out_mode = "a" if (out_path.exists() and not args.no_resume) else "w"
    failed_f = open(failed_path, "a", encoding="utf-8")
    try:
        with out_path.open(out_mode, encoding="utf-8") as out:
            for i, row in enumerate(rows, 1):
                qid = row.get("question_id", "?")
                if qid in done_ids:
                    skipped += 1
                    continue
                try:
                    result = translate_row(row, engine_fn, **kw)
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()  # a completed row is on disk before we move on
                    translated += 1
                except Exception as e:
                    errors += 1
                    failed_f.write(json.dumps(
                        {"question_id": qid, "error": f"{type(e).__name__}: {e}", "row": row},
                        ensure_ascii=False) + "\n")
                    failed_f.flush()
                    _safe_print(f"  FAILED {qid}: {type(e).__name__}: {e}", stream=sys.stderr)
                if i % 25 == 0 or i == len(rows):
                    _safe_print(f"  {i}/{len(rows)}  ok={translated} failed={errors} skipped={skipped}")
    except KeyboardInterrupt:
        interrupted = True
        _safe_print("\n[interrupted] progress saved — re-run the same command to resume")
    finally:
        failed_f.close()

    elapsed = time.time() - t0
    _safe_print(f"\n=== {'Interrupted' if interrupted else 'Done'} ===")
    _safe_print(f"Translated: {translated:,}  failed: {errors}  skipped: {skipped}  in {elapsed:.0f}s")
    _safe_print(f"Output: {out_path}")
    if errors:
        _safe_print(f"Failed rows (retry by re-running): {failed_path}")
    _safe_print("Cost estimate: python data/estimate_translation_cost.py " + str(args.input))


if __name__ == "__main__":
    main()
