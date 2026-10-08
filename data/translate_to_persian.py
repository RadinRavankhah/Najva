"""
Translate a Jev-format JSONL from English to Persian (Farsi).

Only NATURAL-LANGUAGE fields are translated:
    state, instructions, criteria values (option texts)

EVERYTHING ELSE is copied untouched:
    question_id, category, type, answer, soft_target, split, source,
    criteria KEYS (opt0, opt1, …)

Engines (pick one):
    gpt4o-mini  — OpenAI API, cheapest reliable (~$0.05-0.20 per 5 K rows).
                  Set OPENAI_API_KEY env var.
    opus-mt     — Helsinki-NLP Opus-MT en→fa, FREE, offline.
    argos       — Argos Translate, FREE, offline (pip install argostranslate).

Usage:
    python data/translate_to_persian.py data/jev_dataset_train.jsonl
    python data/translate_to_persian.py data/jev_dataset.jsonl --engine opus-mt --limit 500
    python data/translate_to_persian.py data/jev_dataset.jsonl --engine gpt4o-mini
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


# ------------------------------------------------------------------ engines

def _translate_gpt4o_mini(texts, api_key, batch=16):
    import openai
    client = openai.OpenAI(api_key=api_key)
    out = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i + batch]
        completions = []
        for t in batch_texts:
            completions.append(
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            "Translate the following English text to Persian (Farsi). "
                            "Return ONLY the translation, no quotes, no explanation, no markdown."
                        ) + f"\n\nTEXT:\n{t}"
                    }],
                    temperature=0.0,
                )
            )
        for c in completions:
            out.append(c.choices[0].message.content.strip().strip("\"'"))
        print(f"  gpt4o-mini: {min(i + batch, len(texts))}/{len(texts)}", flush=True)
        time.sleep(0.5)  # mild rate-limit cushion
    return out


def _translate_nllb(texts):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("[translator] loading facebook/nllb-200-distilled-600M en->fa …")
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
        import argostranslate.package
        import argostranslate.translate
    except ImportError:
        raise SystemExit("argos: pip install argostranslate first")
    import argostranslate.package
    from argostranslate import translate as argos_translate
    out = []
    for t in texts:
        out.append(argos_translate.translate(t, "en", "fa"))
    return out


ENGINES = {
    "gpt4o-mini": _translate_gpt4o_mini,
    "nllb": _translate_nllb,
    "argos": _translate_argos,
}


# ------------------------------------------------------------------ main

def translate_row(row, engine_fn, **kw):
    """Translate a single row's text fields via engine_fn(list_of_texts) → list."""
    texts, fields = [], []

    texts.append(row["state"]); fields.append("state")
    texts.append(row["instructions"]); fields.append("instructions")
    crit = row.get("criteria") or {}
    if isinstance(crit, dict):
        for k, v in crit.items():
            texts.append(v); fields.append(f"criteria:{k}")
    elif isinstance(crit, list):
        for j, v in enumerate(crit):
            texts.append(v); fields.append(f"criteria[{j}]")

    translated = engine_fn(texts, **kw) if len(texts) > 1 else engine_fn(texts)

    it = iter(translated)
    out = {**row}
    for f in fields:
        out[f] = next(it)
    return out


def main():
    ap = argparse.ArgumentParser(description="Translate a Jev JSONL to Persian")
    ap.add_argument("input", help="Input JSONL (English)")
    ap.add_argument("--output", default=None, help="Output JSONL (default: input with .fa.jsonl)")
    ap.add_argument("--engine", default="gpt4o-mini", choices=list(ENGINES))
    ap.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                     help="Skip rows already present in output (by question_id)")
    args = ap.parse_args()

    out_path = Path(args.output or str(args.input).replace(".jsonl", ".fa.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load input rows
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{args.input}:{ln}: {e}")
    if args.limit:
        rows = rows[:args.limit]
    print(f"[load] {len(rows):,} rows from {args.input}")

    # resume: skip already-written question_ids
    done_ids = set()
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r.get("question_id"))
        print(f"[resume] {len(done_ids):,} already translated, skipping")

    engine_fn = ENGINES[args.engine]
    kw = {}
    if args.engine == "gpt4o-mini":
        kw["api_key"] = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not kw["api_key"]:
            raise SystemExit("gpt4o-mini: set OPENAI_API_KEY env var or pass --api-key")

    errors = 0
    translated = 0
    t0 = time.time()

    mode = "a" if (args.resume and out_path.exists()) else "w"
    with out_path.open(mode, encoding="utf-8") as out:
        for row in rows:
            qid = row.get("question_id", "?")
            if qid in done_ids:
                continue
            try:
                result = translate_row(row, engine_fn, **kw)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                translated += 1
            except Exception as e:
                errors += 1
                import traceback
                msg = f"  ERROR {qid}: {type(e).__name__}: {e}"
                try:
                    print(msg, file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                except UnicodeEncodeError:
                    for line in traceback.format_exc().encode(sys.stderr.encoding, errors="replace").decode(sys.stderr.encoding, errors="replace").splitlines():
                        print(line, file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"Translated: {translated:,}  errors: {errors}  in {elapsed:.0f}s")
    print(f"Output: {out_path}")

    # cost estimate (gpt4o-mini: $0.15 / 1M input tokens; ~1.3 tokens/char)
    if args.engine == "gpt4o-mini":
        chars = sum(len(r.get("state", "")) + len(r.get("instructions", "")) for r in rows)
        est = chars * 1.3 / 1e6 * 0.15
        print(f"Estimated API cost (gpt4o-mini): ${est:.2f} on {chars:,} chars")


if __name__ == "__main__":
    main()
