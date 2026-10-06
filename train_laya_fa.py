"""
Fine-tune Laya-multilingual for Persian decision tasks (LoRA / QLoRA-style) on local JSONL data.

*** NOT RUN YET - created as part of the plan (plans/laya_main_plan.md Phase 3). ***

    python train_laya_fa.py --train data/persian_train.jsonl --val data/persian_val.jsonl

Input format (one JSON object per line - identical to benchmark_zero_shot.py):
    {"state": "...", "type": "choice", "instructions": "...", "criteria": {"key": "desc", ...}, "answer": "key"}
    {"state": "...", "type": "score",  "instructions": "...", "criteria": ["lvl0", "lvl1", ...], "answer": 1}
    {"state": "...", "type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}, "answer": true}
    Optional: "soft_target": [...]  (list of K floats) overrides the one-hot built from `answer` -
    useful for translated soft-target data (the chinese-laya "Strategy D" approach).

WHY THIS SCRIPT DOES NOT FOLLOW plans/laya_main_plan.md SECTION 5.3 VERBATIM:
    The plan's training script uses AutoModelForSequenceClassification. That is WRONG for
    this checkpoint: laya 0.3.4 ships a custom "RL agent" architecture - DecisionModel
    (laya/common.py) = HF encoder + typed-decision head (type_emb / scorer / act_head),
    driven by [MASK] marker positions, NOT a classification head. Its checkpoint contains
    rl_agent_config.json and weight prefixes encoder. / type_emb. / scorer. / act_head.,
    which an HF SequenceClassification head can neither load nor reproduce. So this script
    trains the REAL architecture: we reuse laya's own build_sequence / collate_items so
    training sequences are byte-identical to inference sequences, and put the loss on the
    marker logits.

Design notes:
    - Memory: the encoder is frozen and trained via LoRA only. Weights stay fp32 with bf16
      autocast (stable, no GradScaler needed). Base (322M fp32 ~1.3 GB) + LoRA/head grads +
      checkpointed activations fits comfortably in ~3-4 GB, well under the RTX 3070's 8 GB.
    - 4-bit (bitsandbytes) is intentionally NOT implemented: on Windows the bnb runtime is
      fragile, it cannot be combined with this checkpoint's weight layout without custom
      conversion, and - with 8 GB VRAM - it is unnecessary. If you move to a 4 GB GPU,
      revisit (quantize the encoder before wrapping with PEFT).
    - act_head (action-cost head) is frozen: the decision datasets carry no action labels,
      so there is nothing to train it on.

Output (out_dir, default ./laya-fa-checkpoints):
    adapter/             PEFT LoRA adapter (adapter_model.safetensors + adapter_config.json)
    heads.safetensors    decision head weights: head / type_emb / scorer / act_head / temperature
    rl_agent_config.json + tokenizer/   copied from the base checkpoint

Loading the result for inference:
    # Option A (after --merge): the folder becomes a normal laya checkpoint:
    #     from laya import load; agent = load("<out_dir>")
    # Option B (adapter, no merge):
    #     build the base model exactly like laya does (build_model + load_state_dict),
    #     torch.load heads.safetensors into it (strict=False), then
    #     PeftModel.from_pretrained(base, "<out_dir>/adapter") and .eval().
"""
import argparse
import copy
import json
import os
import random
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE_MODEL = "convaiinnovations/laya-multilingual"

# suffixes of attention projection Linear layers we consider for LoRA.
# mmBERT is ModernBERT-family (fused Wqkv / Wo), BERT-family uses query/key/value.
ATTN_SUFFIXES = ("query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "Wqkv", "Wo")


# --------------------------------------------------------------------------- data

def row_to_item(row: dict, tok, max_len: int, head_max_len: int, ln: int):
    """One JSONL row -> one pre-tokenized training item (exactly what inference builds)."""
    from laya.common import QTYPES, build_sequence

    t = row.get("type")
    if t not in QTYPES:
        raise SystemExit(f"line {ln}: 'type' must be one of {list(QTYPES)}, got {t!r}")
    crit = row.get("criteria")
    if t == "choice":
        if not isinstance(crit, dict):
            raise SystemExit(f"line {ln}: choice needs 'criteria' as dict")
        keys = list(crit.keys())
        ans = row["answer"]
        if ans not in keys:
            raise SystemExit(f"line {ln}: answer {ans!r} not in criteria keys")
    elif t == "score":
        if not isinstance(crit, list):
            raise SystemExit(f"line {ln}: score needs 'criteria' as list")
        keys = list(range(len(crit)))
        ans = int(row["answer"])
        if not 0 <= ans < len(crit):
            raise SystemExit(f"line {ln}: score answer {ans} out of range")
    else:  # noul
        keys = ["false", "true"]
        ans = bool(row["answer"])

    # target distribution over the K option markers (soft targets supported)
    if "soft_target" in row:
        target = [float(x) for x in row["soft_target"]]
        if len(target) != len(keys) or abs(sum(target) - 1.0) > 1e-3:
            raise SystemExit(f"line {ln}: soft_target must have {len(keys)} entries summing to 1")
    elif t == "noul":
        target = [1.0 - float(ans), float(ans)]
    else:
        target = [1.0 if k == ans else 0.0 for k in keys]

    qdef = {"t": t, "ins": str(row["instructions"]), "crit": crit if t != "noul" else (crit or {})}
    ids, markers = build_sequence(tok, row["state"], qdef, max_len, head_max_len)
    return {"ids": ids, "markers": markers, "qtype": QTYPES[t], "target": target, "label": keys.index(ans)}


def load_split(path: str, tok, max_len: int, head_max_len: int):
    from laya.common import collate_items

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            items.append(row_to_item(json.loads(line), tok, max_len, head_max_len, ln))

    def collate(batch):
        return collate_items([[it] for it in batch], tok.pad_token_id)

    return items, collate


# --------------------------------------------------------------------------- model

def resolve_base_dir(model_ref: str) -> str:
    """Local path if given; otherwise snapshot_download the standalone repo (NOT the bundle)."""
    if os.path.exists(model_ref):
        return model_ref
    from huggingface_hub import snapshot_download
    return snapshot_download(model_ref)


def detect_lora_targets(model, verbose=True):
    """Find attention Linear layers inside the encoder and build a PEFT regex for them.

    Auto-detection because target module names differ across encoder families
    (mmBERT/ModernBERT: Wqkv, Wo; BERT: query/key/value/dense) and hardcoding the
    wrong names is the #1 way a LoRA run silently trains nothing.
    """
    names = set()
    for full_name, mod in model.named_modules():
        if not full_name.startswith("encoder."):
            continue  # never touch the decision head with the encoder regex
        if isinstance(mod, torch.nn.Linear):
            suffix = full_name.split(".")[-1]
            if suffix in ATTN_SUFFIXES:
                names.add(suffix)
    if not names:  # unknown family: fall back to every Linear in the encoder
        for full_name, mod in model.named_modules():
            if full_name.startswith("encoder.") and isinstance(mod, torch.nn.Linear):
                names.add(full_name.split(".")[-1])
    pattern = r"encoder\..*(%s)$" % "|".join(sorted(names))
    if verbose:
        print(f"[lora] target modules regex: {pattern}")
    return pattern


def build_trainable_model(base_dir: str, lora_r: int, lora_alpha: int, lora_dropout: float, device: torch.device):
    """Load the pretrained RL-agent checkpoint, freeze the encoder, attach LoRA."""
    from laya.common import build_model
    from peft import LoraConfig, get_peft_model

    with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    from safetensors.torch import load_file
    model = build_model(cfg, encoder_dir=os.path.join(base_dir, "encoder")
                        if os.path.isdir(os.path.join(base_dir, "encoder")) else None)
    weights = load_file(os.path.join(base_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    # keep ModernBERT off torch.compile (same reason laya's Agent does this)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass

    # gradient checkpointing on the frozen encoder: activations are recomputed per step,
    # trading ~30% step time for the bulk of activation memory
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    pattern = detect_lora_targets(model)
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=pattern, bias="none",
    )
    model = get_peft_model(model, lora_cfg)  # freezes everything except LoRA adapters
    model.base_model.model.act_head.requires_grad_(False)  # no action labels in the data

    # unfreeze the (small) decision head so it can adapt to the Persian domain too
    head_model = model.base_model.model
    for block in (head_model.head, head_model.type_emb, head_model.scorer):
        for p in block.parameters():
            p.requires_grad_(True)

    model.to(device)
    verbose_trainable(model)
    return model, cfg


def verbose_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {trainable / 1e6:.1f}M / total {total / 1e6:.1f}M "
          f"({100 * trainable / total:.2f}%)")


# --------------------------------------------------------------------------- train/eval

def make_optim(model, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    try:  # 8-bit optimizer if bitsandbytes works on this machine, else plain AdamW
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.01)
    except Exception as e:
        print(f"[optim] bitsandbytes unavailable ({type(e).__name__}); using AdamW")
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


def loss_from_logits(logits, batch):
    """Soft cross-entropy on the option-marker logits.

    The model already masks invalid marker slots to -1e4, so log_softmax puts ~0
    probability there and the target (length K, padded with zeros) picks it up cleanly.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(batch["target"].to(logp.device) * logp).sum(-1).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    from laya.common import confidence_from_probs, ece_score

    model.eval()
    per_type = {}
    total_loss, n = 0.0, 0
    for batch in loader:
        if batch is None:
            continue
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                              batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                              batch["qtype"].to(device))
        total_loss += loss_from_logits(logits, batch).item() * len(batch["label"])
        n += len(batch["label"])
        for i, qt in enumerate(batch["qtype"].tolist()):
            k = int(batch["marker_mask"][i].sum())
            p = torch.softmax(logits[i, :k].float(), -1).cpu().numpy()
            rec = per_type.setdefault(int(qt), {"conf": [], "cor": []})
            rec["conf"].append(confidence_from_probs(p, k))
            rec["cor"].append(float(int(p.argmax()) == int(batch["label"][i])))
    model.train()
    names = {0: "choice", 1: "score", 2: "noul"}
    out = {"loss": total_loss / max(1, n)}
    for qt, rec in per_type.items():
        conf, cor = np.array(rec["conf"]), np.array(rec["cor"])
        out[names[qt]] = {"acc": round(float(cor.mean()), 4), "ece": round(ece_score(conf, cor), 4), "n": len(cor)}
    return out


def merge_and_export(peft_model, out_dir):
    """Rebuild a full fp32 checkpoint (LoRA merged in) that laya.load() can open directly.

    Runs on CPU fp32 (merging needs unquantized weights; ~1.3 GB RAM, no GPU needed).
    Merging mutates weights in place, so it operates on a deep copy - the live training
    object (and any further training) stays untouched.
    """
    from peft import PeftModel

    merged = PeftModel.from_pretrained(copy.deepcopy(peft_model.base_model.model),
                                       os.path.join(out_dir, "adapter"))
    merged = merged.merge_and_unload().cpu().eval()

    from safetensors.torch import save_file
    sd = {k: v.contiguous().cpu() for k, v in merged.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    print(f"[merge] full checkpoint written: {out_dir}{os.sep}model.safetensors")
    print(f"[merge] load with: from laya import load; agent = load({out_dir!r})")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Fine-tune laya RL-agent checkpoint on Persian decision data")
    ap.add_argument("--train", required=True, help="training JSONL")
    ap.add_argument("--val", default=None, help="validation JSONL (optional)")
    ap.add_argument("--model", default=BASE_MODEL, help="standalone laya checkpoint (repo id or local dir)")
    ap.add_argument("--out", default="./laya-fa-checkpoints")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4, help="per-device batch (questions)")
    ap.add_argument("--accum", type=int, default=16, help="gradient accumulation (effective = batch * accum)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--head-max-len", type=int, default=192)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--merge", action="store_true", help="after training, export a merged full checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    # ---- base checkpoint (standalone multilingual repo - never the 3-checkpoint bundle)
    base_dir = resolve_base_dir(args.model)
    print(f"[model] base checkpoint: {base_dir}")

    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    tok_dir = os.path.join(base_dir, "tokenizer")
    _fix_tokenizer_config(base_dir)  # patch tokenizer_config quirks before loading (same as laya.Agent)
    tok = AutoTokenizer.from_pretrained(tok_dir if os.path.isdir(tok_dir) else json.load(
        open(os.path.join(base_dir, "rl_agent_config.json")))["encoder"])
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.sep_token_id

    # ---- data
    train_items, collate = load_split(args.train, tok, args.max_len, args.head_max_len)
    val_loader = None
    if args.val:
        val_items, collate = load_split(args.val, tok, args.max_len, args.head_max_len)
        val_loader = DataLoader(val_items, batch_size=args.batch, shuffle=False, collate_fn=collate)
    print(f"[data] train={len(train_items)}" + (f"  val={len(val_items)}" if args.val else ""))

    # ---- model
    model, cfg = build_trainable_model(base_dir, args.lora_r, args.lora_alpha, args.lora_dropout, device)
    model.train()

    steps_per_epoch = max(1, len(train_items) // (args.batch * args.accum))
    total_steps = steps_per_epoch * args.epochs
    from transformers import get_linear_schedule_with_warmup
    optimizer = make_optim(model, args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    loader = DataLoader(train_items, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=False)

    # ---- training loop
    step, done, running = 0, 0, 0.0
    micro = 0
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            if batch is None:
                continue
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                  batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                                  batch["qtype"].to(device))
            loss = loss_from_logits(logits, batch) / args.accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            done += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch}  step {step}/{total_steps}  loss {running / (args.accum * args.log_every):.4f}")
                running = 0.0
        if val_loader:
            print(f"[eval] epoch {epoch}: {evaluate(model, val_loader, device)}")

    # ---- save
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(os.path.join(args.out, "adapter"))
    from safetensors.torch import save_file
    head_model = model.base_model.model
    heads_sd = {}
    for name in ("head", "type_emb", "scorer", "act_head"):
        for k, v in getattr(head_model, name).state_dict().items():
            heads_sd[f"{name}.{k}"] = v.contiguous().cpu()
    heads_sd["temperature"] = head_model.temperature.cpu()
    save_file(heads_sd, os.path.join(args.out, "heads.safetensors"), metadata={"format": "pt"})
    shutil.copy(os.path.join(base_dir, "rl_agent_config.json"), args.out)
    if os.path.isdir(tok_dir):
        shutil.copytree(tok_dir, os.path.join(args.out, "tokenizer"), dirs_exist_ok=True)
    print(f"[save] adapter + heads + config -> {args.out}")

    if args.merge:
        merge_and_export(model, args.out)


if __name__ == "__main__":
    main()
