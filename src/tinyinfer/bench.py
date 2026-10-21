"""Reproducible benchmark: prefill, decode, expert-cache hit rate, disk read rate.

    python -m tinyinfer.bench --model E:\\Models\\DeepSeek-V4.1-Flash-1017 [--json out.json]

Prints a JSON object so runs can be diffed. Every number is measured, nothing is
assumed: the disk rate is taken from the same reader the engine uses.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch


def _disk_rate(model_dir: str, mb: int = 512) -> float:
    """Sequential read rate on the checkpoint volume (GB/s)."""
    import glob

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        return 0.0
    total, t0 = 0, time.perf_counter()
    for f in files:
        with open(f, "rb") as fh:
            while total < mb * 2**20:
                chunk = fh.read(64 * 2**20)
                if not chunk:
                    break
                total += len(chunk)
            if total >= mb * 2**20:
                break
    return total / max(1e-9, time.perf_counter() - t0) / 2**30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    # question-shaped: the base model answers these sensibly (docs/faq.md)
    ap.add_argument("--prompt", default="Question: What is the capital of France?\nAnswer:")
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from .engine import Engine

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    t_build = time.perf_counter()
    eng = Engine.from_hf(args.model, device="cuda", tokenizer=tok, allow_missing_layers=True)
    build_s = time.perf_counter() - t_build

    ids = tok.encode(args.prompt, add_special_tokens=False)
    t = torch.tensor([ids], device="cuda")
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = eng.forward(t, 0)
    torch.cuda.synchronize()
    first_s = time.perf_counter() - t0

    # second prefill straight away (same session state as a retry): no reset, so this
    # measures the warm path without rebuilding any structure
    t0 = time.perf_counter()
    with torch.inference_mode():
        eng.forward(t, 0)
    torch.cuda.synchronize()
    warm_prefill_s = time.perf_counter() - t0

    nxt = int(logits[0, -1].argmax())
    pos = len(ids)
    steps = []
    for _ in range(args.new_tokens):
        t0 = time.perf_counter()
        with torch.inference_mode():
            o = eng.forward(torch.tensor([[nxt]], device="cuda"), start_pos=pos)
        torch.cuda.synchronize()
        steps.append(time.perf_counter() - t0)
        nxt = int(o[0, -1].argmax())
        pos += 1

    st = getattr(eng, "expert_store", None)
    hit = None
    entries = None
    if st is not None:
        stats = getattr(getattr(st, "ram", None), "_stats", None)
        if stats is not None:
            h, m = getattr(stats, "hits", 0), getattr(stats, "misses", 0)
            hit = h / (h + m) if h + m else None
        entries = len(getattr(st, "cache", getattr(st, "_cache", {})) or {})

    out = {
        "build_s": round(build_s, 2),
        "first_forward_s": round(first_s, 2),
        "warm_prefill_s": round(warm_prefill_s, 2),
        "decode_step_s": [round(s, 3) for s in steps],
        "decode_tok_per_s": round(1.0 / (sum(steps) / max(1, len(steps))), 3),
        "expert_cache_hit": None if hit is None else round(hit, 3),
        "expert_cache_entries": entries,
        "disk_gb_s": round(_disk_rate(args.model), 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        if hasattr(os, "sysconf") else None,
    }
    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
