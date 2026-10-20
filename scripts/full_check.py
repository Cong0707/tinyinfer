"""Readiness check + quick benchmark once the full 40-layer download is complete.

Usage:
    python scripts/full_check.py                     # check only
    python scripts/full_check.py --bench             # check + prefill/decode timing
    python scripts/full_check.py --layers 0-39       # custom layer set
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch


def parse_layers(text):
    if not text:
        return None
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "cache" / "weights"))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--tier", default="balanced")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--decode", type=int, default=8)
    args = ap.parse_args()

    from safetensors import safe_open
    from tinyinfer.config import TinyConfig
    from tinyinfer.weights import ShardStore

    model_dir = Path(args.model_dir)
    cfg = TinyConfig.from_hf(str(model_dir))
    store = ShardStore(str(model_dir))
    layers = parse_layers(args.layers)

    # shard integrity for the requested layers
    complete = 0
    for i in (layers if layers else range(cfg.n_layers)):
        shard = store.shard_of(f"layers.{i}.attn.wq_a.weight")
        path = model_dir / shard
        ok = False
        if path.exists():
            try:
                with safe_open(str(path), framework="pt", device="cpu") as f:
                    ok = f"layers.{i}.attn.wq_b.weight" in f.keys()
            except Exception:
                ok = False
        if ok:
            complete += 1
        else:
            print(f"layer {i}: INCOMPLETE ({shard})")
    total = len(layers) if layers else cfg.n_layers
    print(f"layers complete: {complete}/{total}")
    if complete < total:
        print("download not finished; run again later")
        return 2

    if not args.bench:
        return 0

    from tinyinfer.engine import Engine
    from tinyinfer.tiers import OptLevel

    eng = Engine(str(model_dir), tier=OptLevel[args.tier.upper()], device="cuda",
                 layers=layers, allow_missing_layers=False)
    prompt = list(range(16))
    with torch.inference_mode():
        t0 = time.perf_counter()
        logits = eng.forward(torch.tensor([prompt]), 0)
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1e3
        nid = int(logits[0, -1].argmax())
        t0 = time.perf_counter()
        for step in range(args.decode):
            logits = eng.forward(torch.tensor([[nid]]), len(prompt) + step)
            nid = int(logits[0, -1].argmax())
        torch.cuda.synchronize()
        dec = (time.perf_counter() - t0) / args.decode * 1e3
    print(f"prefill 16 tok: {ttft:.0f} ms | decode: {dec:.0f} ms/token ({1000/dec:.2f} tok/s)")
    print("stats:", json.dumps(eng.stats(), default=str)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
