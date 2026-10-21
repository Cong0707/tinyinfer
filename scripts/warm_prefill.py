"""Warm the tilelang JIT kernels for prefill chunk sizes (long-context readiness).

Each new M value (chunk length) triggers a tilelang compile on first use (5-10 min
on this box). Run this once before serving long prompts: it compiles M=8/16/32/64/128.

Usage: python scripts/warm_prefill.py [model_dir]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/Develop/Deepseek/tinyinfer/cache/weights")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyinfer.engine import Engine

    eng = Engine.from_hf(model_dir, device="cuda", max_layers=40, allow_missing_layers=True)
    for m in (8, 16, 32, 64, 128):
        t0 = time.time()
        ids = torch.tensor([list(range(1000, 1000 + m))], device="cuda")
        try:
            eng.forward(ids, 0)
            torch.cuda.synchronize()
            print(f"warm M={m}: {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"warm M={m} failed: {e}", flush=True)
        # reset KV between sizes (fresh prefill each time)
        for b in eng._blocks.values():
            if hasattr(b, "attn") and hasattr(b.attn, "reset"):
                b.attn.reset()
            if getattr(b, "engram", None) is not None:
                b.engram.reset()
        if getattr(eng, "hashes", None) is not None:
            eng.hashes.reset()
    print("prefill warm done", flush=True)


if __name__ == "__main__":
    main()
