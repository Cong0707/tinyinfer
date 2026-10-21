"""Precompile the long-context prefill chunk (M=2048) once.

Engine.prefill_chunk is 2048, so a long prompt is split into 2048-token chunks; the
first such chunk triggers a tilelang compile per distinct (shape, M) combination
(~5s each, tens of minutes in total). Run this once and the cache makes long-context
prefill start immediately afterwards.

Usage: python scripts/precompile_long.py [model_dir]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/Models/DeepSeek-V4.1-Flash-1017")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyinfer.engine import Engine

    eng = Engine.from_hf(model_dir, device="cuda", max_layers=40, allow_missing_layers=True)
    ids = torch.tensor([list(range(1000, 1000 + 2048))], device="cuda")
    t0 = time.time()
    try:
        out = eng.forward(ids, 0)
        torch.cuda.synchronize()
        print(f"M=2048 prefill: {time.time()-t0:.1f}s finite={torch.isfinite(out).all().item()}", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"M=2048 OOM at {time.time()-t0:.1f}s: {e}", flush=True)
    print("long-context precompile done", flush=True)


if __name__ == "__main__":
    main()
