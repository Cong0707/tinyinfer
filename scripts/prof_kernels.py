"""Kernel-level profile of one 40-layer decode step (where do the GPU ms go?).

Usage: python scripts/prof_kernels.py [model_dir] [tokens]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/Develop/Deepseek/tinyinfer/cache/weights")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyinfer.engine import Engine

    eng = Engine.from_hf(model_dir, device="cuda", max_layers=40, allow_missing_layers=True)
    ids = torch.tensor([[19923]], device="cuda")
    with torch.inference_mode():
        eng.forward(ids, 0)
        for p in (1, 2):
            eng.forward(ids, p)
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            eng.forward(ids, 3)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
