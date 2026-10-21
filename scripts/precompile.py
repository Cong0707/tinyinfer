"""Precompile every tilelang kernel the engine uses, so cold starts only load cubins.

The first forward compiles (or reuses cached) kernels per (shape, M) combination; an
uncompiled shape costs ~5s, which is why a fresh start takes ~27s. This script walks the
real scenarios once (prefill chunks, decode, MTP stages) and leaves a warm
~/.tilelang/cache that can be shipped with a deployment.

Usage: python scripts/precompile.py [model_dir]
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

    t_all = time.time()
    print("== load engine", flush=True)
    eng = Engine.from_hf(model_dir, device="cuda", max_layers=40, allow_missing_layers=True)
    print(f"engine ready in {time.time()-t_all:.1f}s", flush=True)

    with torch.inference_mode():
        # prefill chunk sizes that trigger distinct kernel shapes
        for m in (8, 16, 32, 64):
            t0 = time.time()
            ids = torch.tensor([list(range(1000, 1000 + m))], device="cuda")
            try:
                eng.forward(ids, 0)
                torch.cuda.synchronize()
                print(f"prefill M={m}: {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"prefill M={m} failed: {e}", flush=True)
            eng.reset()
            for b in eng._blocks.values():
                if hasattr(b, "attn") and hasattr(b.attn, "reset"):
                    b.attn.reset()
                if getattr(b, "engram", None) is not None:
                    b.engram.reset()
            if getattr(eng, "hashes", None) is not None:
                eng.hashes.reset()

        # decode M=1
        t0 = time.time()
        eng.forward(torch.tensor([[19923]], device="cuda"), 0)
        torch.cuda.synchronize()
        print(f"decode M=1: {time.time()-t0:.1f}s", flush=True)

        # MTP stages (their kernels are separate)
        try:
            if eng.load_mtp():
                t0 = time.time()
                ids = torch.tensor([[19923]], device="cuda")
                eng.forward(ids, 0)
                th = eng._target_hidden
                if th:
                    s0 = eng.mtp[0]
                    main_x = s0.main_hiddens([th[i] for i in sorted(th.keys())])
                    x = s0.forward_embed(ids)
                    pre = torch.zeros(1, x.size(1), eng.cfg.hc_mult, device="cuda")
                    pre[:, :, 0] = 1.0
                    for st in eng.mtp:
                        x, pre = st.forward(x, 1, pre, main_x)
                    eng.mtp[-1].forward_head(x, pre, ids)
                torch.cuda.synchronize()
                print(f"mtp chain: {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"mtp precompile skipped: {e}", flush=True)

    print(f"precompile done in {time.time()-t_all:.1f}s; cache at ~/.tilelang", flush=True)


if __name__ == "__main__":
    main()
