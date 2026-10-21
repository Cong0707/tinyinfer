"""Quantization headroom experiment: how much error would a private sub-4bit expert
format add relative to the official fp4 (E2M1 + ue8m0 per-32) packing.

We take one real expert weight, dequantize it to bf16 (reference), then simulate
2/3/4-bit quantization with per-32 scales (matching the official block layout) and
report relative error and bytes per token.

Run: python scripts/quant_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def quantize_sim(w: np.ndarray, bits: int, block: int = 32) -> np.ndarray:
    """Per-block symmetric quantization to 2^bits levels, back to float."""
    flat = w.reshape(-1)
    n = flat.size
    pad = (-n) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blk = flat.reshape(-1, block)
    absmax = np.abs(blk).max(axis=1, keepdims=True)
    absmax = np.where(absmax == 0, 1.0, absmax)
    levels = 2 ** (bits - 1) - 1
    q = np.round(blk / absmax * levels).clip(-levels - 1, levels)
    out = q * absmax / levels
    return out.reshape(-1)[:n].astype(np.float32)


def e2m1_values() -> np.ndarray:
    """E2M1 grid used by the official fp4 experts."""
    vals = []
    for e in range(4):
        for m in range(2):
            if e == 0:
                v = m * 0.5
            else:
                v = (1 + m * 0.5) * (2.0 ** (e - 1))
            vals += [v, -v]
    return np.array(sorted(set(vals)), dtype=np.float32)


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyinfer.config import TinyConfig
    from tinyinfer.weights import ShardStore
    from tinyinfer.quant import dequant_fp4_block

    model_dir = Path("E:/Develop/Deepseek/tinyinfer/cache/weights")
    cfg = TinyConfig.from_hf(model_dir)
    store = ShardStore(model_dir)
    p = "layers.5.ffn.experts.0.w1."
    w = store.get(p + "weight")
    s = store.get(p + "scale")
    print("expert w1:", tuple(w.shape), w.dtype, "scale:", tuple(s.shape), flush=True)

    import torch
    ref = dequant_fp4_block(w, s).float().cpu().numpy()
    g = e2m1_values()
    print("E2M1 grid:", g[:8], "...", len(g), "levels", flush=True)

    rms = float(np.sqrt((ref.astype(np.float32) ** 2).mean()))
    ref_f = ref.astype(np.float32).reshape(-1)
    for bits in (2, 3, 4):
        q = quantize_sim(ref_f, bits)
        err = float(np.sqrt(((q - ref_f) ** 2).mean()))
        print(f"sim {bits}bit: rel_err={err/rms:.3f}  levels={2**bits}", flush=True)

    # official fp4 E2M1 reference error (round-trip of the packed values)
    q4 = np.vectorize(lambda v: g[np.argmin(np.abs(g - v))])(ref.astype(np.float32))
    err4 = float(np.sqrt(((q4 - ref.astype(np.float32)) ** 2).mean()))
    print(f"fp4 E2M1 round-trip: rel_err={err4/rms:.3f}", flush=True)
    print(flush=True)
    print("bytes/token @ 384 experts x 6 activated x 40 layers:", flush=True)
    per = 18_800_640
    for b in (2, 3, 4):
        print(f"  {b}bit-effective: {per * b / 4 * 240 / (1 << 30):.2f} GB/token (official fp4 = 4.51 GB)")


if __name__ == "__main__":
    main()
