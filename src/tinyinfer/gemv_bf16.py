"""Dense bf16 GEMV fast path for M=1, and helpers to fuse same-input projections.

The card's limiting factor at decode is not bandwidth but how well small kernels cover
memory latency: measured 174 GB/s for a 1024x1024 weight, 540 GB/s for 6144x1024, and
700 GB/s when several projections are joined into one contiguous read. A transformer layer
at decode is a stack of small GEMVs over the *same* input row, so joining them is free at
load time and nearly 3x faster at run time (four projections: 64.2us -> 22.5us).
"""
from __future__ import annotations

import torch

from . import native

_ENABLED = True


def available() -> bool:
    global _ENABLED
    if not _ENABLED:
        return False
    try:
        native.kernel("gemv_bf16", "gemv_bf16")
        return True
    except Exception:
        _ENABLED = False
        return False


def gemv(x, W, out=None, warps: int = 8):
    """out = x @ W.T for a single row in bf16, fp32 accumulate. x:[K] or [1,K]."""
    if x.dim() == 2:
        if x.shape[0] != 1:
            raise ValueError("gemv_bf16 is the M=1 path")
        x = x[0]
    N, K = W.shape
    if out is None:
        out = torch.empty(N, dtype=torch.float32, device=x.device)
    # The kernel already strides by gridDim.x * warps, so the grid may be capped; capping
    # it too low silently leaves rows uncomputed (measured: a 248320-row head with the
    # grid clamped to 8192 produced 1/8 of the output and a "correct-looking" 676 GB/s).
    blocks = min((N + warps - 1) // warps, 65535)
    native.launch(native.kernel("gemv_bf16", "gemv_bf16"), (blocks,), (warps * 32,),
                  [x.contiguous(), W, out, N, K])
    return out


def join(weights):
    """Concatenate [N_i, K] weights along N for one launch (all must share K)."""
    return torch.cat([w.contiguous() for w in weights], dim=0).contiguous()
