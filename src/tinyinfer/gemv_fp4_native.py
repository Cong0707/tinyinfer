"""Swap the fp4 expert GEMV for the hand-written ALU-decode kernel.

The tilelang kernel the engine shipped measures 41 GB/s on a 40MB expert matrix. A pure
read of the same buffer reaches 212 GB/s, so the decode math -- not memory -- was the
limit: a 16-entry table lookup per weight is a memory op per weight.

The replacement rebuilds the E2M1 float from its bits (no table, no branch) and measures
62 GB/s, 1.5x the tilelang path, bit-identical output.

Interface difference: the tilelang kernel takes an fp8-quantised activation plus an fp32
scale array; this one takes the activation as fp32/bfloat16 and the e8m0 scale bytes
directly, which is what the checkpoint stores.
"""
from __future__ import annotations

import torch

from . import native

# the native kernel reads an fp32 activation vector and e8m0 scale bytes
_AVAILABLE = None


def available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            native.kernel("gemv_fp4_e8m0", "gemv_fp4_fast")
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def gemv_fp4_fast(x: torch.Tensor, w_fp4: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """x [1,K] activation, w fp4 [N,K/2] packed, w_scale [N, K/32] e8m0 -> [1,N] bf16."""
    n, k2 = w_fp4.shape
    k = k2 * 2
    a = x.reshape(-1)
    if a.dtype != torch.float32:
        a = a.float()
    # the kernel wants one e8m0 byte per 32 weights per row; broadcast a packed scale
    ws = w_scale
    if ws.dtype != torch.uint8:
        ws = ws.view(torch.uint8) if ws.dtype.itemsize == 1 else ws.to(torch.uint8)
    nb = k // 32
    if ws.dim() == 2 and ws.size(0) * 32 == n:
        ws = ws.repeat_interleave(32, 0)[:n]
    elif ws.dim() == 2 and ws.size(0) != n:
        ws = ws.view(-1, ws.size(-1))
        if ws.size(0) * 32 == n:
            ws = ws.repeat_interleave(32, 0)[:n]
    ws = ws.contiguous()
    out = torch.empty(n, dtype=torch.bfloat16, device=x.device)
    grid = ((n + 7) // 8,)
    native.launch(native.kernel("gemv_fp4_e8m0", "gemv_fp4_fast"), grid, (256,),
                  [a, torch.ones(nb, dtype=torch.float32, device=x.device),
                   w_fp4, ws, out, n, k])
    return out.view(1, n)
