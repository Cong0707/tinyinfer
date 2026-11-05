"""Native fp4 GEMV entry points (no tilelang).

deepseek/gemv_fp4.py imports tilelang at module scope (its kernels are tilelang jit
functions), so importing it fails on a deployment without tilelang even though every
call it serves now has a native kernel behind it. These wrappers expose just the native
path, and moe.py prefers them:

    fp4_native.gemv_fp4_pq  == gemv_fp4.gemv_fp4_pq on the native kernel (bit-exact
                               against the tilelang one on real expert weights)

gemv_fp4 is still imported as a fallback where it can be, so nothing changes on a box
that has tilelang.
"""
from __future__ import annotations

import torch


def _scale(s: torch.Tensor) -> torch.Tensor:
    """e8m0 scale bytes -> fp32 values (2^(b-127))."""
    if s.dtype == torch.float8_e8m0fnu:
        b = s.view(torch.uint8).to(torch.int32)
        return torch.pow(2.0, (b - 127).to(torch.float32))
    return s.float()


def gemv_fp4_pq(aq, asc, w_fp4, w_scale, ws=None):
    """fp8 activation [K] x fp4 weight [N, K/2] -> [1, N] bf16."""
    from ..kernels import native

    n, k2 = w_fp4.shape
    k = k2 * 2
    if ws is None:
        ws = _scale(w_scale)
    if ws.size(0) != n:
        ws = ws.view(-1, ws.size(-1)).repeat_interleave(32, 0)[:n]
    asc2 = asc if asc.dtype == torch.float32 else asc.float()
    out = torch.empty(n, dtype=torch.bfloat16, device=aq.device)
    wb = w_fp4 if w_fp4.dtype == torch.uint8 else w_fp4.view(torch.uint8)
    native.launch(native.kernel("gemv_fp4_w4a8", "gemv_fp4_w4a8"),
                  ((n + 7) // 8,), (256,),
                  [aq.reshape(-1), asc2.reshape(-1), wb.reshape(-1),
                   ws.reshape(-1), out, 1, int(n), int(k)])
    return out.view(1, n)


def quantize_act(x: torch.Tensor, k: int):
    """One fp8 activation quantisation for a whole expert batch -> (fp8, scale fp32).

    Same contract as gemv_fp4.quantize_act, on our own act_quant kernel. The scale comes
    out of the kernel as e8m0 (an exponent byte, 2^(b-127)), and the fp4 GEMV wants fp32
    scales, so it is expanded here once per layer instead of once per expert.
    """
    from ..kernels.quant import act_quant

    flat = x.reshape(-1, k) if x.dim() != 2 else x
    q, s = act_quant(flat, 32, "ue8m0", torch.float8_e8m0fnu)
    return q, _scale(s)


def available() -> bool:
    try:
        from ..kernels import native

        native.kernel("gemv_fp4_w4a8", "gemv_fp4_w4a8")
        return True
    except Exception:
        return False
