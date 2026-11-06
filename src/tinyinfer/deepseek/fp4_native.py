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


# Launchers for the e8m0/shared-memory variant, keyed by K (its dynamic smem depends on
# it). The kernel takes the checkpoint's scale bytes directly, so the caller no longer
# pays a pow(2, s-127) pass over [N, K/32] before every expert: that conversion measured
# 9.3 us against a 20.9 us kernel and inflated the scale table 4x.
_E8M0_BOUND: dict = {}
_E8M0_SMEM_RAISED = False


def _e8m0_launcher(n: int, k: int):
    global _E8M0_SMEM_RAISED
    from ..kernels import native

    # Keyed by (n, k): the grid is a function of n and the dynamic shared memory a
    # function of k, so a k-only key handed a launcher built for one output size to a
    # caller asking for another (cuLaunchKernel returned 1 on the next expert shape).
    key = (n, k)
    hit = _E8M0_BOUND.get(key)
    if hit is not None:
        return hit
    nb = k // 32
    smem = ((k + 15) & ~15) + 4 * nb
    # No set_dynamic_smem: this kernel asks for K + 4*K/32 bytes, which is 5.8 KB at
    # K=5120 -- far below the 48 KB a launch gets without raising the function attribute.
    # (Raising it to the device maximum instead made cuFuncSetAttribute return 1, and
    # setting it per shape lowered the ceiling for larger K launchers already bound.)
    # 4 rows per block (128 threads), matching TI_FP4_ROWS in gemv_fp4_w4a8.cu. The
    # reduction is warp-local, so this is only a packing choice, and 4 nearly doubles the
    # throughput against 8: measured [2304,5120] 25.0 us / 251 GB/s at ROWS=8 versus
    # 14.4 us / 434 GB/s at ROWS=4 (weights rotating over 8 copies to stay out of L2).
    bound = native.bind("gemv_fp4_w4a8", "gemv_fp4_w4a8_e8m0", ((n + 3) // 4,), (128,),
                        "TOOOOiii", shared=smem)
    _E8M0_BOUND[key] = bound
    return bound


def gemv_fp4_pq(aq, asc, w_fp4, w_scale, ws=None):
    """fp8 activation [K] x fp4 weight [N, K/2] -> [1, N] bf16."""
    from ..kernels import native

    n, k2 = w_fp4.shape
    k = k2 * 2
    # e8m0 path: no fp32 scale materialisation, activation staged in shared memory.
    # Bit-identical to the fp32 form (verified maxdiff 0.0) and 1.87x faster on
    # [2304, 5120]: 39.1 us -> 20.9 us, i.e. 300 GB/s against this card's 384.
    if (ws is None and asc.dtype == torch.float8_e8m0fnu
            and w_scale.dtype == torch.float8_e8m0fnu):
        wb0 = w_fp4 if w_fp4.dtype == torch.uint8 else w_fp4.view(torch.uint8)
        out0 = torch.empty(n, dtype=torch.bfloat16, device=aq.device)
        launch = _e8m0_launcher(n, k)
        launch(aq.reshape(-1), asc.view(torch.uint8).reshape(-1), wb0.reshape(-1),
               w_scale.view(torch.uint8).reshape(-1), out0, 1, int(n), int(k),
               stream=None)
        return out0.view(1, n)
    if ws is None:
        ws = _scale(w_scale)
    # No repeat_interleave here: gemv_fp4_pq's caller already hands an [N, K/32] scale
    # (the broadcast form belongs to gemv_fp4(), which takes the packed [N/32, nb] one).
    # Adding it here expanded a scale that was already correct and changed the output.
    asc2 = asc if asc.dtype == torch.float32 else asc.float()
    out = torch.empty(n, dtype=torch.bfloat16, device=aq.device)
    wb = w_fp4 if w_fp4.dtype == torch.uint8 else w_fp4.view(torch.uint8)
    # Same ROWS=4 packing as the e8m0 entry point above (see TI_FP4_ROWS).
    native.launch(native.kernel("gemv_fp4_w4a8", "gemv_fp4_w4a8"),
                  ((n + 3) // 4,), (128,),
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
    # Return the scale as e8m0, exactly like gemv_fp4.quantize_act does. Expanding it to
    # fp32 here looked harmless (the values are identical), but the downstream kernels
    # take different code paths for the two dtypes and the end-to-end decode diverged
    # ([989,43,...] -> [95991,...]). Keep the contract identical instead.
    return q, s


_DIFF_SEEN = [0]


# quantize_act_checked used to live here: it ran the native quantiser and the tilelang
# reference side by side on real activations and reported the first mismatch. That
# comparison is done: over a full decode they agree exactly, so the instrument is gone
# along with the reference it was comparing against.


def available() -> bool:
    try:
        from ..kernels import native

        native.kernel("gemv_fp4_w4a8", "gemv_fp4_w4a8")
        return True
    except Exception:
        return False