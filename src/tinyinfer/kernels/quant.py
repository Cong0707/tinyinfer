"""Block-quantisation entry points backed by our own CUDA kernels.

The reference implementation of act_quant lived in official_kernel_ref (tilelang). That
made tilelang a hard dependency of the whole engine: blocking the import made prefill
raise ImportError, it did not fall back. These wrappers take the same arguments and are
verified bit-exact against the reference (fp8 codes AND the e8m0 scale bytes, 0 diffs over
four shapes including a zeroed row), so they can stand in for it on any CUDA card.

Anything the kernels do not cover (CPU tensors, block sizes we do not compile, an fp32
scale request) falls back to the reference, which keeps the behaviour identical where
these do not apply.
"""
from __future__ import annotations

import torch


def _native_ok() -> bool:
    try:
        from . import native

        native.kernel("act_quant", "act_quant")
        return True
    except Exception:
        return False


def _launch(x: torch.Tensor, out, scale, entry: str) -> None:
    from . import native

    n, k = x.shape
    nwarp = (n * k) // 32
    blk = 128
    grid = (((nwarp * 32) + blk - 1) // blk,)
    native.launch(native.kernel("act_quant", entry), grid, (blk,),
                  # 1e-4 is the reference's floor (official_kernel_ref: T.max(amax, 1e-4));
                  # using 1e-8 changed the decode output on real weights.
                  [x, out, scale, int(n), int(k), 1e-4])


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None,
              scale_dtype: torch.dtype = torch.float32,
              inplace: bool = False):
    """Block-wise fp8 quantisation, same contract as the reference.

    inplace=True returns x itself holding quantised-then-restored bf16 (the KV window
    cache wants exactly that); otherwise returns (fp8 tensor, scale tensor).
    """
    k = int(x.size(-1))
    use_native = (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and block_size == 32
        and k % 32 == 0
        and scale_dtype == torch.float8_e8m0fnu
        and scale_fmt is not None
        and _native_ok()
    )
    if not use_native:
        from ..official_kernel_ref import act_quant as _ref

        return _ref(x, block_size, scale_fmt, scale_dtype, inplace)

    z = x.contiguous()
    n = z.numel() // k
    flat = z.view(n, k)
    s = z.new_empty(*z.shape[:-1], k // 32, dtype=scale_dtype)
    s_flat = s.view(n, k // 32)
    if inplace:
        y = torch.empty_like(flat)
        _launch(flat, y, s_flat, "act_quant_dequant")
        x.copy_(y.view_as(x))
        return x
    y = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=z.device)
    _launch(flat, y, s_flat, "act_quant")
    return y.view(*z.shape), s
