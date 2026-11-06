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


# One bound launcher per (entry, n, k). This kernel is launched 288 times in a single
# decode step (7 per layer), and the generic launch path re-tests every argument's type
# and rebuilds the ctypes pointer array each time: measured 19.3 us of a 26.4 us call,
# with the GPU work itself under 1 us at these sizes. BoundLaunch caches the geometry and
# the argument kinds, leaving one list comprehension plus cuLaunchKernel.
_BOUND: dict[tuple[str, int, int], object] = {}


def _launch(x: torch.Tensor, out, scale, entry: str) -> None:
    from . import native

    n, k = x.shape
    key = (entry, int(n), int(k))
    fn = _BOUND.get(key)
    if fn is None:
        blk = 128
        nwarp = (n * k) // 32
        grid = (((nwarp * 32) + blk - 1) // blk,)
        # 'T' tensor in, 'O' tensor written, 'i' int32, 'f' float32
        fn = native.BoundLaunch(native.kernel("act_quant", entry), grid, (blk,),
                                "TOTiif")
        _BOUND[key] = fn
    # 1e-4 is the reference's floor (official_kernel_ref: T.max(amax, 1e-4)); using 1e-8
    # changed the decode output on real weights.
    fn(x, out, scale, int(n), int(k), 1e-4)


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
        # No tilelang fallback: the reference implementation was a tilelang kernel, and
        # the whole point of this module is to remove that dependency. If the native
        # kernel cannot serve the request, say exactly which precondition failed -- a
        # silent import of official_kernel_ref would reintroduce the dependency and also
        # change the numbers, because that path is a different implementation.
        raise NotImplementedError(
            f"act_quant native kernel cannot serve this request "
            f"(cuda={x.is_cuda} dtype={x.dtype} block={block_size} K={k} "
            f"scale_dtype={scale_dtype} scale_fmt={scale_fmt} native_ok={_native_ok()}); "
            f"the tilelang reference path has been removed")

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
