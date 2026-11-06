"""fp8 blockwise GEMM on tensor cores (kernels/cuda/fp8_gemm_mma.cu)."""
from __future__ import annotations

import torch

_ONE = object()
_BOUND: dict = {}


def available() -> bool:
    try:
        from . import native

        native.kernel('fp8_gemm_mma', 'fp8_gemm_mma')
        return True
    except Exception:
        return False


def fp8_gemm_mma(x, w_fp8, w_scale, block: int = 32):
    """C = x @ w_fp8.T with 1x32x32 blockwise scales, on mma.sync tensor cores."""
    from . import native
    from .quant import act_quant

    k = int(x.size(-1))
    n = int(w_fp8.size(0))
    if k % 32 or n % 8:
        raise ValueError(f'fp8_gemm_mma needs K%32==0 and N%8==0, got K={k} N={n}')
    rows = x.size(0) if x.dim() == 2 else 1
    if rows < 64:
        # A block covers BLOCK_M=64 rows with 4 warps; each warp writes its own 16 rows
        # with bounds checks, so a partial last block is fine but a block whose leading
        # warps are the only ones alive would leave rows unwritten. Below one block, take
        # the GEMV path instead.
        raise ValueError(f'fp8_gemm_mma needs M>=64, got M={rows}')

    xf = x.reshape(-1, k) if x.dim() != 2 else x
    if xf.dtype != torch.bfloat16:
        xf = xf.to(torch.bfloat16)
    aq, asc = act_quant(xf, 32, 'ue8m0', torch.float8_e8m0fnu)

    out = torch.empty(xf.size(0), n, dtype=torch.bfloat16, device=xf.device)
    # These MUST match the constants in kernels/cuda/fp8_gemm_mma.cu: a block covers
    # BLOCK_M rows with (BLOCK_M/16) warps, and each warp covers WARP_N*NB columns. When
    # they drift the launch simply does less work than asked -- which reads as a large
    # speedup and a NaN result, exactly what happened when NB changed from 8 to 1 here
    # without this grid following it.
    _BLOCK_M, _WARP_M, _WARP_N, _NB = 128, 16, 8, 4
    _STAGES = 4
    _NWARP = _BLOCK_M // _WARP_M
    _BCOL = _WARP_N * _NB
    grid = ((n + _BCOL - 1) // _BCOL, (xf.size(0) + _BLOCK_M - 1) // _BLOCK_M)
    threads = _NWARP * 32
    # Dynamic shared memory: NWARP*STAGES warp-private A tiles (16*32 B each) followed by
    # STAGES B tiles (NCOL*32 B each) = 8*4*512 + 4*1024 = 20 KB at NCOL=32. Constant per
    # launch, so bind once per geometry instead of re-marshalling arguments every call.
    _smem = _NWARP * _STAGES * (_WARP_M * 32) + _STAGES * (_BCOL * 32)
    _key = ('fp8_gemm_mma', int(n), int(k), int(xf.size(0)))
    _fn = _BOUND.get(_key)
    if _fn is None:
        _fn = native.BoundLaunch(
            native.kernel('fp8_gemm_mma', 'fp8_gemm_mma'), grid, (threads,),
            "TTTTOiii", shared=_smem)
        _BOUND[_key] = _fn
    _fn(aq, asc, w_fp8, w_scale, out, int(xf.size(0)), n, k, stream=None)
    return out.view(*x.shape[:-1], n)
