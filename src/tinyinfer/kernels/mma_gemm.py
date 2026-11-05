"""fp8 blockwise GEMM on tensor cores (kernels/cuda/fp8_gemm_mma.cu)."""
from __future__ import annotations

import torch

_ONE = object()


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
    if k % 32 or n % 64:
        raise ValueError(f'fp8_gemm_mma needs K%32==0 and N%64==0, got K={k} N={n}')
    rows = x.size(0) if x.dim() == 2 else 1
    if rows < 128:
        # See the note in gemm.fp8_linear: a block covers 128 rows with 8 warps, and a
        # partial first block only writes its leading warps. Silently returning a tensor
        # with 112 unwritten rows is worse than making the caller take the GEMV path.
        raise ValueError(f'fp8_gemm_mma needs M>=128, got M={rows}')

    xf = x.reshape(-1, k) if x.dim() != 2 else x
    if xf.dtype != torch.bfloat16:
        xf = xf.to(torch.bfloat16)
    aq, asc = act_quant(xf, 32, 'ue8m0', torch.float8_e8m0fnu)

    out = torch.empty(xf.size(0), n, dtype=torch.bfloat16, device=xf.device)
    grid = ((n + 63) // 64, (xf.size(0) + 127) // 128)
    native.launch(native.kernel('fp8_gemm_mma', 'fp8_gemm_mma'), grid, (256,),
                  [aq, asc, w_fp8, w_scale, out, int(xf.size(0)), n, k])
    return out.view(*x.shape[:-1], n)
