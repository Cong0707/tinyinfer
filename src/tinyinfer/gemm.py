"""O3: reuse the official reference tilelang kernels (kernel.py) for quantized GEMM.

Official code is at reference/official/kernel.py (act_quant / fp8_gemm / fp4_gemm).
These are lazy: importing this module does not compile anything; the first call
compiles (needs vcvars64 + nvcc).
"""
import sys
from pathlib import Path

import torch

_REF = Path(__file__).resolve().parents[2] / "reference" / "official"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

try:
    import kernel as _k  # noqa: E402  (reference/official/kernel.py)
except Exception:  # pragma: no cover
    _k = None

# HF's V4.1 fp8 blocks use ue8m0 scales (see official model.py globals)
SCALE_FMT = "ue8m0"
SCALE_DTYPE = torch.float8_e8m0fnu


def available() -> bool:
    return _k is not None


def _bf16_default():
    import contextlib

    @contextlib.contextmanager
    def cm():
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            yield
        finally:
            torch.set_default_dtype(prev)
    return cm()


def fp8_linear(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """C = x @ w_fp8.T with blockwise fp8 activation quant (official kernel path)."""
    with _bf16_default():
        x_fp8, x_s = _k.act_quant(x, block, SCALE_FMT, SCALE_DTYPE)
        return _k.fp8_gemm(x_fp8, x_s, w_fp8, w_scale, SCALE_DTYPE, block_size=block)


def fp4_linear(x: torch.Tensor, w_fp4: torch.Tensor, w_scale: torch.Tensor, act_block: int = 32) -> torch.Tensor:
    """C = x @ w_fp4.T with fp8 activation quant, fp4 weight (official kernel path)."""
    with _bf16_default():
        x_fp8, x_s = _k.act_quant(x, act_block, SCALE_FMT, SCALE_DTYPE)
        return _k.fp4_gemm(x_fp8, x_s, w_fp4, w_scale, SCALE_DTYPE, act_block_size=act_block)
