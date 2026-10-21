"""O3: reuse the official reference tilelang kernels (kernel.py) for quantized GEMM.

Reference implementations (consult when fixing this file):
  * Official   reference/official/kernel.py (act_quant / w8a8_block_fp8_linear / fp4_gemm)
  * DeepInfer  deepinfer/model_executor/layers/quantization (fp8/fp4 linear, ue8m0 scales)
  * PyTorch    torch._scaled_mm (cuBLAS fp8) and plain F.linear fallbacks

Tolerance: fp8 block GEMM rel <= 1e-2, fp4 rel <= 3e-2 vs bf16 reference.
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
    """C = x @ w_fp8.T with blockwise fp8 activation quant (official kernel path).

    decode (M==1) dispatches to the fused GEMV kernel (~1.8x faster than the
    block_M=32 GEMM, which wastes 31/32 of the tile); prefill keeps the GEMM.
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    # decode (M==1): single fused GEMV kernel instead of the block_M=32 GEMM
    if x.dim() == 2 and x.size(0) == 1 and x.is_cuda:
        try:
            from .gemv_fp8 import gemv_fp8
            return gemv_fp8(x, w_fp8, w_scale)
        except Exception:
            pass
    with _bf16_default():
        x_fp8, x_s = _k.act_quant(x, block, SCALE_FMT, SCALE_DTYPE)
        return _k.fp8_gemm(x_fp8, x_s, w_fp8, w_scale, SCALE_DTYPE, block_size=block)


def fp4_linear(x: torch.Tensor, w_fp4: torch.Tensor, w_scale: torch.Tensor, act_block: int = 32) -> torch.Tensor:
    """C = x @ w_fp4.T with fp8 activation quant, fp4 weight (official kernel path)."""
    with _bf16_default():
        x_fp8, x_s = _k.act_quant(x, act_block, SCALE_FMT, SCALE_DTYPE)
        return _k.fp4_gemm(x_fp8, x_s, w_fp4, w_scale, SCALE_DTYPE, act_block_size=act_block)
