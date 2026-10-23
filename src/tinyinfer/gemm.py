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
    # decode (M==1): the fused GEMV kernel when it is available. NOTE: looping
    # fused_fp8_gemv for M in 2..8 crashes (misaligned address inside
    # gemv_bf16_kernel at N=1280/K=5120), so small-M fp8 stays on the GEMM for now.
    if x.dim() == 2 and x.size(0) == 1 and x.is_cuda:
        # Official block-scaled GEMV (b12x algorithm ported to TileLang): 16 output
        # features per CTA, 8 warps split K by whole scale blocks, FP32 block-scaled
        # accumulate, deterministic warp reduction. Enabled by the tier; if the port
        # does not cover the shape it returns None and the old kernel runs.
        # Measured crossover on SM120 (probe_official_gemv.py): below ~4096 output
        # features the engine's per-row kernel wins (it owns 16 features per CTA and
        # never pads), from 4096 up the ported 128-feature/CTA kernel wins by 12-25x
        # (wq_b at N=32768, wo_b at 5120, the vocabulary head).
        if w_fp8.shape[0] >= 4096:
            try:
                from .official_kernels import gemv_fp8_block

                with _bf16_default():
                    x_fp8, x_s = _k.act_quant(x, block, SCALE_FMT, SCALE_DTYPE)
                out = gemv_fp8_block(x_fp8, w_fp8, x_s.float().reshape(1, -1),
                                     w_scale.float(), block)
                if out is not None:
                    return out
            except Exception:
                pass
        try:
            from .gemv_fp8 import gemv_fp8
            return gemv_fp8(x, w_fp8, w_scale)
        except Exception:
            pass
    # M in 2..8: pre-quantise once, then one gemv_pq per row. The bf16-input gemv
    # variant crashes (misaligned address at N=1280/K=5120), the pq variant does not.
    if x.dim() == 2 and 1 < x.size(0) <= 8 and x.is_cuda:
        try:
            from .gemv_fp8 import fused_fp8_gemv_pq
            with _bf16_default():
                x_fp8, x_s = _k.act_quant(x, block, SCALE_FMT, SCALE_DTYPE)
            outs = []
            for i in range(x.size(0)):
                outs.append(fused_fp8_gemv_pq(
                    x_fp8[i].reshape(-1), x_s[i].reshape(-1), w_fp8, w_scale).reshape(1, -1))
            return torch.cat(outs, dim=0)
        except Exception:
            pass
    with _bf16_default():
        x_fp8, x_s = _k.act_quant(x, block, SCALE_FMT, SCALE_DTYPE)
        return _k.fp8_gemm(x_fp8, x_s, w_fp8, w_scale, SCALE_DTYPE, block_size=block)


def fp4_linear(x: torch.Tensor, w_fp4: torch.Tensor, w_scale: torch.Tensor, act_block: int = 32) -> torch.Tensor:
    """C = x @ w_fp4.T with fp8 activation quant, fp4 weight (official kernel path)."""
    if x.dim() == 2 and 0 < x.size(0) <= 8 and x.is_cuda:
        # same block_M=32 waste as fp8: at M=2 the GEMM took 334ms/layer. Rows are done
        # one at a time with the single-row kernel: gemv_fp4_m8's multi-row grid was
        # measured 35% off on some (weight, row) pairs, and a chunk only has a handful
        # of rows per expert, so exactness wins over the launch count.
        try:
            from .gemv_fp4 import gemv_fp4_m8, gemv_fp4_pq, quantize_act
            aq, asc = quantize_act(x, x.size(-1))
            if aq.size(0) > 1 and __import__("os").environ.get("TI_MULTIROW_GEMV") == "1":
                return gemv_fp4_m8(aq, asc, w_fp4, w_scale)
            if aq.size(0) > 1:
                out = [gemv_fp4_pq(aq[i], asc[i], w_fp4, w_scale) for i in range(aq.size(0))]
                return torch.cat(out, 0)
            return gemv_fp4_pq(aq[0], asc[0], w_fp4, w_scale)
        except Exception:
            try:
                from .gemv_fp4 import gemv_fp4
                rows = [gemv_fp4(x[i:i + 1], w_fp4, w_scale) for i in range(x.size(0))]
                return rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)
            except Exception:
                pass
    with _bf16_default():
        x_fp8, x_s = _k.act_quant(x, act_block, SCALE_FMT, SCALE_DTYPE)
        out = _k.fp4_gemm(x_fp8, x_s, w_fp4, w_scale, SCALE_DTYPE, act_block_size=act_block)
    # same story as gemv_fp4._keep: the tilelang GEMM launches itself, so the allocator
    # has no record of the expert weights it is still reading
    if x_fp8.is_cuda:
        try:
            _cur = torch.cuda.current_stream()
            for _t in (x_fp8, x_s, w_fp4, w_scale, out):
                _t.record_stream(_cur)
        except Exception:
            pass
    return out