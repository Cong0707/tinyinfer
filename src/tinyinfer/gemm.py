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

# Block quantisation has a native implementation (kernels/cuda/act_quant.cu) that is
# bit-identical to the reference's; it is used for every act_quant call below so the
# decode path does not need tilelang's JIT. _k is still what provides fp8_gemm/fp4_gemm
# for the M>1 fallback (prefill), so tilelang remains required for prefill only.
from .kernels.quant import act_quant as _act_quant  # noqa: E402

# HF's V4.1 fp8 blocks use ue8m0 scales (see official model.py globals)
SCALE_FMT = "ue8m0"
SCALE_DTYPE = torch.float8_e8m0fnu


def available() -> bool:
    """True when the fp8 kernels can run.

    NOT "tilelang imported". fp8_linear's decode path (M=1) is the native fp8_gemv_bs
    kernel and needs no tilelang at all; gating on _k made every QuantLinear fall back to
    dequantise-then-matmul, which is a DIFFERENT computation -- measured: a run with
    tilelang blocked produced [2058, 3] where the fp8 path produces [989, 43].
    tilelang is still needed for the M>1 (prefill) branch, and that is the branch that
    falls back to _k.fp8_gemm.
    """
    if _k is not None:
        return True
    try:
        from .kernels import native

        native.kernel("fp8_gemv_bs", "fp8_gemv_bs")
        return True
    except Exception:
        return False


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


def _fp8_native() -> bool:
    """True when e4m3 conversion is one hardware instruction (SM 89 and up).

    Probed once and cached: the classification never changes for a process, and asking
    torch on every linear layer would put a property lookup on the decode path.
    """
    global _FP8_NATIVE
    if _FP8_NATIVE is None:
        try:
            from .caps import caps

            _FP8_NATIVE = bool(caps().fp8_native)
        except Exception:
            _FP8_NATIVE = True          # unknown: keep the fast path, it is correct anyway
    return _FP8_NATIVE


_FP8_NATIVE = None


def fp8_linear(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    # Cast BEFORE the fast path decides anything. act_quant only has a native
    # kernel for bf16 input; handing it fp32 falls back to the tilelang one, which
    # raises ImportError on a deployment without tilelang and looks exactly like
    # "the fast path is broken". The cast used to sit below the branch.
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if (x.is_cuda and x.dim() == 2 and x.size(0) == 1
            # K need not be a multiple of 512: the kernel rounds its warp iterations up
            # and handles a short tail. That matters because two of the largest
            # projections are NOT 512-aligned -- wq_b is K=1280 (42 MB) and the shared
            # expert's w2 is K=2304 -- and they used to fall through to the block_M=32
            # GEMM, which is the slowest thing left in the decode step.
            # K must be a multiple of 512. The kernel itself handles other K correctly
            # (wq_b at K=1280: 124us vs 253us, 1 element out of 32768 off by one bf16
            # ulp; w2 at K=2304: bit-identical), but enabling it here makes the TILELANG
            # kernels that still run start throwing: profile shows 90 calls into
            # tvm_ffi/error.py:_parse_backtrace and 303 re-resolutions of the executable,
            # 79ms each, turning a 0.8s step into 10.7s. Something about the fast path's
            # allocations destabilises the tilelang side; not diagnosed yet. Keep the gate
            # closed until it is.
            and block == 32
            # Every K that is a multiple of 16 now takes the native GEMV: the tilelang
            # GEMM is gone from this path. K not a multiple of 512 differs from the
            # tilelang result by one bf16 ulp on a handful of outputs (fp32 accumulation
            # order), which used to change decoded tokens after 40 layers -- that was
            # measured against tilelang. Now that M>=2 goes through our own mma kernel,
            # both paths have the same accumulation semantics and there is no third
            # reference to disagree with.
            and w_fp8.shape[1] % 16 == 0
            and w_fp8.shape[0] % 16 == 0
            # fp8_gemv_bs decodes e4m3 with the hardware converter. On a card without
            # one (SM < 89, i.e. Turing and Ampere) that call becomes a software
            # sequence per element and this kernel loses to the plain GEMM it replaces,
            # so the gate is the capability, not the shape.
            and _fp8_native()):
        # kernels/cuda/fp8_gemv_bs.cu: 16 output rows x 32 lanes per block, one LDG.E.128
        # per lane, so a warp reads 512 consecutive k. Measured 25.2 us vs 45.5 for the
        # block_M=32 tilelang path on [2304, 5120] (1.8x, bit-identical). This is the
        # shared-expert path, ~248 calls per decode token.
        try:
            from .kernels import native

            aq, asc = _act_quant(x, block, SCALE_FMT, SCALE_DTYPE)
            n, k = w_fp8.shape
            y = torch.empty(n, device=x.device, dtype=torch.bfloat16)
            native.launch(native.kernel("fp8_gemv_bs", "fp8_gemv_bs"),
                          ((n + 15) // 16,), (512,),
                          [aq.view(torch.uint8).reshape(-1),
                           asc.view(torch.uint8).reshape(-1),
                           w_fp8.view(torch.uint8).reshape(-1),
                           w_scale.view(torch.uint8).reshape(-1),
                           y, int(n), int(k)])
            return y.view(1, n)
        except Exception as _exc:
            if __import__("os").environ.get("TI_FP8_DEBUG") == "1":
                import traceback as _tb

                print(f"[fp8-fast-fail] {_exc!r}", flush=True)
                _tb.print_exc()
            pass

    """C = x @ w_fp8.T with blockwise fp8 activation quant (official kernel path).

    decode (M==1) dispatches to the fused GEMV kernel (~1.8x faster than the
    block_M=32 GEMM, which wastes 31/32 of the tile); prefill keeps the GEMM.
    """
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
                    x_fp8, x_s = _act_quant(x, block, SCALE_FMT, SCALE_DTYPE)
                out = gemv_fp8_block(x_fp8, w_fp8, x_s.float().reshape(1, -1),
                                     w_scale.float(), block)
                if out is not None:
                    return out
            except Exception:
                pass
        try:
            # The name this used to import, gemv_fp8, has never existed in that module
            # (it exports fused_fp8_gemv / fused_fp8_gemv_pq). The ImportError raised on
            # every call was swallowed here, so the whole M=1 fast path was dead: all 248
            # fp8 linears per decode token fell through to the block_M=32 tilelang GEMM at
            # ~96 GB/s while this kernel reaches ~160.
            #
            # Fixed the name, but turning it on makes the engine die with
            # "CUDA error: misaligned address" a few layers later (gemv_fp8.py's own
            # docstring warns the bf16 variant crashes at N=1280/K=5120; shared expert is
            # N=2304/K=5120 and it faults there too, asynchronously inside hc_post).
            # Forcing a 16-byte-aligned copy of x did not fix it. Off until that kernel's
            # alignment assumptions are understood: TI_FP8_GEMV=1 enables it.
            import os as _os_g8

            if _os_g8.environ.get("TI_FP8_GEMV") == "1":
                from .gemv_fp8 import fused_fp8_gemv

                return fused_fp8_gemv(x, w_fp8, w_scale)
        except Exception:
            pass
    # The tensor-core kernel is faster than a row loop once M is large enough for the
    # 16-row mma tile to fill: at M=1 it was measured 4.2x (wq_b) to 6.5x (w1) SLOWER
    # than the SIMT GEMV, because 15/16 of the tile is waste. The threshold below is
    # where that inverts, not a guess about "batch".
    #
    # It is off by default right now for a different reason: a prompt's prefill and its
    # per-token decode must agree, and with prefill on mma and decode on SIMT the two
    # paths differ by one bf16 ulp, which is enough to change decoded tokens over 40
    # layers ([989,43,4571,5958] -> [223,19,17,20]). Using one path for both is the only
    # way to keep them consistent until the two kernels are proven equivalent.
    if (x.is_cuda and x.dim() == 2 and x.size(0) >= 128 and block == 32
            and __import__("os").environ.get("TI_MMA_GEMM") == "1"):
        try:
            from .kernels import mma_gemm

            return mma_gemm.fp8_gemm_mma(x, w_fp8, w_scale)
        except Exception:
            pass
    # M in 2..8 used to go to the tilelang fused_fp8_gemv_pq here. It is off because it
    # is a DIFFERENT kernel from the native GEMV that M=1 uses, and a prompt's prefill
    # and its decode have to agree: with prefill on that path the decoded tokens changed
    # ([989,43,4571,5958] -> [223,19,17,20]). Those rows now fall through to the row
    # loop below, which is the same kernel as M=1 -- fewer launches would be nicer, but
    # identical arithmetic matters more. TI_FP8_PQ=1 restores the old path.
    if (x.dim() == 2 and 1 < x.size(0) <= 8 and x.is_cuda
            and __import__("os").environ.get("TI_FP8_PQ") == "1"):
        try:
            from .gemv_fp8 import fused_fp8_gemv_pq
            with _bf16_default():
                x_fp8, x_s = _act_quant(x, block, SCALE_FMT, SCALE_DTYPE)
            outs = []
            for i in range(x.size(0)):
                outs.append(fused_fp8_gemv_pq(
                    x_fp8[i].reshape(-1), x_s[i].reshape(-1), w_fp8, w_scale).reshape(1, -1))
            return torch.cat(outs, dim=0)
        except Exception:
            pass
    if x.is_cuda:
        # Prefill runs the SAME kernel as decode, one row at a time. That is what keeps a
        # prompt's prefill and its decode consistent: routing prefill through tilelang's
        # fp8_gemm while decode used the native GEMV made the two differ by one bf16 ulp,
        # and after 40 layers the decoded tokens changed
        # ([989,43,4571,5958] -> [223,19,17,20]). Slower per prompt, but a prefill happens
        # once and a wrong KV cache poisons every token after it.
        #
        # TI_PREFILL_TL=1 routes it back through tilelang's block-M GEMM for comparison.
        if __import__("os").environ.get("TI_PREFILL_TL") != "1":
            xf = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
            if xf.size(0) > 1:
                rows = [fp8_linear(xf[i:i + 1], w_fp8, w_scale, block)
                        for i in range(xf.size(0))]
                out = torch.cat(rows, dim=0)
                return out.view(*x.shape[:-1], out.shape[-1])
    if _k is not None:
        with _bf16_default():
            x_fp8, x_s = _act_quant(x, block, SCALE_FMT, SCALE_DTYPE)
            return _k.fp8_gemm(x_fp8, x_s, w_fp8, w_scale, SCALE_DTYPE, block_size=block)
    # No tilelang, or the caller's shape fell through the branches above: the only kernel
    # left is the native M=1 one, so run it row by row. Same numbers as the block-M GEMM
    # (it is the same maths, one row at a time), just slower.
    xf = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
    if xf.size(0) == 1:
        # A single row got past every fp8 branch above: either the weight shape fails the
        # %16 checks, the card has no native e4m3 converter, or x is not CUDA. Falling
        # back to dequantise-then-matmul silently would produce different numbers than the
        # fp8 path, so say which it is instead of recursing forever.
        raise RuntimeError(
            f"fp8_linear: no path for M=1 (is_cuda={x.is_cuda} dim={x.dim()} "
            f"M={x.size(0)} block={block} w={tuple(w_fp8.shape)} "
            f"K%16={(w_fp8.shape[1] % 16)} N%16={(w_fp8.shape[0] % 16)} "
            f"native={_fp8_native()} k={_k is not None} "
            f"xdt={x.dtype} wdt={w_fp8.dtype})")
    rows = [fp8_linear(xf[i:i + 1], w_fp8, w_scale, block) for i in range(xf.size(0))]
    out = torch.cat(rows, dim=0)
    return out.view(*x.shape[:-1], out.shape[-1])


def fp4_linear(x: torch.Tensor, w_fp4: torch.Tensor, w_scale: torch.Tensor, act_block: int = 32) -> torch.Tensor:
    """C = x @ w_fp4.T with fp8 activation quant, fp4 weight (official kernel path)."""
    # vLLM's Marlin streams the expert weights at 284 GB/s on this card where the
    # hand-written GEMV manages 59 -- it is the same matrix, decoded in ldmatrix
    # fragments off a four-stage cp.async pipeline instead of by scalar loads. Route
    # the small-M case there and keep the fp8 path as the fallback.
    if x.is_cuda and x.dim() == 2 and 0 < x.size(0) <= 8:
        try:
            from .deepseek.gemv_fp4_marlin import gemv_fp4_fast as _marlin_gemv

            return _marlin_gemv(x, w_fp4, w_scale)
        except Exception:
            pass
    if x.dim() == 2 and 0 < x.size(0) <= 8 and x.is_cuda:
        # same block_M=32 waste as fp8: at M=2 the GEMM took 334ms/layer. Rows are done
        # one at a time with the single-row kernel: gemv_fp4_m8's multi-row grid was
        # measured 35% off on some (weight, row) pairs, and a chunk only has a handful
        # of rows per expert, so exactness wins over the launch count.
        try:
            from .deepseek.gemv_fp4 import gemv_fp4_m8, gemv_fp4_pq, quantize_act
            aq, asc = quantize_act(x, x.size(-1))
            if aq.size(0) > 1 and __import__("os").environ.get("TI_MULTIROW_GEMV") == "1":
                return gemv_fp4_m8(aq, asc, w_fp4, w_scale)
            if aq.size(0) > 1:
                out = [gemv_fp4_pq(aq[i], asc[i], w_fp4, w_scale) for i in range(aq.size(0))]
                return torch.cat(out, 0)
            return gemv_fp4_pq(aq[0], asc[0], w_fp4, w_scale)
        except Exception:
            try:
                from .deepseek.gemv_fp4 import gemv_fp4
                rows = [gemv_fp4(x[i:i + 1], w_fp4, w_scale) for i in range(x.size(0))]
                return rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)
            except Exception:
                pass
    with _bf16_default():
        x_fp8, x_s = _act_quant(x, act_block, SCALE_FMT, SCALE_DTYPE)
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
