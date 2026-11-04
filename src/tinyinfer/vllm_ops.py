"""vLLM-derived kernels, ported into native/ and called through the driver API.

These are ports, not calls into vLLM's binary. Each one's source lives in
src/tinyinfer/native/vllm_*.cu with its upstream file and the changes noted at the
top; they are compiled by scripts/build_native.ps1 like every other kernel here and
launched through native.py.

Why they exist: the hand-written versions measured slower than vLLM's on this card
(rms_norm 1.44x off at M=1, 3.53x off at M=8, because it launched one 32-thread
block per row) and vLLM's versions had already been independently reviewed and run
on far more hardware than this box. Porting the device code keeps that work while
leaving the code in this repository, editable, with our argument order.

Fallback order per kernel:
  1. native/vllm_<name>.cu  (ours; requires scripts/build_native.ps1 to have run)
  2. torch.ops._C / _moe_C  (the vendored pyd, if it is still present)
  3. None                   (the caller's own torch implementation)

Porting status: rms_norm, silu_and_mul_with_clamp and topk_softplus_sqrt are done
and verified bit-identical to the pyd. marlin_gemm is not ported yet, so it still
comes from the pyd.
"""
from __future__ import annotations

import torch

from . import native

_OPS = None
_TRIED = False

# Bound launchers, created once per (kernel, geometry) and reused: building the
# ctypes argument array per call costs more than the launch itself.
_BOUND: dict = {}


# Retains the previous call's tensors until the next call: a raw driver launch is
# invisible to torch's caching allocator, so a result tensor dropped right after the
# call can be recycled while the kernel is still writing it. Holding one generation
# back is enough -- by the time the next launch runs, the previous one has retired in
# stream order.
_RETAIN: dict = {}


def _bind(key, cubin, func, grid, block, kinds):
    """Cached native.bind(); returns None when the cubin has not been built."""
    got = _BOUND.get(key)
    if got is not None:
        return got if got is not False else None
    try:
        if not native.available(cubin, func):
            _BOUND[key] = False
            return None
        b = native.bind(cubin, func, grid, block, kinds)
    except Exception:
        _BOUND[key] = False
        return None
    _BOUND[key] = b
    return b


def _call(tag, bound, *args):
    """Launch and keep this call's tensors alive until the same kernel runs again.

    A raw driver launch is invisible to torch's caching allocator: dropping the
    result tensor right after the call lets a later allocation reuse the memory the
    kernel is still writing, which surfaces as CUDACachingAllocator's
    "!handles_.at(i)" assert. One generation of retention per kernel is enough --
    the next launch of that kernel is ordered after the previous one completes.
    """
    _RETAIN[tag] = args
    return bound(*args)


def _ops():
    global _OPS, _TRIED
    if not _TRIED:
        _TRIED = True
        # tinyinfer ships vLLM's compiled extension (see _C_stable_libtorch.pyd next to
        # this file) so the engine does not depend on the vllm Python package being
        # installed. torch must be imported first: its __init__ registers the DLL
        # search directories the extension needs on Windows.
        _OPS = None
        for mod in ("tinyinfer._C_stable_libtorch", "vllm._C_stable_libtorch"):
            try:
                __import__(mod)
                _OPS = torch.ops._C
                break
            except Exception:
                continue
    return _OPS


def available() -> bool:
    return _ops() is not None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor | None:
    """Match nn.rms_norm's contract: fp32 statistics, scale by weight, keep x's dtype."""
    if not x.is_cuda or weight is None:
        return None
    if weight.numel() != x.shape[-1]:
        return None
    if x.dtype not in (torch.bfloat16, torch.float16):
        return None
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    out = torch.empty_like(flat)
    # our port first: native/vllm_rms_norm.cu
    if x.dtype == torch.bfloat16:
        rows = flat.size(0)
        hidden = flat.size(-1)
        vec = min(8, hidden)
        if hidden % vec == 0:
            block = min(hidden // vec, 1024 if rows < 256 else 256)
            # (out, input, weight, hidden, eps): 'O' marks the one buffer we write
            b = _bind(("rms", rows, block), "vllm_rms_norm", "vllm_rms_norm_bf16",
                      (rows,), (block,), "OTTif")
            if b is not None:
                try:
                    _call("rms", b, out, flat.contiguous(), weight.contiguous(),
                          hidden, float(eps))
                    return out.view(*shape)
                except Exception:
                    pass
    ops = _ops()
    if ops is None:
        return None
    try:
        ops.rms_norm(out, flat.contiguous(), weight.contiguous(), float(eps))
    except Exception:
        return None
    return out.view(*shape)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor | None:
    """x [..., 2N] -> silu(x[..., :N]) * x[..., N:]."""
    ops = _ops()
    if ops is None or not x.is_cuda:
        return None
    if x.dtype not in (torch.bfloat16, torch.float16):
        return None
    half = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], half, dtype=x.dtype, device=x.device)
    try:
        ops.silu_and_mul(out, x.contiguous())
    except Exception:
        return None
    return out


def clamp_swiglu(x: torch.Tensor, limit: float) -> torch.Tensor | None:
    """DeepSeek's SwiGLU: silu(clamp(gate, max=limit)) * clamp(up, +-limit).

    x is [..., 2*d] with the gate half first. vLLM's act_first branch clamps gate with
    fminf(g, limit) only and folds up into [-limit, limit], which is exactly the shape
    this model's swiglu_limit asks for -- verified line by line against
    activation_kernels.cu:64-75.
    """
    if not x.is_cuda or limit <= 0:
        return None
    if x.dtype not in (torch.bfloat16, torch.float16):
        return None
    half = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], half, dtype=x.dtype, device=x.device)
    # our port first: native/vllm_silu_and_mul.cu
    if x.dtype == torch.bfloat16:
        rows = x.numel() // x.shape[-1]
        vec = min(8, half)
        if half % vec == 0:
            block = min(half // vec, 1024)
            # (out, input, d, limit)
            b = _bind(("silu", rows, block), "vllm_silu_and_mul",
                      "vllm_silu_and_mul_clamp_bf16", (rows,), (block,), "OTif")
            if b is not None:
                try:
                    _call("silu", b, out, x.contiguous(), half, float(limit))
                    return out
                except Exception:
                    pass
    ops = _ops()
    if ops is None:
        return None
    try:
        ops.silu_and_mul_with_clamp(out, x.contiguous(), float(limit))
    except Exception:
        return None
    return out


_MOE = None


def _moe_ops():
    """torch.ops._moe_C, which is a separate extension from _C."""
    global _MOE
    if _MOE is None:
        # Importing the extension a second time aborts its TORCH_LIBRARY registration
        # ("Duplicate registration" across the libtorch C API boundary) and leaves the
        # namespace present but EMPTY -- every op then reports AttributeError. Reuse a
        # namespace that already advertises ops instead of importing again.
        try:
            seen = torch.ops._moe_C
            if any(not n.startswith("_") for n in dir(seen)):
                _MOE = seen
                return _MOE
        except Exception:
            pass
        for mod in ("tinyinfer._moe_C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
            try:
                __import__(mod)
                _MOE = torch.ops._moe_C
                break
            except Exception:
                continue
    return _MOE


def gate_topk(
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    scaling: float,
    bias: torch.Tensor | None = None,
):
    """DeepSeek's router in one kernel: softplus(x).sqrt() + bias + top-k + renormalize.

    gating is the raw gate logits [T, E]. vLLM's kernel applies the sqrtsoftplus
    scoring, adds e_score_correction_bias, takes the top-k and folds the routing
    scale in, so the engine drops the argsort + gather + normalise sequence. The
    top-k tie order is the sorted-expert order the kernel uses, which is what this
    model needs; the previous torch path had to force a stable argsort for the same
    reason. Returns (weights f32 [T, topk], ids int32 [T, topk]) or None.
    """
    if not gating.is_cuda or gating.dim() != 2:
        return None
    n_tok, n_expert = gating.shape
    if bias is not None and bias.numel() != n_expert:
        return None
    w = torch.empty(n_tok, topk, dtype=torch.float32, device=gating.device)
    idx = torch.empty(n_tok, topk, dtype=torch.int32, device=gating.device)
    tei = torch.empty(n_tok, topk, dtype=torch.int32, device=gating.device)
    # our port first: native/vllm_topk_softplus_sqrt.cu (E=384, k=6 only for now)
    if n_expert == 384 and topk == 6 and gating.dtype == torch.float32:
        block = 32 * min(32, max(1, n_tok))
        grid = (n_tok * 32 + block - 1) // block
        # (in, weights, indices, source_rows, bias, num_rows, renormalize, scaling);
        # the three outputs are written and must be recorded on the stream
        b = _bind(("topk", n_tok, block), "vllm_topk_softplus_sqrt",
                  "vllm_topk_softplus_sqrt_e384_k6", (grid,), (block,), "TOOOTiif")
        if b is not None:
            try:
                _call("topk", b, gating.contiguous().float(), w, idx, tei,
                      bias.contiguous().float() if bias is not None else None,
                      n_tok, 1 if renormalize else 0, float(scaling))
                return w, idx
            except Exception:
                pass
    moe = _moe_ops()
    if moe is None:
        return None
    try:
        moe.topk_softplus_sqrt(
            w, idx, tei, gating.contiguous().float(),
            bool(renormalize), float(scaling),
            bias.contiguous().float() if bias is not None else None,
            None, None, None, None, 0,
        )
    except Exception:
        return None
    return w, idx
