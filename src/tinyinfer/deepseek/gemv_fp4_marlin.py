"""fp4 expert GEMV through vLLM's Marlin kernel.

The hand-written ALU-decode kernel reaches 59 GB/s on a 40MB expert matrix; the
tilelang one 40 GB/s. vLLM's Marlin measures 294 GB/s on the same buffer --
90%+ of this card's pure-read ceiling -- because it streams the weights through a
four-stage cp.async pipeline and decodes them in ldmatrix-sized fragments instead
of scalar loads.

So we do not hand-write this. The weights are repacked once at load time into the
layout Marlin wants (gptq_marlin_repack) and the e8m0 scales are permuted into the
block order its dequantiser reads; after that every call is a single fused kernel.
"""
from __future__ import annotations

import torch

# vllm::kFE2M1f -- the E2M1 four-bit float id, from scalar_types.float4_e2m1f.id
FP4_TYPE_ID = 562949953487106

_ops_mod = None


def _ops():
    """Import the vendored vLLM extension lazily so the engine still runs without it."""
    global _ops_mod
    if _ops_mod is None:
        last = None
        for mod in ("tinyinfer._C_stable_libtorch", "vllm._C_stable_libtorch"):
            try:
                __import__(mod)  # registers torch.ops._C
                _ops_mod = torch.ops._C
                break
            except Exception as exc:  # pragma: no cover - depends on the install
                last = exc
        if _ops_mod is None:
            raise last if last is not None else RuntimeError("vLLM extension unavailable")
    return _ops_mod


def available() -> bool:
    try:
        _ops()
        return True
    except Exception:
        return False


def _scale_perms():
    perm, perm_single = [], []
    for i in range(8):
        perm.extend([i + 8 * j for j in range(8)])
    for i in range(4):
        perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, perm_single


def _permute_scales(s: torch.Tensor, size_k: int, size_n: int, group_size: int) -> torch.Tensor:
    perm, perm_single = _scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(perm)))[:, perm]
    else:
        s = s.reshape((-1, len(perm_single)))[:, perm_single]
    return s.reshape((-1, size_n)).contiguous()


def _mxfp4_scales(s: torch.Tensor) -> torch.Tensor:
    # Marlin's fp8 dequantiser wants the four scales of a 128-weight block interleaved
    # as 0,2,1,3, then stored as a bare exponent (S0E8M0) so a shift replaces the
    # multiply.
    s = s.view(-1, 4)[:, [0, 2, 1, 3]].view(s.size(0), -1)
    return s.to(torch.float8_e8m0fnu)


_PREP = None


def _prep_scales(e8m0: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    """[N, K/32] e8m0 -> Marlin's [K/32, N] permuted grid, in one launch."""
    global _PREP
    if _PREP is None:
        try:
            from ..kernels import native

            k32 = size_k // 32
            grid = (((size_n * k32) + 255) // 256,)
            _PREP = native.bind("marlin_prep_scales", "marlin_prep_scales", grid, (256,),
                                "TTii")
        except Exception:
            _PREP = False
    k32 = size_k // 32
    if _PREP:
        out = torch.empty((k32, size_n), dtype=torch.uint8, device=e8m0.device)
        _PREP(e8m0.view(torch.uint8), out, int(size_n), int(k32), stream=None)
        return out.view(torch.float8_e8m0fnu)
    # fallback: the three torch passes this kernel replaces
    return _mxfp4_scales(_permute_scales(e8m0.T.contiguous(), size_k, size_n, 32))


# Repack kernel symbol for the only configuration this engine uses: 4-bit weights on
# 256 threads with 16-bit activations. Same mangled name as the reference build, because
# it is the same template instantiated from our copy of the sources (native/marlin).
_REPACK_SYMBOL = "_ZN6marlin25gptq_marlin_repack_kernelILi256ELi4ELb0EEEvPKjPjii"
_REPACK = None


def _repack_launcher():
    """cubin-backed repack, or None when it was not built."""
    global _REPACK
    if _REPACK is not None:
        return _REPACK or None
    try:
        from ..kernels import native

        props = torch.cuda.get_device_properties(0)
        sms = int(props.multi_processor_count)
        smem = int(props.shared_memory_per_block_optin)
        native.set_dynamic_smem("marlin_repack", _REPACK_SYMBOL, smem)
        # The kernel is (b_q_weight, out, size_k, size_n): FOUR arguments. This used to
        # declare "Oii" (out + two ints, three total) and every call raised TypeError
        # "expected 3 args, got 4", which the caller caught and silently fell back to
        # ops.gptq_marlin_repack -- so the native repack never ran and the pyd was still
        # a build-time dependency on every expert load.
        _REPACK = (native.bind("marlin_repack", _REPACK_SYMBOL, (sms,), (256,),
                               "OOii", shared=smem), sms)
    except Exception:
        _REPACK = False
    return _REPACK or None


def _repack(src: torch.Tensor, size_k: int, size_n: int, ops) -> torch.Tensor:
    """GPTQ-packed int4 -> the interleaved layout the Marlin GEMV reads."""
    got = _repack_launcher()
    if got is not None:
        launch, _sms = got
        out = torch.empty(size_k // 16, size_n * 16 // 8, dtype=torch.int32,
                          device=src.device)
        # kernel signature is (b_q_weight, out, size_k, size_n); 4 int4 tiles per K row
        launch(src, out, size_k, size_n)
        return out
    return ops.gptq_marlin_repack(src, size_k, size_n, 4, False)


class MarlinFp4Weight:
    """A repacked fp4 expert matrix, ready to multiply against an activation row."""

    __slots__ = ("qweight", "scales", "size_n", "size_k")

    def __init__(self, w_fp4: torch.Tensor, w_scale: torch.Tensor):
        size_n, k2 = w_fp4.shape
        size_k = k2 * 2
        ops = _ops()
        packed = w_fp4.reshape(size_n, k2).contiguous()
        if packed.dtype != torch.uint8:
            packed = packed.view(torch.uint8)
        # Marlin stores K-major; the repack both transposes and interleaves.
        src = packed.view(torch.int32).T.contiguous()
        self.qweight = _repack(src, size_k, size_n, ops)
        e8m0 = w_scale.reshape(size_n, size_k // 32)
        if e8m0.dtype != torch.float8_e8m0fnu:
            e8m0 = e8m0.view(torch.uint8).view(torch.float8_e8m0fnu)
        # One gather instead of three torch passes (transpose, 8x8 block permute, 0,2,1,3
        # sub-block permute). Those three measured 85 us of a 133 us repack on
        # [2304,5120]; this kernel is bit-identical to them and does it in one launch.
        self.scales = _prep_scales(e8m0, size_k, size_n)
        self.size_n = size_n
        self.size_k = size_k


# one repack per distinct weight buffer; the buffer itself is kept alive as the key
_CACHE: dict[int, tuple[torch.Tensor, MarlinFp4Weight, torch.Tensor]] = {}
_WORKSPACE: dict[int, torch.Tensor] = {}


def _workspace(device) -> torch.Tensor:
    key = device.index or 0
    ws = _WORKSPACE.get(key)
    if ws is None:
        # one int per SM, as vLLM's marlin_make_workspace_new allocates
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        ws = torch.zeros(int(sms), dtype=torch.int32, device=device)
        _WORKSPACE[key] = ws
    return ws


# Caching a repacked expert costs VRAM as well as the repack: the entry has to keep the
# source tensors alive (a recycled _blob_stage slot would otherwise invalidate the copy),
# which is 6.0 MB per expert at this model's shape. Measured on an 8GB card:
#
#     TI_MARLIN_CACHE=1    prefill(8 tokens)  1.9 s
#     TI_MARLIN_CACHE=64   prefill(8 tokens)  4.8 s
#     TI_MARLIN_CACHE=512  prefill(8 tokens) 11.3 s
#
# Larger is monotonically worse -- the retained weights evict buffers the streaming ring
# needs, and an unbounded cache tripped the CUDA allocator outright
# (`!handles_.at(i) INTERNAL ASSERT FAILED`). So the repack is paid per call. That is not
# a bug to fix here: it is the 8GB card saying it cannot afford a repack cache, and the
# decode path does not go through marlin at all (fp4_linear is reached only by the shared
# expert during prefill), so there was nothing to win.
_CACHE_MAX = 1


def _prepared(w_fp4: torch.Tensor, w_scale: torch.Tensor, key=None) -> MarlinFp4Weight:
    if key is None:
        key = w_fp4.data_ptr()
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is w_fp4:
        return hit[1]
    weight = MarlinFp4Weight(w_fp4, w_scale)
    _CACHE.clear()
    _CACHE[key] = (w_fp4, weight, w_scale)
    return weight


# The Marlin kernel this file was written around, now built from our own copy of its
# sources (native/marlin) instead of coming from the vendored extension. The mangled name
# is the C++ template instantiation for
#   Marlin<bf16 act, fp4 weight, bf16 out, e8m0 scales,
#          256 threads, 1 m-block, 8 n-blocks, 8 k-blocks, m_block_size_8,
#          4 stages, group_blocks=2, is_zp_float=false>
# which is the configuration the reference dispatcher picks for every shape this engine
# uses (read back from the launched kernel name with torch.profiler).
_MARLIN_SYMBOL = (
    "_ZN6marlin6MarlinILx1125899906909960ELx562949953487106ELx1125899906909960"
    "ELx2814749767106568ELi256ELi1ELi8ELi8ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKf"
    "S3_S6_S3_iiiiPibbbi"
)
_NATIVE = None  # launcher, or False once we know it is unavailable


def _native_launcher():
    """Build (once) the cubin-backed Marlin launcher, or return None."""
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE or None
    try:
        from ..kernels import native

        props = torch.cuda.get_device_properties(0)
        sms = int(props.multi_processor_count)
        smem = int(props.shared_memory_per_block_optin)
        native.set_dynamic_smem("marlin_fp4", _MARLIN_SYMBOL, smem)
        _NATIVE = (native.bind("marlin_fp4", _MARLIN_SYMBOL, (sms,), (256,),
                               "TTOnnnTnni" + "iii" + "Ti" + "iii", shared=smem), smem)
    except Exception:
        _NATIVE = False
    return _NATIVE or None


def gemv_fp4_fast(x: torch.Tensor, w_fp4: torch.Tensor, w_scale: torch.Tensor,
                 key=None) -> torch.Tensor:
    """x [M,K] activation, w fp4 [N,K/2] packed, w_scale [N,K/32] e8m0 -> [M,N] bf16."""
    weight = _prepared(w_fp4, w_scale, key)
    a = x.reshape(-1, weight.size_k)
    if a.dtype != torch.bfloat16:
        a = a.to(torch.bfloat16)
    if not a.is_contiguous():
        a = a.contiguous()
    import os as _os
    got = None if _os.environ.get("TI_MARLIN_PYD") == "1" else _native_launcher()
    if got is not None:
        launch, smem = got
        out = torch.empty(a.size(0), weight.size_n, dtype=torch.bfloat16,
                          device=a.device)
        # A, B, C, C_tmp, bias, a_scales, b_scales, global_scale, zp,
        # prob_m, prob_n, prob_k, lda, locks, has_bias, atomic_add, fp32_reduce, smem
        launch(a, weight.qweight, out, None, None, None, weight.scales, None, None,
               a.size(0), weight.size_n, weight.size_k, a.stride(0),
               _workspace(a.device), 0, 0, 0, smem)
        return out
    return _ops().marlin_gemm(
        a,
        None,
        weight.qweight,
        None,
        weight.scales,
        None,
        None,
        None,
        _workspace(x.device),
        FP4_TYPE_ID,
        a.size(0),
        weight.size_n,
        weight.size_k,
        False,
        False,
        False,
    )