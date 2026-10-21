"""Optional tilelang backends (O2). Lazy, cached, fallback-safe.

The tilelang JIT needs nvcc + MSVC cl.exe (vcvars64); when unavailable the
dispatch silently falls back to the PyTorch reference implementation.
"""
import functools

import torch

from .nn import sparse_attn_ref

_TL_STATE: bool | None = None  # None = untried, True = works, False = disabled

try:
    import tilelang as _TL
    _PASS = {
        _TL.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        _TL.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
except Exception:  # pragma: no cover - environment without tilelang
    _TL = None
    _PASS = None


if _TL is not None:

    @_TL.jit(pass_configs=_PASS)
    def _sparse_attn_kernel(h: int, d: int, h_chunk: int, block: int, scale: float):
        import tilelang.language as T

        BF16, FP32, INT32 = "bfloat16", "float32", "int32"
        nchunks = h // h_chunk
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        topk = T.symbolic("topk")
        num_blocks = _TL.cdiv(topk, block)

        @T.prim_func
        def main(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            o: T.Tensor[(b, m, h, d), BF16],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
        ):
            with T.Kernel(m, b, nchunks, threads=128) as (bx, by, bz):
                h0 = bz * h_chunk
                q_shared = T.alloc_shared((h_chunk, d), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
                o_shared = T.alloc_shared((h_chunk, d), BF16)
                acc_s_cast = T.alloc_shared((h_chunk, block), BF16)
                idxs = T.alloc_fragment(block, INT32)
                acc_s = T.alloc_fragment((h_chunk, block), FP32)
                acc_o = T.alloc_fragment((h_chunk, d), FP32)
                scores_max = T.alloc_fragment(h_chunk, FP32)
                scores_max_prev = T.alloc_fragment(h_chunk, FP32)
                scores_scale = T.alloc_fragment(h_chunk, FP32)
                scores_sum = T.alloc_fragment(h_chunk, FP32)
                sum_exp = T.alloc_fragment(h_chunk, FP32)
                sink_local = T.alloc_fragment(h_chunk, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -1e30)
                T.copy(q[by, bx, h0, 0], q_shared)
                for i in T.Parallel(h_chunk):
                    sink_local[i] = attn_sink[h0 + i]

                for t in T.Pipelined(num_blocks, num_stages=2):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                    for i, j in T.Parallel(h_chunk, block):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h_chunk, block):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h_chunk):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(h_chunk, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h_chunk):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h_chunk, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(h_chunk):
                    sum_exp[i] += T.exp(sink_local[i] - scores_max[i])
                for i, j in T.Parallel(h_chunk, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, h0, 0])

        return main

else:  # pragma: no cover
    _sparse_attn_kernel = None


def sparse_attn_tl(q, kv, attn_sink, topk_idxs, scale: float, h_chunk: int = 16, block: int = 32):
    """tilelang sm120 sparse attention (head-chunked; 99KB shared-mem limit)."""
    import os as _os
    h_chunk = int(_os.environ.get("TI_SA_HCHUNK", h_chunk))
    block = int(_os.environ.get("TI_SA_BLOCK", block))
    b, s, h, d = q.shape
    if h % h_chunk:
        h_chunk = h
    kernel = _sparse_attn_kernel(h, d, h_chunk, block, float(scale))
    o = torch.empty_like(q)
    kernel(q.contiguous(), kv.contiguous(), o, attn_sink.contiguous().float(), topk_idxs.contiguous().int())
    return o


def sparse_attn_dispatch(q, kv, attn_sink, topk_idxs, scale: float, prefer_tilelang: bool = True):
    import os as _o
    if _o.environ.get("TI_REF_ATTN") == "1":
        # Reference form: plain gather + torch softmax. Bit-matches the reference and is
        # used to test whether the tiled online-softmax is what makes a multi-row
        # prefill differ from the token-by-token path. The alias keeps the name local:
        # a bare import here shadowed the module-level fallback below and raised
        # UnboundLocalError in tests.
        from .nn import sparse_attn_ref as _ref
        return _ref(q, kv, attn_sink, topk_idxs, scale)
    """Official sparse_attn first (tilelang kernel.py), then our kernel, then torch.

    Levels 0-2 must produce the same numbers as the reference implementation, so the
    official kernel -- the one the checkpoint was validated against -- is the default
    path. Our own kernel and the pure-torch reference remain as fallbacks.
    """
    global _TL_STATE
    try:
        from .official_kernel_ref import sparse_attn as _official_sparse_attn

        # the official window helper hands back CPU int64 indices; its kernel wants CUDA int32
        return _official_sparse_attn(q, kv, attn_sink,
                                     topk_idxs.to(q.device).int(), float(scale))
    except Exception as _exc:
        import os as _os2
        if _os2.environ.get("TI_TL_DEBUG"):
            print(f"[sparse_attn] official kernel failed: {_exc!r}", flush=True)
    if prefer_tilelang and _TL_STATE is not False and _TL is not None:
        try:
            out = sparse_attn_tl(q, kv, attn_sink, topk_idxs, scale)
            _TL_STATE = True
            return out
        except Exception as _exc:
            import os as _os
            if _os.environ.get("TI_TL_DEBUG"):
                import traceback
                print(f"[sparse_attn] tilelang failed for shape q={tuple(q.shape)} "
                      f"kv={tuple(kv.shape)} idx={tuple(topk_idxs.shape)}: {_exc!r}", flush=True)
                traceback.print_exc()
            _TL_STATE = False
    return sparse_attn_ref(q, kv, attn_sink, topk_idxs, scale)
