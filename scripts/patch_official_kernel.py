"""Patch reference/official/kernel.py to use the sm120 sparse_attn (shared-mem 99KB limit).

The official kernel asks for 141KB dynamic shared memory; RTX 5060 caps at 99KB.
We keep the official file intact except for sparse_attn, which delegates to a
head-chunked implementation (same math, 65KB smem).
"""
import re
from pathlib import Path

KERNEL = Path(__file__).resolve().parents[1] / "reference" / "official" / "kernel.py"
src = KERNEL.read_text(encoding="utf-8")

MARK = "# --- tinyinfer sm120 patch: head-chunked sparse_attn (see tinyinfer/src/tinyinfer/kernels.py) ---"
if MARK in src:
    print("already patched")
else:
    # append a redefinition of sparse_attn that uses head chunking and replaces the original
    patch = '''

# --- tinyinfer sm120 patch: head-chunked sparse_attn (see tinyinfer/src/tinyinfer/kernels.py) ---
_tinyinfer_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

@tilelang.jit(pass_configs=_tinyinfer_pass_configs)
def _tinyinfer_sparse_attn_kernel(h: int, d: int, h_chunk: int, block: int, scale: float):
    nchunks = h // h_chunk
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    num_blocks = tilelang.cdiv(topk, block)

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


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """tinyinfer sm120 override (official signature): head-chunked to fit 99KB shared mem."""
    b, s, h, d = q.size()
    pad = 16 - h if h < 16 else 0
    if pad:
        q = torch.cat([q, q.new_zeros(b, s, pad, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(pad)])
    hh = q.size(2)
    h_chunk = 16 if hh % 16 == 0 else hh
    o = torch.empty_like(q)
    kernel = _tinyinfer_sparse_attn_kernel(hh, d, h_chunk, 32, softmax_scale)
    kernel(q.contiguous(), kv.contiguous(), o, attn_sink.contiguous().float(), topk_idxs.contiguous().int())
    return o.narrow(2, 0, h).contiguous() if pad else o
'''
    KERNEL.write_text(src + patch, encoding="utf-8")
    print("patched", KERNEL)
