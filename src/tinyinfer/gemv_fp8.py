"""Decode-only FP8 GEMV + optional RoPE epilogue (sm_120 / tilelang 0.1.15).

For M=1 the official fp8_gemm uses block_M=32 (31/32 wasted) and only reaches ~160 GB/s.
This kernel: one thread per (output row, k-lane), K serial accumulation, rows=16 per block,
coalesced 32-byte weight reads, per-32 activation scale applied inside the dot.

Variants:
  gemv_pq  : pre-quantized activation (Aq fp8 [K], As fp32 [K/32]) -> fastest path
  gemv_pq_rope : + RoPE epilogue on the last rope_dim of each head (wq_b / wkv)
  gemv_bf16: bf16 input, per-block internal activation quant (one kernel, no act_quant launch)
"""
import torch
import tilelang
import tilelang.language as T

tilelang.set_log_level("WARNING")
PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
BF16, FP32, FP8, FE8M0 = "bfloat16", "float32", "float8_e4m3", "float8_e8m0fnu"


def _dot_body(A, AS, W, WS, Y, bx, rows, K, nb, rope, head_dim, rope_dim, COS, SIN, N):
    pass


@tilelang.jit(pass_configs=PASS)
def gemv_pq_kernel(N: int, K: int, rows: int, rope: bool, head_dim: int, rope_dim: int):
    nb = K // 32
    grid = T.ceildiv(N, rows)
    half = rope_dim // 2

    @T.prim_func
    def main(
        A: T.Tensor[(K,), FP8],
        AS: T.Tensor[(nb,), FP32],
        W: T.Tensor[(N, K), FP8],
        WS: T.Tensor[(N // 32, nb), FE8M0],
        Y: T.Tensor[(N,), BF16],
        COS: T.Tensor[(1, half), FP32],
        SIN: T.Tensor[(1, half), FP32],
    ):
        with T.Kernel(grid, threads=rows * 32) as bx:
            acc = T.alloc_fragment((rows, 32), FP32)
            ysum = T.alloc_fragment((rows,), FP32)
            yrow = T.alloc_shared((rows,), FP32)
            n0 = bx * rows
            for r, l in T.Parallel(rows, 32):
                acc[r, l] = 0.0
            for kb in T.serial(nb):
                for r, l in T.Parallel(rows, 32):
                    n = n0 + r
                    acc[r, l] += T.Cast(FP32, A[kb * 32 + l]) * T.Cast(FP32, W[n, kb * 32 + l]) * AS[kb] * T.Cast(FP32, WS[n // 32, kb])
            T.reduce_sum(acc, ysum, dim=1)
            if rope:
                for r in T.Parallel(rows):
                    yrow[r] = ysum[r]
                for p in T.Parallel(rows // 2):
                    loc = (n0 + 2 * p) % head_dim
                    inr = loc >= head_dim - rope_dim
                    pid = T.max((loc - (head_dim - rope_dim)) // 2, 0)
                    a = yrow[2 * p]
                    b = yrow[2 * p + 1]
                    c = T.if_then_else(inr, COS[0, pid], 1.0)
                    s = T.if_then_else(inr, SIN[0, pid], 0.0)
                    Y[n0 + 2 * p] = T.Cast(BF16, T.if_then_else(inr, a * c - b * s, a))
                    Y[n0 + 2 * p + 1] = T.Cast(BF16, T.if_then_else(inr, a * s + b * c, b))
            else:
                for r in T.Parallel(rows):
                    Y[n0 + r] = T.Cast(BF16, ysum[r])

    return main


@tilelang.jit(pass_configs=PASS)
def gemv_bf16_kernel_v2(N: int, K: int, rows: int):
    nb = K // 32
    grid = T.ceildiv(N, rows)

    @T.prim_func
    def main(
        X: T.Tensor[(K,), BF16],
        W: T.Tensor[(N, K), FP8],
        WS: T.Tensor[(N // 32, nb), FE8M0],
        Y: T.Tensor[(N,), BF16],
    ):
        with T.Kernel(grid, threads=rows * 32) as bx:
            xv = T.alloc_fragment((nb, 32), FP32)
            av = T.alloc_fragment((nb, 32), FP32)
            segmax = T.alloc_fragment((nb,), FP32)
            aq = T.alloc_shared((K,), FP8)
            asc = T.alloc_shared((nb,), FP32)
            acc = T.alloc_fragment((rows, 32), FP32)
            ysum = T.alloc_fragment((rows,), FP32)
            n0 = bx * rows
            for kb, j in T.Parallel(nb, 32):
                xv[kb, j] = T.Cast(FP32, X[kb * 32 + j])
            for kb, j in T.Parallel(nb, 32):
                av[kb, j] = T.abs(xv[kb, j])
            T.reduce_max(av, segmax, dim=1, clear=True)
            for kb in T.Parallel(nb):
                asc[kb] = T.pow(2.0, T.ceil(T.log2(T.max(segmax[kb], 1e-4) / 448.0)))
            for kb, j in T.Parallel(nb, 32):
                aq[kb * 32 + j] = T.Cast(FP8, T.min(T.max(xv[kb, j] / asc[kb], -448.0), 448.0))
            for r, l in T.Parallel(rows, 32):
                acc[r, l] = 0.0
            for kb in T.serial(nb):
                for r, l in T.Parallel(rows, 32):
                    n = n0 + r
                    acc[r, l] += T.Cast(FP32, aq[kb * 32 + l]) * T.Cast(FP32, W[n, kb * 32 + l]) * asc[kb] * T.Cast(FP32, WS[n // 32, kb])
            T.reduce_sum(acc, ysum, dim=1)
            for r in T.Parallel(rows):
                Y[n0 + r] = T.Cast(BF16, ysum[r])

    return main


@tilelang.jit(pass_configs=PASS)
def gemv_pq4_kernel(N: int, K: int, rows: int, rope: bool, head_dim: int, rope_dim: int):
    """4 k-values per thread (lanes=8 per row): fewer iterations + 32-bit weight loads."""
    nb = K // 32
    lanes = 8
    grid = T.ceildiv(N, rows)
    half = rope_dim // 2

    @T.prim_func
    def main(
        A: T.Tensor[(K,), FP8],
        AS: T.Tensor[(nb,), FP32],
        W: T.Tensor[(N, K), FP8],
        WS: T.Tensor[(N // 32, nb), FE8M0],
        Y: T.Tensor[(N,), BF16],
        COS: T.Tensor[(1, half), FP32],
        SIN: T.Tensor[(1, half), FP32],
    ):
        with T.Kernel(grid, threads=rows * lanes) as bx:
            acc = T.alloc_fragment((rows, lanes), FP32)
            ysum = T.alloc_fragment((rows,), FP32)
            yrow = T.alloc_shared((rows,), FP32)
            n0 = bx * rows
            for r, l in T.Parallel(rows, lanes):
                acc[r, l] = 0.0
            for kb in T.serial(nb):
                for r, l in T.Parallel(rows, lanes):
                    n = n0 + r
                    k0 = kb * 32 + l * 4
                    s4 = (T.Cast(FP32, A[k0 + 0]) * T.Cast(FP32, W[n, k0 + 0])
                          + T.Cast(FP32, A[k0 + 1]) * T.Cast(FP32, W[n, k0 + 1])
                          + T.Cast(FP32, A[k0 + 2]) * T.Cast(FP32, W[n, k0 + 2])
                          + T.Cast(FP32, A[k0 + 3]) * T.Cast(FP32, W[n, k0 + 3]))
                    acc[r, l] += s4 * AS[kb] * T.Cast(FP32, WS[n // 32, kb])
            T.reduce_sum(acc, ysum, dim=1)
            if rope:
                for r in T.Parallel(rows):
                    yrow[r] = ysum[r]
                for p in T.Parallel(rows // 2):
                    loc = (n0 + 2 * p) % head_dim
                    inr = loc >= head_dim - rope_dim
                    pid = T.max((loc - (head_dim - rope_dim)) // 2, 0)
                    a = yrow[2 * p]
                    b = yrow[2 * p + 1]
                    c = T.if_then_else(inr, COS[0, pid], 1.0)
                    si = T.if_then_else(inr, SIN[0, pid], 0.0)
                    Y[n0 + 2 * p] = T.Cast(BF16, T.if_then_else(inr, a * c - b * si, a))
                    Y[n0 + 2 * p + 1] = T.Cast(BF16, T.if_then_else(inr, a * si + b * c, b))
            else:
                for r in T.Parallel(rows):
                    Y[n0 + r] = T.Cast(BF16, ysum[r])

    return main


# ---------------- wrappers ----------------

def fused_fp8_gemv(x_bf16: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor,
                   rows: int = 16) -> torch.Tensor:
    """x [1,K] bf16, w [N,K] fp8, w_scale [N/32,K/32] -> y [1,N] bf16 (internal act quant)."""
    x = x_bf16.reshape(-1).contiguous()
    k = x.numel()
    n = w_fp8.shape[0]
    y = torch.empty(n, device=x.device, dtype=torch.bfloat16)
    gemv_bf16_kernel_v2(n, k, int(rows))(x, w_fp8.contiguous(), w_scale.contiguous(), y)
    return y.view(1, n)


def fused_fp8_gemv_pq(aq: torch.Tensor, a_scale: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor,
                       rows: int = 16, rope: bool = False, head_dim: int = 512, rope_dim: int = 64,
                       cos: torch.Tensor | None = None, sin: torch.Tensor | None = None) -> torch.Tensor:
    """pre-quantized A: aq [K] fp8, a_scale [K/32] fp32; optional RoPE epilogue."""
    k = aq.numel(); n = w_fp8.shape[0]
    y = torch.empty(n, device=aq.device, dtype=torch.bfloat16)
    if rope:
        assert cos is not None and sin is not None
        gemv_pq_kernel(n, k, int(rows), True, int(head_dim), int(rope_dim))(
            aq.contiguous(), a_scale.float().contiguous(), w_fp8.contiguous(), w_scale.contiguous(), y,
            cos.float().contiguous(), sin.float().contiguous())
    else:
        empty = torch.empty(1, 0, device=aq.device, dtype=torch.float32)
        gemv_pq_kernel(n, k, int(rows), False, 0, 0)(
            aq.contiguous(), a_scale.float().contiguous(), w_fp8.contiguous(), w_scale.contiguous(), y,
            empty, empty)
    return y.view(1, n)

def fused_fp8_gemv_pq4(aq: torch.Tensor, a_scale: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor,
                        rows: int = 64, rope: bool = False, head_dim: int = 512, rope_dim: int = 64,
                        cos: torch.Tensor | None = None, sin: torch.Tensor | None = None) -> torch.Tensor:
    k = aq.numel(); n = w_fp8.shape[0]
    y = torch.empty(n, device=aq.device, dtype=torch.bfloat16)
    if rope:
        gemv_pq4_kernel(n, k, int(rows), True, int(head_dim), int(rope_dim))(
            aq.contiguous(), a_scale.float().contiguous(), w_fp8.contiguous(), w_scale.contiguous(), y,
            cos.float().contiguous(), sin.float().contiguous())
    else:
        empty = torch.empty(1, 0, device=aq.device, dtype=torch.float32)
        gemv_pq4_kernel(n, k, int(rows), False, 0, 0)(
            aq.contiguous(), a_scale.float().contiguous(), w_fp8.contiguous(), w_scale.contiguous(), y,
            empty, empty)
    return y.view(1, n)