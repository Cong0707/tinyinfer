"""M==1 fp4 GEMM (block_M=16): the official kernel fixes block_M=32, wasting 31/32
rows for decode. This drops the tile to 16 (still mma-friendly) so M=1 wastes 15/16.

Same math as reference/official/kernel.py fp4_gemm (32x32 ue8m0 scales, fp4->fp8 cast).
"""
import torch
import tilelang
import tilelang.language as T

tilelang.set_log_level("WARNING")
PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
BF16, FP32, FP8, FP4, FE8M0 = "bfloat16", "float32", "float8_e4m3", "float4_e2m1fn", "float8_e8m0fnu"


@tilelang.jit(pass_configs=PASS)
def fp4_gemm_m1_kernel(N: int, K: int, block_M: int = 16, block_N: int = 128, block_K: int = 32,
                       scale_dtype: str = FE8M0):
    M = T.symbolic("M")
    n_sub = 1

    @T.prim_func
    def main(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP4],
        C: T.Tensor[(M, N), BF16],
        scales_a: T.Tensor[(M, T.ceildiv(K, 32)), FP32],
        scales_b: T.Tensor[(N, T.ceildiv(K, 32)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), FP8)
            Bp = T.alloc_shared((block_N, block_K), FP4)
            Bs = T.alloc_shared((block_N, block_K), FP8)
            Cs = T.alloc_shared((block_M, block_N), BF16)
            Cacc = T.alloc_fragment((block_M, block_N), FP32)
            Cl = T.alloc_fragment((block_M, block_N), FP32)
            sa = T.alloc_fragment((block_M,), FP32)
            sb = T.alloc_fragment((block_N,), FP32)
            T.use_swizzle(panel_size=10)
            T.clear(Cacc)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_s)
                T.copy(B[bx * block_N, k * block_K], Bp)
                for i, j in T.Parallel(block_N, block_K):
                    Bs[i, j] = T.Cast(FP8, T.Cast(FP32, Bp[i, j]))
                for i in T.Parallel(block_N):
                    sb[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])
                for i in T.Parallel(block_M):
                    sa[i] = scales_a[by * block_M + i, k]
                T.clear(Cl)  # gemm accumulates into C; must start clean each k step
                T.gemm(A_s, Bs, Cl, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    Cacc[i, j] += Cl[i, j] * sa[i] * sb[j]
            T.copy(Cacc, Cs)
            T.copy(Cs, C[by * block_M, bx * block_N])

    return main


def fp4_gemm_m1(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """a fp8 [M,K] (M small), a_s fp32 [M,K/32], b fp4 [N,K] packed, b_s [N,K/32] ue8m0."""
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    fp4_gemm_m1_kernel(N, K)(a.view(M, K), b, c, a_s.view(M, -1).float(), b_s)
    return c
