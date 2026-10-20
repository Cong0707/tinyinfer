"""Fused kernels for the per-layer attention Python loop (sm_120 / tilelang 0.1.15).

fused_rmsnorm_rope : row RMSNorm (fp32) + RoPE on the last rope_dim elements, one kernel.
fused_gemm_rope    : official blockwise fp8 GEMM (32x32 ue8m0) + RoPE epilogue, one kernel.

Both take precomputed cos/sin tables [M, rope_dim/2] (fp32), one row per token/row;
the wrapper can expand a [B*S, rope/2] table over heads for q.
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


@tilelang.jit(pass_configs=PASS)
def rmsnorm_rope_kernel(N: int, rope_dim: int, eps: float):
    """x [M,N] bf16 -> y [M,N] bf16: y = rmsnorm(x)*w, then RoPE on x[..., N-rope_dim:]."""
    M = T.symbolic("M")
    half = rope_dim // 2
    threads = 256 if N >= 1024 else 128

    @T.prim_func
    def main(
        X: T.Tensor[(M, N), BF16],
        W: T.Tensor[(N,), BF16],
        COS: T.Tensor[(M, half), FP32],
        SIN: T.Tensor[(M, half), FP32],
        Y: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(M, threads=threads) as bx:
            x_local = T.alloc_fragment((N,), FP32)
            sq = T.alloc_fragment((N,), FP32)
            xs = T.alloc_shared((N,), FP32)
            ss = T.alloc_fragment((1,), FP32)
            r = T.alloc_fragment((1,), FP32)
            for j in T.Parallel(N):
                x_local[j] = T.Cast(FP32, X[bx, j])
            for j in T.Parallel(N):
                sq[j] = x_local[j] * x_local[j]
            T.reduce_sum(sq, ss, dim=0)
            r[0] = T.rsqrt(ss[0] / N + eps)
            for j in T.Parallel(N):
                xs[j] = x_local[j] * r[0] * T.Cast(FP32, W[j])
            if rope_dim > 0:
                for p in T.Parallel(half):
                    a = xs[N - rope_dim + 2 * p]
                    b = xs[N - rope_dim + 2 * p + 1]
                    c = COS[bx, p]
                    s = SIN[bx, p]
                    xs[N - rope_dim + 2 * p] = a * c - b * s
                    xs[N - rope_dim + 2 * p + 1] = a * s + b * c
            for j in T.Parallel(N):
                Y[bx, j] = T.Cast(BF16, xs[j])

    return main


@tilelang.jit(pass_configs=PASS)
def gemm_rope_kernel(N: int, K: int, head_dim: int, rope_dim: int, block_size: int = 32):
    """A fp8 [M,K] x B fp8 [N,K]^T (32x32 ue8m0 scales) + RoPE epilogue on the last rope_dim of each head."""
    M = T.symbolic("M")
    group_size = block_size
    block_M = 32
    block_N = 128
    block_K = group_size
    num_stages = 2
    nbh = head_dim // block_N          # 128-col blocks per head
    rope_half = rope_dim // 2
    rope_pairs = rope_dim // 2
    rope_start = block_N - rope_dim    # local start of the rope region in the last block of a head

    @T.prim_func
    def main(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), BF16],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), FE8M0],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), FE8M0],
        COS: T.Tensor[(M, rope_half), FP32],
        SIN: T.Tensor[(M, rope_half), FP32],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), BF16)
            C_rope = T.alloc_shared((block_M, block_N), BF16)
            C_local = T.alloc_fragment((block_M, block_N), FP32)
            C_accum = T.alloc_fragment((block_M, block_N), FP32)
            scale_a_frag = T.alloc_fragment((block_M,), FP32)
            scale_b_frag = T.alloc_fragment((block_N,), FP32)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                for i in T.Parallel(block_M):
                    scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k])
                for j in T.Parallel(block_N):
                    scale_b_frag[j] = T.Cast(FP32, scales_b[(bx * block_N + j) // group_size, k])
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                T.clear(C_local)

            # epilogue: raw result to shared, then RoPE on the last 128-col block of every head.
            # T.copy to C handles the M boundary (rows >= M are dropped).
            T.copy(C_accum, C_shared)
            for i, pp in T.Parallel(block_M, block_N // 2):
                row = T.min(by * block_M + i, M - 1)
                pp_idx = pp - rope_start // 2
                rope_sel = T.if_then_else((bx % nbh) == (nbh - 1),
                                          T.if_then_else(pp_idx >= 0, 1, 0), 0)
                pid = T.max(pp_idx, 0)
                c = T.if_then_else(rope_sel == 1, COS[row, pid], 1.0)
                s = T.if_then_else(rope_sel == 1, SIN[row, pid], 0.0)
                a = T.Cast(FP32, C_shared[i, 2 * pp])
                b = T.Cast(FP32, C_shared[i, 2 * pp + 1])
                C_rope[i, 2 * pp] = T.Cast(BF16, T.if_then_else(rope_sel == 1, a * c - b * s, a))
                C_rope[i, 2 * pp + 1] = T.Cast(BF16, T.if_then_else(rope_sel == 1, a * s + b * c, b))
            T.copy(C_rope, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(pass_configs=PASS)
def rope_inplace_kernel(N: int, rope_dim: int, inverse: bool, single: bool):
    """Rotate the last rope_dim elements of each row in place (optionally inverse, single cos/sin row)."""
    M = T.symbolic("M")
    half = rope_dim // 2

    rows = 1 if single else M

    @T.prim_func
    def main(
        X: T.Tensor[(M, N), BF16],
        COS: T.Tensor[(rows, half), FP32],
        SIN: T.Tensor[(rows, half), FP32],
    ):
        with T.Kernel(M, threads=128) as bx:
            for p in T.Parallel(half):
                r = 0 if single else bx
                a = T.Cast(FP32, X[bx, N - rope_dim + 2 * p])
                b = T.Cast(FP32, X[bx, N - rope_dim + 2 * p + 1])
                c = COS[r, p]
                s = SIN[r, p]
                if inverse:
                    s = -s
                X[bx, N - rope_dim + 2 * p] = T.Cast(BF16, a * c - b * s)
                X[bx, N - rope_dim + 2 * p + 1] = T.Cast(BF16, a * s + b * c)

    return main


# ---------------- wrappers ----------------

def fused_rmsnorm_rope(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       rope_dim: int, eps: float = 1e-20) -> torch.Tensor:
    """x [M,N] (or [B,S,N]) bf16; cos/sin [M, rope_dim/2] fp32; output same shape as x."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    m, n = x2.shape
    if rope_dim:
        assert cos.shape == (m, rope_dim // 2), (cos.shape, (m, rope_dim // 2))
        assert sin.shape == cos.shape
    else:
        cos = torch.empty(m, 0, device=x.device, dtype=torch.float32)
        sin = torch.empty(m, 0, device=x.device, dtype=torch.float32)
    y = torch.empty_like(x2)
    rmsnorm_rope_kernel(n, int(rope_dim), float(eps))(x2, weight.contiguous(), cos.float().contiguous(), sin.float().contiguous(), y)
    return y.view(shape)


def fused_rope_inplace(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int,
                      inverse: bool = False) -> None:
    """In-place RoPE on the last rope_dim of x [M,N] (cos/sin [1,half] shared or [M,half])."""
    x2 = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    m = x2.shape[0]
    single = cos.shape[0] == 1
    if not single:
        assert cos.shape[0] == m, (cos.shape, m)
    rope_inplace_kernel(x2.shape[1], int(rope_dim), bool(inverse), bool(single))(
        x2, cos.float().contiguous(), sin.float().contiguous())


def fused_gemm_rope(a_fp8: torch.Tensor, a_scale: torch.Tensor, b_fp8: torch.Tensor, b_scale: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor, head_dim: int = 512, rope_dim: int = 64,
                    block_size: int = 32) -> torch.Tensor:
    """C = A_fp8 @ B_fp8^T + RoPE epilogue -> bf16 [M,N] (C dtype bf16 required by the kernel)."""
    m, k = a_fp8.shape
    n = b_fp8.shape[0]
    c = torch.empty(m, n, device=a_fp8.device, dtype=torch.bfloat16)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        gemm_rope_kernel(n, k, int(head_dim), int(rope_dim), int(block_size))(
            a_fp8.contiguous(), b_fp8.contiguous(), c, a_scale.contiguous(), b_scale.contiguous(),
            cos.contiguous().float(), sin.contiguous().float())
    finally:
        torch.set_default_dtype(prev)
    return c