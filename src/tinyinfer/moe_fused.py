"""Fused MoE expert kernels (fp4 weights, sm_120 / tilelang 0.1.15).

fused_fp4_swiglu:  gate = x @ w1^T, up = x @ w3^T in ONE kernel (shared A quant),
                   clamp + silu(gate)*up epilogue -> h [M,N] bf16.
fused_fp4_down  :  y = (h @ w2^T) * weights[:,None] in ONE kernel (h quant inside).

Weights are the HF layout: packed E2M1 (2 per byte) + one e8m0 scale per 32 along K.
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
def fp4_swiglu_kernel(N: int, K: int, limit: float, block_M: int = 32):
    M = T.symbolic("M")
    block_N = 128
    block_K = 32
    nb = K // 32

    @T.prim_func
    def main(
        X: T.Tensor[(M, K), BF16],
        W1: T.Tensor[(N, K), FP4],
        S1: T.Tensor[(N, nb), FE8M0],
        W3: T.Tensor[(N, K), FP4],
        S3: T.Tensor[(N, nb), FE8M0],
        H: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_bf = T.alloc_shared((block_M, block_K), BF16)
            A_q = T.alloc_shared((block_M, block_K), FP8)
            B1p = T.alloc_shared((block_N, block_K), FP4)
            B1 = T.alloc_shared((block_N, block_K), FP8)
            B3p = T.alloc_shared((block_N, block_K), FP4)
            B3 = T.alloc_shared((block_N, block_K), FP8)
            C_local = T.alloc_fragment((block_M, block_N), FP32)
            Acc1 = T.alloc_fragment((block_M, block_N), FP32)
            Acc3 = T.alloc_fragment((block_M, block_N), FP32)
            A_f = T.alloc_fragment((block_M, block_K), FP32)
            A_abs = T.alloc_fragment((block_M, block_K), FP32)
            amax = T.alloc_fragment((block_M,), FP32)
            sa = T.alloc_fragment((block_M,), FP32)
            sb1 = T.alloc_fragment((block_N,), FP32)
            sb3 = T.alloc_fragment((block_N,), FP32)
            H_shared = T.alloc_shared((block_M, block_N), BF16)

            T.use_swizzle(panel_size=10)
            T.clear(Acc1)
            T.clear(Acc3)
            T.clear(C_local)
            for k in T.Pipelined(nb, num_stages=2):
                T.copy(X[by * block_M, k * block_K], A_bf)
                for i, j in T.Parallel(block_M, block_K):
                    A_f[i, j] = T.Cast(FP32, A_bf[i, j])
                for i, j in T.Parallel(block_M, block_K):
                    A_abs[i, j] = T.abs(A_f[i, j])
                T.reduce_max(A_abs, amax, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    sa[i] = T.pow(2.0, T.ceil(T.log2(T.max(amax[i], 1e-4) / 448.0)))
                for i, j in T.Parallel(block_M, block_K):
                    A_q[i, j] = T.Cast(FP8, T.min(T.max(A_f[i, j] / sa[i], -448.0), 448.0))

                T.copy(W1[bx * block_N, k * block_K], B1p)
                T.copy(W3[bx * block_N, k * block_K], B3p)
                for i, j in T.Parallel(block_N, block_K):
                    B1[i, j] = T.Cast(FP8, T.Cast(FP32, B1p[i, j]))
                for i, j in T.Parallel(block_N, block_K):
                    B3[i, j] = T.Cast(FP8, T.Cast(FP32, B3p[i, j]))
                for j in T.Parallel(block_N):
                    sb1[j] = T.Cast(FP32, S1[bx * block_N + j, k])
                for j in T.Parallel(block_N):
                    sb3[j] = T.Cast(FP32, S3[bx * block_N + j, k])

                T.gemm(A_q, B1, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    Acc1[i, j] += C_local[i, j] * sa[i] * sb1[j]
                T.clear(C_local)
                T.gemm(A_q, B3, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    Acc3[i, j] += C_local[i, j] * sa[i] * sb3[j]
                T.clear(C_local)

            for i, j in T.Parallel(block_M, block_N):
                # official semantics: gate/up are bf16 (fp4_gemm output) before clamp/silu
                g = T.Cast(FP32, T.Cast(BF16, Acc1[i, j]))
                u = T.Cast(FP32, T.Cast(BF16, Acc3[i, j]))
                if limit > 0:
                    u = T.min(T.max(u, -limit), limit)
                    g = T.min(g, limit)
                H_shared[i, j] = T.Cast(BF16, (g / (1.0 + T.exp(-g))) * u)
            T.copy(H_shared, H[by * block_M, bx * block_N])

    return main


@tilelang.jit(pass_configs=PASS)
def fp4_down_kernel(N: int, K: int, block_M: int = 32):
    """Y[M,N] = (H[M,K] @ W2[N,K]^T) * weights[:,None] ; N=dim (5120), K=inter (2304)."""
    M = T.symbolic("M")
    block_N = 128
    block_K = 32
    nb = K // 32

    @T.prim_func
    def main(
        H: T.Tensor[(M, K), BF16],
        W2: T.Tensor[(N, K), FP4],
        S2: T.Tensor[(N, nb), FE8M0],
        RW: T.Tensor[(M,), FP32],
        Y: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_bf = T.alloc_shared((block_M, block_K), BF16)
            A_q = T.alloc_shared((block_M, block_K), FP8)
            Bp = T.alloc_shared((block_N, block_K), FP4)
            B = T.alloc_shared((block_N, block_K), FP8)
            C_local = T.alloc_fragment((block_M, block_N), FP32)
            Acc = T.alloc_fragment((block_M, block_N), FP32)
            A_f = T.alloc_fragment((block_M, block_K), FP32)
            A_abs = T.alloc_fragment((block_M, block_K), FP32)
            amax = T.alloc_fragment((block_M,), FP32)
            sa = T.alloc_fragment((block_M,), FP32)
            sb = T.alloc_fragment((block_N,), FP32)
            Y_shared = T.alloc_shared((block_M, block_N), BF16)

            T.use_swizzle(panel_size=10)
            T.clear(Acc)
            T.clear(C_local)
            for k in T.Pipelined(nb, num_stages=2):
                T.copy(H[by * block_M, k * block_K], A_bf)
                for i, j in T.Parallel(block_M, block_K):
                    A_f[i, j] = T.Cast(FP32, A_bf[i, j])
                for i, j in T.Parallel(block_M, block_K):
                    A_abs[i, j] = T.abs(A_f[i, j])
                T.reduce_max(A_abs, amax, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    sa[i] = T.pow(2.0, T.ceil(T.log2(T.max(amax[i], 1e-4) / 448.0)))
                for i, j in T.Parallel(block_M, block_K):
                    A_q[i, j] = T.Cast(FP8, T.min(T.max(A_f[i, j] / sa[i], -448.0), 448.0))
                T.copy(W2[bx * block_N, k * block_K], Bp)
                for i, j in T.Parallel(block_N, block_K):
                    B[i, j] = T.Cast(FP8, T.Cast(FP32, Bp[i, j]))
                for j in T.Parallel(block_N):
                    sb[j] = T.Cast(FP32, S2[bx * block_N + j, k])
                T.gemm(A_q, B, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    Acc[i, j] += C_local[i, j] * sa[i] * sb[j]
                T.clear(C_local)
            for i, j in T.Parallel(block_M, block_N):
                Y_shared[i, j] = T.Cast(BF16, Acc[i, j] * RW[by * block_M + i])
            T.copy(Y_shared, Y[by * block_M, bx * block_N])

    return main


@tilelang.jit(pass_configs=PASS)
def fp4_swiglu_pq_kernel(N: int, K: int, limit: float, block_M: int = 32, block_N: int = 128):
    """A pre-quantized (Aq fp8, As e8m0): gate+up fused, silu epilogue."""
    M = T.symbolic("M")
    block_K = 32
    nb = K // 32

    @T.prim_func
    def main(
        Aq: T.Tensor[(M, K), FP8],
        AS: T.Tensor[(M, nb), FE8M0],
        W1: T.Tensor[(N, K), FP4],
        S1: T.Tensor[(N, nb), FE8M0],
        W3: T.Tensor[(N, K), FP4],
        S3: T.Tensor[(N, nb), FE8M0],
        H: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B1p = T.alloc_shared((block_N, block_K), FP4)
            B1 = T.alloc_shared((block_N, block_K), FP8)
            B3p = T.alloc_shared((block_N, block_K), FP4)
            B3 = T.alloc_shared((block_N, block_K), FP8)
            C_local = T.alloc_fragment((block_M, block_N), FP32)
            Acc1 = T.alloc_fragment((block_M, block_N), FP32)
            Acc3 = T.alloc_fragment((block_M, block_N), FP32)
            sa = T.alloc_fragment((block_M,), FP32)
            sb1 = T.alloc_fragment((block_N,), FP32)
            sb3 = T.alloc_fragment((block_N,), FP32)
            H_shared = T.alloc_shared((block_M, block_N), BF16)

            T.use_swizzle(panel_size=10)
            T.clear(Acc1); T.clear(Acc3); T.clear(C_local)
            for k in T.Pipelined(nb, num_stages=2):
                T.copy(Aq[by * block_M, k * block_K], A_shared)
                T.copy(W1[bx * block_N, k * block_K], B1p)
                T.copy(W3[bx * block_N, k * block_K], B3p)
                for j in T.Parallel(block_M):
                    sa[j] = T.Cast(FP32, AS[by * block_M + j, k])
                for i, j in T.Parallel(block_N, block_K):
                    B1[i, j] = T.Cast(FP8, T.Cast(FP32, B1p[i, j]))
                for i, j in T.Parallel(block_N, block_K):
                    B3[i, j] = T.Cast(FP8, T.Cast(FP32, B3p[i, j]))
                for i in T.Parallel(block_N):
                    sb1[i] = T.Cast(FP32, S1[bx * block_N + i, k])
                for i in T.Parallel(block_N):
                    sb3[i] = T.Cast(FP32, S3[bx * block_N + i, k])
                T.gemm(A_shared, B1, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    Acc1[i, j] += C_local[i, j] * sa[i] * sb1[j]
                T.clear(C_local)
                T.gemm(A_shared, B3, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    Acc3[i, j] += C_local[i, j] * sa[i] * sb3[j]
                T.clear(C_local)
            for i, j in T.Parallel(block_M, block_N):
                g = T.Cast(FP32, T.Cast(BF16, Acc1[i, j]))
                u = T.Cast(FP32, T.Cast(BF16, Acc3[i, j]))
                if limit > 0:
                    u = T.min(T.max(u, -limit), limit)
                    g = T.min(g, limit)
                H_shared[i, j] = T.Cast(BF16, (g / (1.0 + T.exp(-g))) * u)
            T.copy(H_shared, H[by * block_M, bx * block_N])

    return main


@tilelang.jit(pass_configs=PASS)
def fp4_down_pq_kernel(N: int, K: int, block_M: int = 32, block_N: int = 128):
    """A pre-quantized down: y = (Aq @ W2^T) * scale * weights."""
    M = T.symbolic("M")
    block_K = 32
    nb = K // 32

    @T.prim_func
    def main(
        Aq: T.Tensor[(M, K), FP8],
        AS: T.Tensor[(M, nb), FE8M0],
        W2: T.Tensor[(N, K), FP4],
        S2: T.Tensor[(N, nb), FE8M0],
        RW: T.Tensor[(M,), FP32],
        Y: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            Bp = T.alloc_shared((block_N, block_K), FP4)
            B = T.alloc_shared((block_N, block_K), FP8)
            C_local = T.alloc_fragment((block_M, block_N), FP32)
            Acc = T.alloc_fragment((block_M, block_N), FP32)
            sa = T.alloc_fragment((block_M,), FP32)
            sb = T.alloc_fragment((block_N,), FP32)
            Y_shared = T.alloc_shared((block_M, block_N), BF16)

            T.use_swizzle(panel_size=10)
            T.clear(Acc); T.clear(C_local)
            for k in T.Pipelined(nb, num_stages=2):
                T.copy(Aq[by * block_M, k * block_K], A_shared)
                T.copy(W2[bx * block_N, k * block_K], Bp)
                for j in T.Parallel(block_M):
                    sa[j] = T.Cast(FP32, AS[by * block_M + j, k])
                for i, j in T.Parallel(block_N, block_K):
                    B[i, j] = T.Cast(FP8, T.Cast(FP32, Bp[i, j]))
                for i in T.Parallel(block_N):
                    sb[i] = T.Cast(FP32, S2[bx * block_N + i, k])
                T.gemm(A_shared, B, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    Acc[i, j] += C_local[i, j] * sa[i] * sb[j]
                T.clear(C_local)
            for i, j in T.Parallel(block_M, block_N):
                Y_shared[i, j] = T.Cast(BF16, Acc[i, j] * RW[by * block_M + i])
            T.copy(Y_shared, Y[by * block_M, bx * block_N])

    return main


# ---------------- wrappers ----------------

def _packed(w: torch.Tensor) -> torch.Tensor:
    if w.dtype == torch.float4_e2m1fn_x2:
        return w
    return w.view(torch.float4_e2m1fn_x2)


def fused_fp4_swiglu(x: torch.Tensor, w1: torch.Tensor, s1: torch.Tensor,
                     w3: torch.Tensor, s3: torch.Tensor, limit: float = 10.0,
                     block_M: int = 32) -> torch.Tensor:
    """x [M,K] bf16 -> h [M,N] bf16 = silu(clamp(gate))*clamp(up), gate/up in one kernel."""
    x = x.contiguous(); m, k = x.shape; n = w1.shape[0]
    h = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    fp4_swiglu_kernel(n, k, float(limit), int(block_M))(
        x, _packed(w1).contiguous(), s1.contiguous(), _packed(w3).contiguous(), s3.contiguous(), h)
    return h


def fused_fp4_down(h: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor, weights: torch.Tensor,
                   block_M: int = 32) -> torch.Tensor:
    """h [M,K] bf16, w2 [N,K] packed fp4, s2 [N,K/32] -> y [M,N] bf16 (routing weights folded in)."""
    h = h.contiguous(); m, k = h.shape; n = w2.shape[0]
    y = torch.empty(m, n, device=h.device, dtype=torch.bfloat16)
    fp4_down_kernel(n, k, int(block_M))(
        h, _packed(w2).contiguous(), s2.contiguous(), weights.float().contiguous(), y)
    return y

def fused_fp4_swiglu_pq(aq, asc, w1, s1, w3, s3, limit=10.0, block_M=32, block_N=128):
    m, k = aq.shape; n = w1.shape[0]
    h = torch.empty(m, n, device=aq.device, dtype=torch.bfloat16)
    fp4_swiglu_pq_kernel(n, k, float(limit), int(block_M), int(block_N))(
        aq.contiguous(), asc.contiguous(), _packed(w1).contiguous(), s1.contiguous(),
        _packed(w3).contiguous(), s3.contiguous(), h)
    return h


def fused_fp4_down_pq(aq, asc, w2, s2, weights, block_M=32, block_N=128):
    m, k = aq.shape; n = w2.shape[0]
    y = torch.empty(m, n, device=aq.device, dtype=torch.bfloat16)
    fp4_down_pq_kernel(n, k, int(block_M), int(block_N))(
        aq.contiguous(), asc.contiguous(), _packed(w2).contiguous(), s2.contiguous(),
        weights.float().contiguous(), y)
    return y

def fused_fp4_expert_ffn(x, w1, s1, w3, s3, w2, s2, weights, limit=10.0,
                          block_M=32, block_N=128):
    """Full routed expert: act_quant(x) -> fused gate/up+silu -> act_quant(h) -> fused down+weights.

    Returns y [M,dim] bf16. This is the integration entry point for tinyinfer.moe.Expert.
    """
    import kernel as _k  # reference/official/kernel.py
    aq, asc = _k.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    h = fused_fp4_swiglu_pq(aq, asc, w1, s1, w3, s3, limit, block_M, block_N)
    hq, hs = _k.act_quant(h, 32, "ue8m0", torch.float8_e8m0fnu)
    return fused_fp4_down_pq(hq, hs, w2, s2, weights, block_M, block_N)
