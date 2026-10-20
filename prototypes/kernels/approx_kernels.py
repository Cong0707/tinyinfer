"""TileLang kernels comparing exact libdevice math vs approx_math approximations.

Implements (each with exact / fast variants where meaningful):
  - rmsnorm:      rsqrtf            vs bit-trick + Newton (approx_math.fast_rsqrt)
  - swiglu:       expf              vs __expf (HW) / polynomial fast_exp (approx_math)
  - softmax:      expf              vs __expf / polynomial fast_exp
  - rope_table:   precomputed cos/sin table (exact values) applied in-kernel

Designed for sm_120 99KB shared-mem limit; one row per block for norm/softmax.
"""
import torch
import tilelang
import tilelang.language as T

tilelang.set_log_level("WARNING")
PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
BF16, FP32 = "bfloat16", "float32"
LOG2E = 1.4426950408889634


def _fast_exp(expr):
    """exp(x) as 2^(x*log2e) with cubic mantissa; mirrors approx_math.fast_exp."""
    z = expr * LOG2E
    zi = T.floor(z)
    f = z - zi
    p = 1.0 + f * (0.6931471805599453 + f * (0.2402265069591007 + f * 0.0555041086648216))
    bits = T.Cast("int32", zi) + 127
    scale = T.reinterpret(FP32, T.shift_left(bits, 23))
    return p * scale


def _fast_rsqrt(expr, iters=1):
    """approx_math.fast_rsqrt: bit-trick initial guess + Newton iterations."""
    i = T.reinterpret("int32", expr)
    y = T.reinterpret(FP32, 0x5F3759DF - T.shift_right(i, 1))
    for _ in range(max(1, iters)):
        y = y * (1.5 - 0.5 * expr * y * y)
    return y


@tilelang.jit(pass_configs=PASS)
def rmsnorm_kernel(dim: int, mode: str, eps: float):
    M = T.symbolic("M")
    threads = 256 if dim >= 1024 else 128

    @T.prim_func
    def main(
        X: T.Tensor[(M, dim), BF16],
        W: T.Tensor[(dim,), BF16],
        Y: T.Tensor[(M, dim), BF16],
    ):
        with T.Kernel(M, threads=threads) as bx:
            x_local = T.alloc_fragment((dim,), FP32)
            sq = T.alloc_fragment((dim,), FP32)
            ss = T.alloc_fragment((1,), FP32)
            r = T.alloc_fragment((1,), FP32)
            for j in T.Parallel(dim):
                x_local[j] = T.Cast(FP32, X[bx, j])
            for j in T.Parallel(dim):
                sq[j] = x_local[j] * x_local[j]
            T.reduce_sum(sq, ss, dim=0)
            mean = ss[0] / dim
            if mode == "exact":
                r[0] = T.rsqrt(mean + eps)
            elif mode == "fast":
                r[0] = _fast_rsqrt(mean + eps, 1)
            else:
                r[0] = 1.0 / T.sqrt(mean + eps)
            for j in T.Parallel(dim):
                Y[bx, j] = T.Cast(BF16, x_local[j] * r[0] * T.Cast(FP32, W[j]))

    return main


@tilelang.jit(pass_configs=PASS)
def swiglu_kernel(dim: int, mode: str, limit: float):
    M = T.symbolic("M")

    @T.prim_func
    def main(
        G: T.Tensor[(M, dim), BF16],
        U: T.Tensor[(M, dim), BF16],
        Y: T.Tensor[(M, dim), BF16],
    ):
        with T.Kernel(M, threads=256) as bx:
            g = T.alloc_fragment((dim,), FP32)
            u = T.alloc_fragment((dim,), FP32)
            for j in T.Parallel(dim):
                g[j] = T.Cast(FP32, G[bx, j])
                u[j] = T.Cast(FP32, U[bx, j])
            for j in T.Parallel(dim):
                if limit > 0:
                    u[j] = T.min(T.max(u[j], -limit), limit)
                    g[j] = T.min(g[j], limit)
                if mode == "exact":
                    sig = 1.0 / (1.0 + T.exp(-g[j]))
                elif mode == "hw":
                    sig = 1.0 / (1.0 + T.__exp(-g[j]))
                else:
                    sig = 1.0 / (1.0 + _fast_exp(-g[j]))
                Y[bx, j] = T.Cast(BF16, (g[j] * sig) * u[j])

    return main


@tilelang.jit(pass_configs=PASS)
def softmax_kernel(cols: int, mode: str):
    M = T.symbolic("M")

    @T.prim_func
    def main(
        S: T.Tensor[(M, cols), FP32],
        P: T.Tensor[(M, cols), FP32],
    ):
        with T.Kernel(M, threads=128) as bx:
            s = T.alloc_fragment((cols,), FP32)
            e = T.alloc_fragment((cols,), FP32)
            mx = T.alloc_fragment((1,), FP32)
            sm = T.alloc_fragment((1,), FP32)
            for j in T.Parallel(cols):
                s[j] = S[bx, j]
            T.reduce_max(s, mx, dim=0, clear=True)
            for j in T.Parallel(cols):
                d = s[j] - mx[0]
                if mode == "exact":
                    e[j] = T.exp(d)
                elif mode == "hw":
                    e[j] = T.__exp(d)
                else:
                    e[j] = _fast_exp(d)
            T.reduce_sum(e, sm, dim=0)
            for j in T.Parallel(cols):
                P[bx, j] = e[j] / sm[0]

    return main


@tilelang.jit(pass_configs=PASS)
def rope_table_kernel(rope_dim: int, rows_per_block: int = 8):
    """Apply RoPE using a precomputed cos/sin table: x [M, rope_dim], table [S, rope_dim/2]."""
    M = T.symbolic("M")
    S = T.symbolic("S")
    half = rope_dim // 2

    @T.prim_func
    def main(
        X: T.Tensor[(M, rope_dim), BF16],
        COS: T.Tensor[(S, half), FP32],
        SIN: T.Tensor[(S, half), FP32],
        POS: T.Tensor[(M,), "int32"],
        Y: T.Tensor[(M, rope_dim), BF16],
    ):
        with T.Kernel(T.ceildiv(M, rows_per_block), threads=128) as bx:
            for r, j in T.Parallel(rows_per_block, half):
                i = bx * rows_per_block + r
                a = T.Cast(FP32, X[i, 2 * j])
                b = T.Cast(FP32, X[i, 2 * j + 1])
                c = COS[POS[i], j]
                s = SIN[POS[i], j]
                Y[i, 2 * j] = T.Cast(BF16, a * c - b * s)
                Y[i, 2 * j + 1] = T.Cast(BF16, a * s + b * c)

    return main


# ---------------- python wrappers ----------------

def rmsnorm(x, w, mode="exact", eps=1e-20):
    x = x.contiguous()
    y = torch.empty_like(x)
    rmsnorm_kernel(x.size(-1), mode, float(eps))(x, w.contiguous(), y)
    return y


def swiglu(gate, up, mode="exact", limit=10.0):
    gate = gate.contiguous(); up = up.contiguous()
    y = torch.empty_like(gate)
    swiglu_kernel(gate.size(-1), mode, float(limit))(gate, up, y)
    return y


def softmax(scores, mode="exact"):
    scores = scores.contiguous().float()
    p = torch.empty_like(scores)
    softmax_kernel(scores.size(-1), mode)(scores, p)
    return p


def rope_table(x, cos, sin, pos):
    x = x.contiguous(); pos = pos.contiguous().int()
    y = torch.empty_like(x)
    rope_table_kernel(x.size(-1))(x, cos.contiguous(), sin.contiguous(), pos, y)
    return y
