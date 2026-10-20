"""Fused Hyper-Connections kernels (sm_120 / tilelang 0.1.15).

fused_hc_mixes : row rsqrt-normalize x [M, hc*D] + small GEMM [24, hc*D] -> raw mixes [M,24]
fused_sinkhorn : pre/post sigmoid + comb softmax + `iters` doubly-stochastic normalizations
                 (exactly the math of tinyinfer.hc.hc_split_sinkhorn; mixes are raw, affine is applied here)
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
HC = 4
MIX = 24  # (2 + hc) * hc


@tilelang.jit(pass_configs=PASS)
def hc_mixes_kernel(HD: int, mix: int, eps: float):
    M = T.symbolic("M")

    @T.prim_func
    def main(
        X: T.Tensor[(M, HD), BF16],
        FN: T.Tensor[(mix, HD), FP32],
        OUT: T.Tensor[(M, mix), FP32],
    ):
        with T.Kernel(M, threads=256) as bx:
            xs = T.alloc_shared((HD,), BF16)
            buf = T.alloc_fragment((HD,), FP32)
            ss = T.alloc_fragment((1,), FP32)
            dot = T.alloc_fragment((1,), FP32)
            T.copy(X[bx, 0], xs)
            for k in T.Parallel(HD):
                buf[k] = T.Cast(FP32, xs[k])
            for k in T.Parallel(HD):
                buf[k] = buf[k] * buf[k]
            T.reduce_sum(buf, ss, dim=0)
            r = T.rsqrt(ss[0] / HD + eps)
            for j in T.serial(mix):
                for k in T.Parallel(HD):
                    buf[k] = T.Cast(FP32, xs[k]) * FN[j, k]
                T.reduce_sum(buf, dot, dim=0)
                OUT[bx, j] = dot[0] * r

    return main


@tilelang.jit(pass_configs=PASS)
def sinkhorn_kernel(hc: int, iters: int, eps: float):
    mix = (2 + hc) * hc
    M = T.symbolic("M")

    @T.prim_func
    def main(
        MIXES: T.Tensor[(M, mix), FP32],
        SCALE: T.Tensor[(3,), FP32],
        BASE: T.Tensor[(mix,), FP32],
        PRE: T.Tensor[(M, hc), FP32],
        POST: T.Tensor[(M, hc), FP32],
        COMB: T.Tensor[(M, hc, hc), FP32],
    ):
        with T.Kernel(M, threads=64) as i:
            m = T.alloc_shared((mix,), FP32)
            comb = T.alloc_fragment((hc, hc), FP32)
            row_sum = T.alloc_fragment((hc,), FP32)
            col_sum = T.alloc_fragment((hc,), FP32)
            row_max = T.alloc_fragment((hc,), FP32)
            T.copy(MIXES[i, :], m)
            for j in T.Parallel(hc):
                PRE[i, j] = 1.0 / (1.0 + T.exp(-(m[j] * SCALE[0] + BASE[j]))) + eps
            for j in T.Parallel(hc):
                POST[i, j] = 2.0 / (1.0 + T.exp(-(m[hc + j] * SCALE[1] + BASE[hc + j])))
            for j, k in T.Parallel(hc, hc):
                comb[j, k] = m[2 * hc + j * hc + k] * SCALE[2] + BASE[2 * hc + j * hc + k]
            T.reduce_max(comb, row_max, dim=1, clear=True)
            for j, k in T.Parallel(hc, hc):
                comb[j, k] = T.exp(comb[j, k] - row_max[j])
            T.reduce_sum(comb, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb[j, k] = comb[j, k] / row_sum[j] + eps
            T.reduce_sum(comb, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb[j, k] = comb[j, k] / (col_sum[k] + eps)
            for _ in T.serial(iters - 1):
                T.reduce_sum(comb, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb[j, k] = comb[j, k] / (row_sum[j] + eps)
                T.reduce_sum(comb, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb[j, k] = comb[j, k] / (col_sum[k] + eps)
            T.copy(comb, COMB[i, :, :])

    return main


@tilelang.jit(pass_configs=PASS)
def hc_pre_kernel(D: int, hc: int):
    """collapse: OUT[m,d] = sum_h PRE[m,h] * X[m,h,d]."""
    M = T.symbolic("M")

    @T.prim_func
    def main(
        X: T.Tensor[(M, hc, D), BF16],
        PRE: T.Tensor[(M, hc), FP32],
        OUT: T.Tensor[(M, D), BF16],
    ):
        with T.Kernel(M, threads=256) as bx:
            xs = T.alloc_shared((hc, D), BF16)
            acc = T.alloc_fragment((D,), FP32)
            T.copy(X[bx, 0, 0], xs)
            for d in T.Parallel(D):
                acc[d] = 0.0
            for h in T.serial(hc):
                for d in T.Parallel(D):
                    acc[d] += PRE[bx, h] * T.Cast(FP32, xs[h, d])
            for d in T.Parallel(D):
                OUT[bx, d] = T.Cast(BF16, acc[d])

    return main


@tilelang.jit(pass_configs=PASS)
def hc_post_kernel(D: int, hc: int):
    """expand: OUT[m,h,d] = POST[m,h]*X[m,d] + sum_h2 COMB[m,h,h2]*RES[m,h2,d]."""
    M = T.symbolic("M")

    @T.prim_func
    def main(
        X: T.Tensor[(M, D), BF16],
        RES: T.Tensor[(M, hc, D), BF16],
        POST: T.Tensor[(M, hc), FP32],
        COMB: T.Tensor[(M, hc, hc), FP32],
        OUT: T.Tensor[(M, hc, D), BF16],
    ):
        with T.Kernel(M, threads=256) as bx:
            xs = T.alloc_shared((D,), BF16)
            rs = T.alloc_shared((hc, D), BF16)
            acc = T.alloc_fragment((D,), FP32)
            T.copy(X[bx, 0], xs)
            T.copy(RES[bx, 0, 0], rs)
            for h in T.serial(hc):
                for d in T.Parallel(D):
                    acc[d] = POST[bx, h] * T.Cast(FP32, xs[d])
                for h2 in T.serial(hc):
                    for d in T.Parallel(D):
                        acc[d] += COMB[bx, h2, h] * T.Cast(FP32, rs[h2, d])
                for d in T.Parallel(D):
                    OUT[bx, h, d] = T.Cast(BF16, acc[d])

    return main


# ---------------- wrappers ----------------

def fused_hc_mixes(x: torch.Tensor, hc_fn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x [B,S,hc,D] or [M,hc*D] bf16; hc_fn [24, hc*D] fp32 -> raw mixes [M,24] fp32."""
    x2 = x.reshape(-1, x.shape[-2] * x.shape[-1]).contiguous()
    m, hd = x2.shape
    out = torch.empty(m, MIX, device=x.device, dtype=torch.float32)
    hc_mixes_kernel(hd, MIX, float(eps))(x2, hc_fn.float().contiguous(), out)
    return out


def fused_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                   hc: int = HC, iters: int = 20, eps: float = 1e-6):
    """mixes [B,S,24] raw (linear*rsqrt) -> pre [B,S,4], post [B,S,4], comb [B,S,4,4] fp32."""
    shape = mixes.shape[:-1]
    m2 = mixes.reshape(-1, MIX).contiguous().float()
    m = m2.shape[0]
    pre = torch.empty(m, hc, device=mixes.device, dtype=torch.float32)
    post = torch.empty(m, hc, device=mixes.device, dtype=torch.float32)
    comb = torch.empty(m, hc, hc, device=mixes.device, dtype=torch.float32)
    sinkhorn_kernel(hc, int(iters), float(eps))(
        m2, hc_scale.float().contiguous(), hc_base.float().contiguous(), pre, post, comb)
    return pre.view(*shape, hc), post.view(*shape, hc), comb.view(*shape, hc, hc)

def fused_hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """x [B,S,hc,D] bf16, pre [B,S,hc] fp32 -> [B,S,D] bf16."""
    shape = x.shape[:-2]
    hc, d = x.shape[-2], x.shape[-1]
    x3 = x.reshape(-1, hc, d).contiguous()
    out = torch.empty(x3.shape[0], d, device=x.device, dtype=torch.bfloat16)
    hc_pre_kernel(d, hc)(x3, pre.reshape(-1, hc).float().contiguous(), out)
    return out.view(*shape, d)


def fused_hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """x [B,S,D] bf16, residual [B,S,hc,D] bf16, post [B,S,hc], comb [B,S,hc,hc] -> [B,S,hc,D] bf16."""
    shape = residual.shape[:-2]
    hc, d = residual.shape[-2], residual.shape[-1]
    x2 = x.reshape(-1, d).contiguous()
    r3 = residual.reshape(-1, hc, d).contiguous()
    out = torch.empty(r3.shape[0], hc, d, device=x.device, dtype=torch.bfloat16)
    hc_post_kernel(d, hc)(x2, r3, post.reshape(-1, hc).float().contiguous(),
                          comb.reshape(-1, hc, hc).float().contiguous(), out)
    return out.view(*shape, hc, d)