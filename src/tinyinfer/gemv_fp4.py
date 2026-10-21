"""Decode-only FP4 GEMV (sm_120 / tilelang 0.1.15).

Same one-thread-per-(row,k-lane) layout as gemv_fp8: for M=1 the official fp4_gemm
uses block_M=32 (31/32 wasted). Weights are packed E2M1 (2/byte) with one e8m0
scale per 32 along K; the dot applies act scale and weight scale inline.
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
def gemv_fp4_kernel_v2(N: int, K: int, rows: int = 16):
    nb = K // 32

    @T.prim_func
    def main(
        A: T.Tensor[(K,), FP8],
        AS: T.Tensor[(nb,), FP32],
        W: T.Tensor[(N, K), "float4_e2m1fn"],
        WS: T.Tensor[(N, nb), FP32],
        Y: T.Tensor[(N,), BF16],
    ):
        with T.Kernel(T.ceildiv(N, rows), threads=rows * 32) as bx:
            acc = T.alloc_fragment((rows, 32), FP32)
            ysum = T.alloc_fragment((rows,), FP32)
            n0 = bx * rows
            for r, l in T.Parallel(rows, 32):
                acc[r, l] = 0.0
            for kb in T.serial(nb):
                for r, l in T.Parallel(rows, 32):
                    n = n0 + r
                    a = T.Cast(FP32, A[kb * 32 + l])
                    wv = T.Cast(FP32, T.Cast(FP8, W[n, kb * 32 + l]))
                    acc[r, l] += a * wv * AS[kb] * WS[n, kb]
            T.reduce_sum(acc, ysum, dim=1)
            for r in T.Parallel(rows):
                if n0 + r < N:
                    Y[n0 + r] = T.Cast(BF16, ysum[r])

    return main


def gemv_fp4(x: torch.Tensor, w_fp4: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """x [1,K] bf16 activation, w fp4 [N,K] packed, scale [N/32,K/32] ue8m0 -> [1,N] bf16."""
    from .gemm import _bf16_default, SCALE_DTYPE, SCALE_FMT
    import kernel as _k

    n, k2 = w_fp4.shape
    k = k2 * 2  # packed E2M1: two values per byte
    with _bf16_default():
        aq, asc = _k.act_quant(x.reshape(1, k), 32, SCALE_FMT, SCALE_DTYPE)
    out = torch.empty(n, device=x.device, dtype=torch.bfloat16)
    ws = _scale(w_scale)
    if ws.size(0) != n:  # packed [N/32, nb] -> broadcast rows to [N, nb] (ABI wants N)
        ws = ws.view(-1, ws.size(-1)).repeat_interleave(32, 0)[:n]
    gemv_fp4_kernel_v2(n, k)(aq.reshape(-1), asc.reshape(-1).float(), w_fp4, ws, out)
    return out.view(1, n)


def _scale(s: torch.Tensor) -> torch.Tensor:
    if s.dtype == torch.float32:
        return s
    if s.dtype == torch.uint8:  # HF stores ue8m0 scales as raw bytes
        return s.view(torch.float8_e8m0fnu).float()
    if s.dtype == torch.float8_e8m0fnu:
        return s.float()
    return s.float()
