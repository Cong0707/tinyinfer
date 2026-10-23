// Qwen3.5 normalisation kernels: one launch per call instead of seven.
//
// Qwen3_5RMSNorm:      y = (x_fp32 * rsqrt(mean(x^2)+eps) * (1+w)).to(dtype)
// Qwen3_5RMSNormGated: y = ((w * normed.to(dtype)) * silu(gate_fp32)).to(dtype)
// The order inside RMSNormGated matters: the reference rounds the normalised value to the
// activation dtype *before* scaling by the weight, so the fp32 product must be rounded at
// the same place to stay bit-identical.

#include <cuda_bf16.h>

// one block per row, 32 lanes reduce over d
extern "C" __global__ void qwen_rms_norm_bf(const __nv_bfloat16* __restrict__ x,
                                            const float* __restrict__ w,
                                            __nv_bfloat16* __restrict__ y,
                                            int rows, int d, float eps, int plus_one) {
    const int r = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const __nv_bfloat16* xr = x + (long)r * d;
    __nv_bfloat16* yr = y + (long)r * d;
    float s = 0.f;
    for (int k = lane; k < d; k += 32) {
        float v = __bfloat162float(xr[k]);
        s += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    s = __shfl_sync(0xffffffffu, s, 0);
    const float rstd = rsqrtf(s / (float)d + eps);
    for (int k = lane; k < d; k += 32) {
        float v = __bfloat162float(xr[k]) * rstd;
        float ww = plus_one ? (1.f + w[k]) : w[k];
        yr[k] = __float2bfloat16(v * ww);
    }
}

// gated variant: y = ((w * bf16(normed)) * silu(gate_fp32)) -> bf16
extern "C" __global__ void qwen_rms_norm_gated_bf(const __nv_bfloat16* __restrict__ x,
                                                  const __nv_bfloat16* __restrict__ g,
                                                  const float* __restrict__ w,
                                                  __nv_bfloat16* __restrict__ y,
                                                  int rows, int d, float eps) {
    const int r = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const __nv_bfloat16* xr = x + (long)r * d;
    const __nv_bfloat16* gr = g + (long)r * d;
    __nv_bfloat16* yr = y + (long)r * d;
    float s = 0.f;
    for (int k = lane; k < d; k += 32) {
        float v = __bfloat162float(xr[k]);
        s += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    s = __shfl_sync(0xffffffffu, s, 0);
    const float rstd = rsqrtf(s / (float)d + eps);
    for (int k = lane; k < d; k += 32) {
        float normed = __bfloat162float(xr[k]) * rstd;
        // reference exactly:
        //   h = (h * rstd).to(dtype)          <- rounds here
        //   h = weight * h                    <- fp32 product, weight is bf16 -> fp32
        //   h = h * silu(gate.float())        <- fp32 product, rounds only at the end
        float h = __bfloat162float(__float2bfloat16(normed));
        h = __bfloat162float(__float2bfloat16(w[k])) * h;
        float gv = __bfloat162float(gr[k]);
        float silu = gv / (1.f + expf(-gv));
        yr[k] = __float2bfloat16(h * silu);
    }
}
