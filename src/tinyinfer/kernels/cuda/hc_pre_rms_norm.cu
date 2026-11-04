// hc_pre + rms_norm fused into one kernel.
//
// The two-step form (hc_pre -> rms_norm) launches twice per normalisation point, twice
// per layer, 40 layers: 160 launches per decode token on this engine. They are fusible
// because hc_pre produces the vector rms_norm immediately consumes, with no other reader
// in between:
//
//   h[m, d] = sum_c pre_mix[m, c] * x[m, c, d]        // hyper-connection collapse
//   r       = rsqrt(mean_d(h[m, :]^2) + eps)
//   y[m, d] = h[m, d] * r * w[d]                     // scale by the norm weight
//
// One block per row; the collapsed row lives in dynamic shared memory between the two
// passes so the input is read once instead of twice. Reduction is a warp-shuffle tree in
// fp32, matching the reference accumulation order closely enough to stay within one bf16
// ulp of the unfused pair.
//
// The unfused kernels (hc.cu: hc_pre_bf, norm/: vllm_rms_norm.cu) stay in the tree: this
// file is an alternative path, selected at call time, and the tests compare the two.
#include <cuda_bf16.h>

#define HPR_BLOCK 256

__device__ __forceinline__ float hpr_warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

extern "C" __global__ void hc_pre_rms_norm_bf(
    const __nv_bfloat16* __restrict__ x,       // [M, hc, D]
    const float* __restrict__ pre_mix,         // [M, hc]
    const __nv_bfloat16* __restrict__ weight,  // [D]
    __nv_bfloat16* __restrict__ out,           // [M, D]
    int M, int hc, int D, float eps) {
    extern __shared__ float s_h[];   // one row of the collapsed activation, fp32

    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const __nv_bfloat16* xm = x + (long)m * hc * D;
    const float* pm = pre_mix + (long)m * hc;

    float local_sq = 0.f;
    for (int d = tid; d < D; d += nthreads) {
        float acc = 0.f;
        for (int c = 0; c < hc; ++c)
            acc += pm[c] * __bfloat162float(xm[(long)c * D + d]);
        // Round to bf16 here, exactly where the unfused path materialises hc_pre's
        // output, then carry it in fp32. Skipping this round trip is measurable: the
        // fused kernel was off by a constant 3.125e-2 against the two-kernel chain on
        // N(0,1) inputs, which is this bf16 ulp and nothing else.
        acc = __bfloat162float(__float2bfloat16(acc));
        s_h[d] = acc;
        local_sq += acc * acc;
    }

    // block-wide sum of squares -> mean
    local_sq = hpr_warp_sum(local_sq);
    __shared__ float s_warp[32];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) s_warp[warp] = local_sq;
    __syncthreads();
    if (warp == 0) {
        const int nwarps = (nthreads + 31) >> 5;
        float v = (lane < nwarps) ? s_warp[lane] : 0.f;
        v = hpr_warp_sum(v);
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();

    const float mean = s_warp[0] / (float)D;
    const float r = rsqrtf(mean + eps);

    for (int d = tid; d < D; d += nthreads)
        out[(long)m * D + d] = __float2bfloat16(s_h[d] * r * __bfloat162float(weight[d]));
}
