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
// Bit-exactness matters here: this replaces two kernels in a 40-layer chain, and one bf16
// ulp of difference per layer compounds into a different argmax within five layers. So the
// reduction structure is copied from the kernel it replaces (norm/vllm_rms_norm.cu)
// instead of being "equivalent": 8-wide vectors, thread i owns vector i, i + blockDim, ...,
// the same shuffle tree, and the caller passes the same block size (min(hidden/8, 1024)).
// The collapsed row is rounded to bf16 exactly where hc_pre's output tensor would have
// landed, then carried in fp32 -- that round trip is not optional, it is worth one bf16 ulp.
//
// The unfused kernels (hc.cu: hc_pre_bf, norm/: vllm_rms_norm.cu) stay in the tree: this
// file is an alternative path, selected with TI_FUSE=1, and the tests compare the two.
#include <cuda_bf16.h>

namespace {

constexpr int kVec = 8;  // 16 bytes of bf16

// Same tree as vllm_rms_norm.cu, for the same reason (no cuda toolkit headers here).
__device__ __forceinline__ float block_reduce_sum(float v, float* s) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (threadIdx.x < nwarps) ? s[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
        if (threadIdx.x == 0) s[0] = t;
    }
    __syncthreads();
    return s[0];
}

}  // namespace

extern "C" __global__ void hc_pre_rms_norm_bf(
    const __nv_bfloat16* __restrict__ x,       // [M, hc, D]
    const float* __restrict__ pre_mix,         // [M, hc]
    const __nv_bfloat16* __restrict__ weight,  // [D]
    __nv_bfloat16* __restrict__ out,           // [M, D]
    int M, int hc, int D, float eps) {
    // row buffer + warp scratch + the broadcast variance, all from one dynamic block
    extern __shared__ float smem[];
    float* s_h = smem;               // D floats
    float* s_warp = smem + D;        // 32 floats
    float* s_var = s_warp + 32;      // 1 float

    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const __nv_bfloat16* xm = x + (long)m * hc * D;
    const float* pm = pre_mix + (long)m * hc;

    float variance = 0.f;
    const int nvec = D / kVec;
    for (int i = tid; i < nvec; i += nthreads) {
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
            const int d = i * kVec + j;
            float acc = 0.f;
            for (int c = 0; c < hc; ++c)
                acc += pm[c] * __bfloat162float(xm[(long)c * D + d]);
            acc = __bfloat162float(__float2bfloat16(acc));   // hc_pre's output step
            s_h[d] = acc;
            variance += acc * acc;
        }
    }
    for (int d = nvec * kVec + tid; d < D; d += nthreads) {
        float acc = 0.f;
        for (int c = 0; c < hc; ++c)
            acc += pm[c] * __bfloat162float(xm[(long)c * D + d]);
        acc = __bfloat162float(__float2bfloat16(acc));
        s_h[d] = acc;
        variance += acc * acc;
    }

    const float total = block_reduce_sum(variance, s_warp);
    if (tid == 0) *s_var = rsqrtf(total / (float)D + eps);
    __syncthreads();
    const float r = *s_var;

    for (int i = tid; i < nvec; i += nthreads) {
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
            const int d = i * kVec + j;
            out[(long)m * D + d] =
                __float2bfloat16(s_h[d] * r * __bfloat162float(weight[d]));
        }
    }
    for (int d = nvec * kVec + tid; d < D; d += nthreads)
        out[(long)m * D + d] =
            __float2bfloat16(s_h[d] * r * __bfloat162float(weight[d]));
}
