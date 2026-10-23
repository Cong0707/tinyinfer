// Deterministic Hyper-Connection kernels (no atomics: one thread per output element,
// sequential accumulation, so the result is bit-reproducible run to run).
//
// hc_pre :  y[m, d]        = sum_h pre[m, h] * x[m, h, d]
// hc_post:  y[m, a, b, d]  = post[m, a, b] * x[m, b, d]
//                          + sum_j comb[m, a, j] * resid[m, j, d]
// The second form is what the eager expression
//     post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * res.unsqueeze(-2), 2)
// broadcasts to: post[i,j] and comb[i,j] index the first two HC axes of the result, the
// value axis comes from x[j] and resid[j]. Replacing it with one kernel removes six
// launches per call (unsqueeze x2, mul x2, sum, cast) twice per layer.

#include <cuda_bf16.h>

extern "C" __global__ void hc_pre_f(const float* __restrict__ x,       // [M, hc, D]
                                    const float* __restrict__ pre,     // [M, hc]
                                    float* __restrict__ out,           // [M, D]
                                    int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * D) return;
    int m = (int)(i / D);
    int d = (int)(i - (long)m * D);
    const float* xm = x + (long)m * hc * D + d;
    const float* pm = pre + (long)m * hc;
    float acc = 0.f;
    for (int h = 0; h < hc; ++h) acc += pm[h] * xm[(long)h * D];
    out[i] = acc;
}

extern "C" __global__ void hc_pre_bf(const __nv_bfloat16* __restrict__ x,
                                     const float* __restrict__ pre,
                                     __nv_bfloat16* __restrict__ out,
                                     int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * D) return;
    int m = (int)(i / D);
    int d = (int)(i - (long)m * D);
    const __nv_bfloat16* xm = x + (long)m * hc * D + d;
    const float* pm = pre + (long)m * hc;
    float acc = 0.f;
    for (int h = 0; h < hc; ++h) acc += pm[h] * __bfloat162float(xm[(long)h * D]);
    out[i] = __float2bfloat16(acc);
}

// out is [M, hc, hc, D]: a and b are the two HC axes.
extern "C" __global__ void hc_post_f(const float* __restrict__ x,        // [M, hc, D]
                                     const float* __restrict__ resid,    // [M, hc, D]
                                     const float* __restrict__ post,     // [M, hc, hc]
                                     const float* __restrict__ comb,     // [M, hc, hc]
                                     float* __restrict__ out,            // [M, hc, hc, D]
                                     int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * hc * hc * D) return;
    int d = (int)(i % D);
    long t = i / D;
    int b = (int)(t % hc);
    t /= hc;
    int a = (int)(t % hc);
    int m = (int)(t / hc);
    const float* xm = x + ((long)m * hc + b) * D + d;
    const float* rm = resid + (long)m * hc * D + d;
    const float* cm = comb + ((long)m * hc + a) * hc;
    float acc = post[((long)m * hc + a) * hc + b] * xm[0];
    for (int j = 0; j < hc; ++j) acc += cm[j] * rm[(long)j * D];
    out[i] = acc;
}

extern "C" __global__ void hc_post_bf(const __nv_bfloat16* __restrict__ x,
                                      const __nv_bfloat16* __restrict__ resid,
                                      const float* __restrict__ post,
                                      const float* __restrict__ comb,
                                      __nv_bfloat16* __restrict__ out,
                                      int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * hc * hc * D) return;
    int d = (int)(i % D);
    long t = i / D;
    int b = (int)(t % hc);
    t /= hc;
    int a = (int)(t % hc);
    int m = (int)(t / hc);
    const __nv_bfloat16* xm = x + ((long)m * hc + b) * D + d;
    const __nv_bfloat16* rm = resid + (long)m * hc * D + d;
    const float* cm = comb + ((long)m * hc + a) * hc;
    float acc = post[((long)m * hc + a) * hc + b] * __bfloat162float(xm[0]);
    for (int j = 0; j < hc; ++j) acc += cm[j] * __bfloat162float(rm[(long)j * D]);
    out[i] = __float2bfloat16(acc);
}
// hc_mixes: row-normalised Hyper-Connection coefficient projection.
//
//   r   = rsqrt(mean_k(x[m,k]^2) + eps)
//   out[m, j] = r * sum_k x[m, k] * FN[j, k]
//
// The eager form is flatten + square + mean + add + rsqrt + linear + mul: seven launches
// for a 24x20480 projection, twice per layer. One block per row, one warp-wide reduction
// for the norm, and the projection rows are split across the block. No atomics: each
// output element is owned by one thread and accumulates sequentially, so the result is
// reproducible and matches the eager arithmetic order (fp32 accumulate).

#include <cuda_bf16.h>

#define HC_BLOCK 256

extern "C" __global__ void hc_mixes_bf(const __nv_bfloat16* __restrict__ x,  // [M, K]
                                       const float* __restrict__ fn,        // [J, K]
                                       float* __restrict__ out,             // [M, J]
                                       int M, int K, int J, float eps,
                                       int inv_k_is_pow2) {
    int m = blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* xm = x + (long)m * K;

    __shared__ float s_norm;
    __shared__ float s_x[HC_BLOCK];
    int tid = threadIdx.x;

    // ---- row norm: sum of squares ----
    float part = 0.f;
    for (int k = tid; k < K; k += HC_BLOCK) {
        float v = __bfloat162float(xm[k]);
        part += v * v;
    }
    // warp tree, then one shared slot per warp
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
    __shared__ float s_warp[HC_BLOCK / 32];
    if ((tid & 31) == 0) s_warp[tid >> 5] = part;
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
        for (int w = 0; w < HC_BLOCK / 32; ++w) s += s_warp[w];
        s_norm = rsqrtf(s / (float)K + eps);
    }
    __syncthreads();
    float r = s_norm;

    // ---- projection: one output row per warp, K walked in HC_BLOCK-wide tiles ----
    for (int j = tid / 32; j < J; j += HC_BLOCK / 32) {
        const float* fnj = fn + (long)j * K;
        float acc = 0.f;
        // one warp per output row; the 32 lanes stride the whole K, then the warp tree
        // combines them. (Stepping k by the block width instead only covered 32 of every
        // 256 elements -- the projection came out 8x too small.)
        for (int k = tid & 31; k < K; k += 32) {
            acc += __bfloat162float(xm[k]) * fnj[k];
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if ((tid & 31) == 0) out[(long)m * J + j] = acc * r;
    }
}

// fp32 activation variant (tier paths that carry fp32 h)
extern "C" __global__ void hc_mixes_f(const float* __restrict__ x,
                                      const float* __restrict__ fn,
                                      float* __restrict__ out,
                                      int M, int K, int J, float eps,
                                      int inv_k_is_pow2) {
    int m = blockIdx.x;
    if (m >= M) return;
    const float* xm = x + (long)m * K;
    __shared__ float s_norm;
    int tid = threadIdx.x;
    float part = 0.f;
    for (int k = tid; k < K; k += HC_BLOCK) {
        float v = xm[k];
        part += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
    __shared__ float s_warp[HC_BLOCK / 32];
    if ((tid & 31) == 0) s_warp[tid >> 5] = part;
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
        for (int w = 0; w < HC_BLOCK / 32; ++w) s += s_warp[w];
        s_norm = rsqrtf(s / (float)K + eps);
    }
    __syncthreads();
    float r = s_norm;
    for (int j = tid / 32; j < J; j += HC_BLOCK / 32) {
        const float* fnj = fn + (long)j * K;
        float acc = 0.f;
        for (int k = tid & 31; k < K; k += 32) {
            acc += xm[k] * fnj[k];
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if ((tid & 31) == 0) out[(long)m * J + j] = acc * r;
    }
}
