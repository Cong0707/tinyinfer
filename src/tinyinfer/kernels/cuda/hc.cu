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
// out[m, o, d] = post[m, o] * x[m, o, d] + sum_i comb[m, i, o] * resid[m, i, d]
//
// This is the Hyper-Connection residual update, and it is the same arithmetic as the
// reference mhc_post_fwd:
//     x_out[n, o, h] = c[n, o] * d[n, h] + sum_i a[n, i, o] * b[n, i, h]
// Note the index order on comb: the sum runs over the *source* hyper-connection index i
// and o is the destination, which is why it is comb[m, i, o] and not comb[m, o, i].
// An earlier version of this kernel carried a fourth output dimension (b), treating post
// as [M, hc, hc] and reading x[m, b, d] -- that matches neither the caller's [M, hc]
// post nor the reference, and the resulting reshape threw on every call, so the fast
// path was dead and every token fell through to the torch expression in hc.py.
// x is the layer output BEFORE it is broadcast into hyper-connection space: it has shape
// [M, D], not [M, hc, D]. residual is the [M, hc, D] one. The caller's guard used to
// require residual.shape == x.shape, which can never hold for these two, so the fast path
// was unreachable on top of the index bug fixed here.
extern "C" __global__ void hc_post_f(const float* __restrict__ x,        // [M, D]
                                     const float* __restrict__ resid,    // [M, hc, D]
                                     const float* __restrict__ post,     // [M, hc]
                                     const float* __restrict__ comb,     // [M, hc, hc]
                                     float* __restrict__ out,            // [M, hc, D]
                                     int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * hc * D) return;
    int d = (int)(i % D);
    long t = i / D;
    int o = (int)(t % hc);
    int m = (int)(t / hc);
    float acc = post[(long)m * hc + o] * x[(long)m * D + d];
    // Mix the hyper-connection states: out[m,o] += comb[m,i,o] * resid[m,i], summed over
    // the SOURCE index i. This matches the reference mhc_post_fwd
    // (a[n,i_mhci,i_mhco] * b[n,i_mhci,h]) and is what "residual update" has to mean --
    // the transpose reading (sum over the destination) leaves every o multiplying only its
    // own residual, which is not a mixing operation at all.
    for (int i_src = 0; i_src < hc; ++i_src)
        acc += comb[((long)m * hc + i_src) * hc + o] *
               resid[((long)m * hc + i_src) * D + d];
    out[i] = acc;
}

extern "C" __global__ void hc_post_bf(const __nv_bfloat16* __restrict__ x,        // [M, D]
                                      const __nv_bfloat16* __restrict__ resid,    // [M, hc, D]
                                      const float* __restrict__ post,             // [M, hc]
                                      const float* __restrict__ comb,             // [M, hc, hc]
                                      __nv_bfloat16* __restrict__ out,            // [M, hc, D]
                                      int M, int hc, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * hc * D) return;
    int d = (int)(i % D);
    long t = i / D;
    int o = (int)(t % hc);
    int m = (int)(t / hc);
    float acc = post[(long)m * hc + o] * __bfloat162float(x[(long)m * D + d]);
    for (int i_src = 0; i_src < hc; ++i_src)
        acc += comb[((long)m * hc + i_src) * hc + o] *
               __bfloat162float(resid[((long)m * hc + i_src) * D + d]);
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
