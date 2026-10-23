// Fused decode kernels for Qwen3.5: one launch per projection group, reading weights from
// a single contiguous per-layer block.
//
// Why a dedicated engine instead of per-module kernels: this card reaches 481 GB/s when
// the same 30MB of layer weights are read as six separate tensors and 774 GB/s when they
// are one contiguous block. The difference is L2 / DRAM stream restarts between kernels,
// not bandwidth. A layer's weights therefore live in ONE buffer, ordered by execution, so
// the whole layer streams through memory once.
//
// Encoded here:
//   qwen_gemv_group : one matmul over a slice of the block (any (N,K) at a fixed offset)
//   qwen_swiglu_q8  : silu(gate)*up followed by block-32 quantisation, writing fp8 + e8m0
//                     so the down projection can run on the int8 tensor cores path via
//                     bf16 accumulate (the reference does the same rounding)
//   qwen_gdn_step   : gated delta rule, one token (already in gdn.cu)
//   qwen_conv4      : depthwise 4-tap causal conv (already in gdn.cu)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// y[n] = sum_k x[k] * W[n0 + n, k]  with W taken as [rows, K] starting at n0.
extern "C" __global__ void qwen_gemv_group(const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ W,
                                           float* __restrict__ y,
                                           int n0, int nrows, int K, int Wstride) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int r = blockIdx.x * 8 + warp;
    if (r >= nrows) return;
    // n0 is a FLAT element offset into the packed layer block, not a row index: the block
    // concatenates weights of different K, so rows are not uniformly spaced from it.
    const __nv_bfloat16* w = W + n0 + (long)r * Wstride;

    float acc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) acc[u] = 0.f;
    int k = lane;
    for (; k + 96 < K; k += 128) {
#pragma unroll
        for (int u = 0; u < 4; ++u)
            acc[u] = fmaf(__bfloat162float(w[k + u * 32]),
                          __bfloat162float(x[k + u * 32]), acc[u]);
    }
    for (; k < K; k += 32)
        acc[0] = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), acc[0]);
    acc[0] += acc[1] + acc[2] + acc[3];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc[0] += __shfl_down_sync(0xffffffffu, acc[0], off);
    if (lane == 0) y[n0 + r] = acc[0];
}

// SwiGLU + per-32 block quantisation to fp8 e4m3 with an e8m0 scale, matching
// official_kernel_ref.act_quant: s = 2^ceil(log2(amax/448)), q = clamp(v/s).
extern "C" __global__ void qwen_swiglu_q8(const float* __restrict__ gate_up,  // [2*inter]
                                          unsigned char* __restrict__ out_q, // [inter]
                                          unsigned char* __restrict__ out_s, // [inter/32]
                                          int inter, float limit) {
    const int b = blockIdx.x;             // one block per 32-value group
    const int lane = threadIdx.x & 31;
    const int i = b * 32 + lane;
    if (i >= inter) return;
    float g = gate_up[i];
    float u = gate_up[inter + i];
    if (limit > 0.f) { u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); }
    float v = (g / (1.f + expf(-g))) * u;
    // block max over the warp
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    amax = fmaxf(amax, 1e-4f);
    // 2^ceil(log2(amax/448))
    float ratio = amax / 448.0f;
    int e; frexpf(ratio, &e);             // ratio = m * 2^e, 0.5 <= m < 1
    if (ratio > ldexpf(1.0f, e)) e += 1;  // ceil to the next power of two
    float scale = ldexpf(1.0f, e);
    float q = v / scale;
    q = fminf(fmaxf(q, -448.0f), 448.0f);
    out_q[i] = (unsigned char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE,
                                                    __NV_E4M3);
    if (lane == 0) out_s[b] = (unsigned char)(e + 127);
}

// Whole-layer GEMV: every projection of one layer in a single launch.
//
// The win measured earlier was 774 GB/s for one contiguous read of the layer block vs
// 481 GB/s for the same bytes split across six tensors -- but that only holds if ONE
// launch walks the block. Calling a per-projection kernel eight times re-enters the
// stream eight times and lands at 139 GB/s, slower than torch. So the layer's projection
// list is passed as a small descriptor array and grid.y selects which one a block serves.
//
// desc: [n_groups][4] int32 -> {flat_offset, rows, K, out_offset}
extern "C" __global__ void qwen_gemv_layer(const __nv_bfloat16* __restrict__ W,
                                           const int* __restrict__ desc,
                                           const __nv_bfloat16* const* __restrict__ xs,
                                           float* __restrict__ out) {
    const int g = blockIdx.y;                 // which projection
    const int off = desc[g * 4 + 0];
    const int rows = desc[g * 4 + 1];
    const int K = desc[g * 4 + 2];
    const int ooff = desc[g * 4 + 3];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int r = blockIdx.x * 8 + warp;
    if (r >= rows) return;

    const __nv_bfloat16* x = xs[g];
    const __nv_bfloat16* w = W + off + (long)r * K;
    float acc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) acc[u] = 0.f;
    int k = lane;
    for (; k + 96 < K; k += 128) {
#pragma unroll
        for (int u = 0; u < 4; ++u)
            acc[u] = fmaf(__bfloat162float(w[k + u * 32]),
                          __bfloat162float(x[k + u * 32]), acc[u]);
    }
    for (; k < K; k += 32)
        acc[0] = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), acc[0]);
    acc[0] += acc[1] + acc[2] + acc[3];
#pragma unroll
    for (int off2 = 16; off2 > 0; off2 >>= 1)
        acc[0] += __shfl_down_sync(0xffffffffu, acc[0], off2);
    if (lane == 0) out[ooff + r] = acc[0];
}

// Whole-layer GEMV over a flat grid: block -> projection is a prefix-sum lookup, so short
// projections (rows=16) do not leave most of the grid idle the way a rectangular
// (max_rows, n_groups) grid does.
//
// desc: [n_groups][4] int32 {flat_offset, rows, K, out_offset}, ordered by execution.
// blocks_of(g) = ceil(desc[g*4+1] / 8); starts are precomputed into desc[g*4+3] caller-side
// as a prefix sum of block counts in \`bstarts\`.
extern "C" __global__ void qwen_gemv_layer_flat(const __nv_bfloat16* __restrict__ W,
                                                const int* __restrict__ desc,
                                                const int* __restrict__ bstarts,
                                                const __nv_bfloat16* const* __restrict__ xs,
                                                float* __restrict__ out,
                                                int n_groups, int max_rows) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row_in_group = (blockIdx.x - 0) * 8 + warp;   // resolved per group below

    // find the group this block belongs to: bstarts[g] <= blockIdx.x < bstarts[g+1]
    int g = 0;
    int bstart = 0;
    for (int i = 0; i < n_groups; ++i) {
        int lo = bstarts[i];
        int hi = bstarts[i + 1];
        if (blockIdx.x >= lo && blockIdx.x < hi) { g = i; bstart = lo; break; }
    }
    const int r = (blockIdx.x - bstart) * 8 + warp;
    const int rows = desc[g * 4 + 1];
    if (r >= rows) return;
    const int off = desc[g * 4 + 0];
    const int K = desc[g * 4 + 2];
    const int ooff = desc[g * 4 + 3];

    const __nv_bfloat16* x = xs[g];
    const __nv_bfloat16* w = W + off + (long)r * K;
    float acc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) acc[u] = 0.f;
    int k = lane;
    for (; k + 96 < K; k += 128) {
#pragma unroll
        for (int u = 0; u < 4; ++u)
            acc[u] = fmaf(__bfloat162float(w[k + u * 32]),
                          __bfloat162float(x[k + u * 32]), acc[u]);
    }
    for (; k < K; k += 32)
        acc[0] = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), acc[0]);
    acc[0] += acc[1] + acc[2] + acc[3];
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1)
        acc[0] += __shfl_down_sync(0xffffffffu, acc[0], o2);
    if (lane == 0) out[ooff + r] = acc[0];
}

// Whole-layer GEMV, tuned block shape: one block covers 32 rows (4 warps, each warp 8 rows
// in a strided loop) so a block reads 32*K*2 bytes = 64KB at K=1024 instead of 16KB.
#ifndef LAYER_ROWS_PER_WARP
#define LAYER_ROWS_PER_WARP 8
#endif
#ifndef LAYER_WARPS
#define LAYER_WARPS 4
#endif
#define LAYER_BLOCK_ROWS (LAYER_ROWS_PER_WARP * LAYER_WARPS)

extern "C" __global__ void qwen_gemv_layer2(const __nv_bfloat16* __restrict__ W,
                                            const int* __restrict__ desc,
                                            const int* __restrict__ bstarts,
                                            const __nv_bfloat16* const* __restrict__ xs,
                                            float* __restrict__ out,
                                            int n_groups, int unused) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int g = 0, bstart = 0;
    for (int i = 0; i < n_groups; ++i) {
        if (blockIdx.x >= bstarts[i] && blockIdx.x < bstarts[i + 1]) {
            g = i; bstart = bstarts[i]; break;
        }
    }
    const int rows = desc[g * 4 + 1];
    const int off = desc[g * 4 + 0];
    const int K = desc[g * 4 + 2];
    const int ooff = desc[g * 4 + 3];
    const __nv_bfloat16* x = xs[g];

    const int base = (blockIdx.x - bstart) * LAYER_BLOCK_ROWS + warp * LAYER_ROWS_PER_WARP;
    for (int j = 0; j < LAYER_ROWS_PER_WARP; ++j) {
        const int r = base + j;
        if (r >= rows) return;
        const __nv_bfloat16* w = W + off + (long)r * K;
        float acc[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) acc[u] = 0.f;
        int k = lane;
        for (; k + 96 < K; k += 128) {
#pragma unroll
            for (int u = 0; u < 4; ++u)
                acc[u] = fmaf(__bfloat162float(w[k + u * 32]),
                              __bfloat162float(x[k + u * 32]), acc[u]);
        }
        for (; k < K; k += 32)
            acc[0] = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), acc[0]);
        acc[0] += acc[1] + acc[2] + acc[3];
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1)
            acc[0] += __shfl_down_sync(0xffffffffu, acc[0], o2);
        if (lane == 0) out[ooff + r] = acc[0];
    }
}
