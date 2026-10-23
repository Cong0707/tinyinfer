// Dense bf16 GEMV for decode (M=1), tuned for the small-K shapes a transformer head has.
//
// cuBLAS at M=1 reaches 343 GB/s on the 1024x248320 lm head on this card -- 1.38ms of a
// 7ms step, for 485MB that is a third of the model. The problem is the shape, not the
// library: K=1024 means a row is 2KB, so a thread-per-row mapping issues far too few
// outstanding memory requests to cover latency.
//
// Layout here: one BLOCK covers ROWS_PER_BLOCK rows, and within the block every warp owns
// one row while its 32 lanes stride K with R_ACC independent accumulators each. The block
// therefore has ROWS_PER_BLOCK * 32 * R_ACC loads in flight, which is what the memory
// system needs; the reduction is a warp shuffle, so the arithmetic order is fixed and the
// result is bit-reproducible.

#include <cuda_bf16.h>

#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 8
#endif
#ifndef R_ACC
#define R_ACC 4
#endif

// y[n] = sum_k x[k] * W[n, k]   x: [K] bf16, W: [N, K] bf16, y: [N] fp32
extern "C" __global__ void gemv_bf16(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ W,
                                     float* __restrict__ y,
                                     int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;              // 0 .. ROWS_PER_BLOCK-1
    const int n = blockIdx.x * ROWS_PER_BLOCK + warp;
    if (n >= N) return;

    const __nv_bfloat16* w = W + (long)n * K;
    float acc[R_ACC];
#pragma unroll
    for (int u = 0; u < R_ACC; ++u) acc[u] = 0.f;

    int k = lane;
    const int step = 32 * R_ACC;
    for (; k + (R_ACC - 1) * 32 < K; k += step) {
#pragma unroll
        for (int u = 0; u < R_ACC; ++u) {
            acc[u] = fmaf(__bfloat162float(w[k + u * 32]),
                          __bfloat162float(x[k + u * 32]), acc[u]);
        }
    }
    for (; k < K; k += 32) {
        acc[0] = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), acc[0]);
    }
#pragma unroll
    for (int u = 1; u < R_ACC; ++u) acc[0] += acc[u];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc[0] += __shfl_down_sync(0xffffffffu, acc[0], off);
    if (lane == 0) y[n] = acc[0];
}
