// Dense bf16 GEMV for decode (M=1), built to keep the memory system saturated.
//
// cuBLAS' GEMV path at M=1 reached 436 GB/s on a 7MB weight (which fits in L2) but only
// ~197 GB/s on the model's 1.4GB of weights, where nothing is cached. The card's SM
// utilization sits at 100% while its memory utilization sits at 66%, which is the
// signature of memory *latency* not being covered: each thread issues too few
// independent loads to hide the round trip.
//
// This kernel gives every output row a whole warp and lets each lane hold several
// outstanding k-lane loads (K_UNROLL independent accumulators), so the loads of one
// iteration overlap the latency of the previous one.

#include <cuda_bf16.h>

#define GEMV_WARPS 8
#ifndef GEMV_UNROLL
#define GEMV_UNROLL 4
#endif

// y[n] = sum_k x[k] * W[n, k]   x: [K] bf16, W: [N, K] bf16, y: [N] fp32
extern "C" __global__ void gemv_bf16(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ W,
                                     float* __restrict__ y,
                                     int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n0 = (blockIdx.x * GEMV_WARPS) + warp;

    // One warp covers whole rows: n0 is fixed per warp, lanes stride k.
    for (int n = n0; n < N; n += (gridDim.x * GEMV_WARPS)) {
        const __nv_bfloat16* w = W + (long)n * K;
        float acc[GEMV_UNROLL];
#pragma unroll
        for (int u = 0; u < GEMV_UNROLL; ++u) acc[u] = 0.f;

        int k = lane;
        // Main loop: GEMV_UNROLL independent loads per iteration per lane, so the warp
        // has GEMV_UNROLL * 32 loads in flight.
        for (; k + (GEMV_UNROLL - 1) * 32 < K; k += GEMV_UNROLL * 32) {
#pragma unroll
            for (int u = 0; u < GEMV_UNROLL; ++u) {
                float wv = __bfloat162float(w[k + u * 32]);
                float xv = __bfloat162float(x[k + u * 32]);
                acc[u] = fmaf(wv, xv, acc[u]);
            }
        }
        for (; k < K; k += 32) {
            acc[0] = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), acc[0]);
        }
#pragma unroll
        for (int u = 1; u < GEMV_UNROLL; ++u) acc[0] += acc[u];
        for (int off = 16; off > 0; off >>= 1) acc[0] += __shfl_down_sync(0xffffffffu, acc[0], off);
        if (lane == 0) y[n] = acc[0];
    }
}
