// Hand-written decode kernels (no torch headers: nvcc -> cubin, loaded by
// tinyinfer/native.py through the CUDA driver API).
//
// head_gemv: the lm-head projection. The checkpoint head is bf16 (129280 x 5120); the
// official path casts it to fp32 per token and runs four chunked fp32 GEMMs, which cost
// ~206ms in the M=1 GEMV shape. This kernel reads the bf16 weights directly, accumulates
// in fp32 and writes fp32 -- the same arithmetic the reference does, one kernel.
//
//   out[n] = sum_k (float)h[k] * (float)w[n][k]

#include <cuda_bf16.h>

extern "C" __global__ void head_gemv(const float* __restrict__ h,
                                     const __nv_bfloat16* __restrict__ w,
                                     float* __restrict__ out,
                                     int n_rows, int k_dim) {
    // one warp per output row; 32 lanes split k
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int n = warp;
    if (n >= n_rows) return;
    const __nv_bfloat16* row = w + (long)n * k_dim;
    float acc = 0.f;
    for (int k = lane; k < k_dim; k += 32) {
        acc += h[k] * __bfloat162float(row[k]);
    }
    // warp reduce
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[n] = acc;
}
