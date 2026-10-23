// Hand-written decode kernels (no torch headers: nvcc -> cubin, loaded by
// tinyinfer/native.py through the CUDA driver API).
//
// rms_norm: the torch form is float/square/mean/rsqrt/mul/mul/to -- seven tiny kernels
// per call, twice per layer. One kernel per row: 32 lanes reduce over d, then scale.

#include <cuda_bf16.h>

extern "C" __global__ void rms_norm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    int rows, int d, float eps) {
    const int r = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const __nv_bfloat16* xr = x + (long)r * d;
    __nv_bfloat16* yr = y + (long)r * d;

    float s = 0.f;
    for (int k = lane; k < d; k += 32) {
        const float v = __bfloat162float(xr[k]);
        s += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    s = __shfl_sync(0xffffffffu, s, 0);
    const float rstd = rsqrtf(s / (float)d + eps);

    for (int k = lane; k < d; k += 32) {
        const float v = __bfloat162float(xr[k]) * rstd * w[k];
        yr[k] = __float2bfloat16(v);
    }
}
