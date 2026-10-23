// Hand-written decode kernels (no torch headers: nvcc -> cubin, loaded by
// tinyinfer/native.py through the CUDA driver API).
//
// gemv_fp4_e8m0: the fp4 expert GEMV reading the checkpoint's e8m0 scale bytes directly.
// The tilelang path needs an fp32 copy of every scale (1.4MB per scale, three per
// expert, 1680 copies per decode step ~= 168ms of the aten::copy_ total). e8m0 is
// 2^(byte-127), so the kernel decodes it inline with one ldexpf.
//
//   A   [K]        fp8 e4m3 activation (already widened to float by the caller)
//   AS  [nb]       fp32 activation scales
//   W   [N, K/2]   packed E2M1, low nibble first
//   WS  [N, nb]    e8m0 scale bytes
//   Y   [N]        bf16

#include <cuda_bf16.h>

__device__ __constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float e8m0(unsigned char b) {
    // 0xFF is NaN and 0x00 is 2^-127 in the format; both are treated as the power of two
    return ldexpf(1.0f, (int)b - 127);
}

extern "C" __global__ void gemv_fp4_e8m0(
    const float* __restrict__ a,
    const float* __restrict__ as,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ ws,
    __nv_bfloat16* __restrict__ y,
    int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;
    const int nb = k_dim >> 5;
    const long long khalf = k_dim >> 1;
    const int n = blockIdx.x * 16 + r;
    if (n >= n_rows) return;

    float acc = 0.f;
    for (int kb = 0; kb < nb; ++kb) {
        float part = 0.f;
        for (int j = lane; j < 32; j += 32) {
            const int k = kb * 32 + j;
            const unsigned char byte = w[(long long)n * khalf + (k >> 1)];
            const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
            part += a[k] * kFp4Lut[nib];
        }
        acc += part * as[kb] * e8m0(ws[n * nb + kb]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}
