// dequant_fp8: fp8 e4m3 weight + block-32 e8m0 scale -> bf16.
//
// Replaces the tilelang _dequant_kernel this tree used to carry. The arithmetic is a
// plain per-element multiply: y[o][i] = float(w[o][i]) * 2^(s[o/32][i/32] - 127).
// One thread owns 16 consecutive i so the loads are 16-byte wide and the scale byte is
// read once per group instead of per element.

#include <cuda_fp8.h>
#include <cuda_bf16.h>

extern "C" __global__ void dequant_fp8(
    const __nv_fp8_e4m3* __restrict__ w,   // [O, I]
    const unsigned char* __restrict__ s,   // [O/32, I/32] e8m0
    __nv_bfloat16* __restrict__ y,         // [O, I]
    int O, int I) {
    const int nbi = I >> 5;                    // scale groups along I
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) << 4;   // 16 per thread
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= O || i0 >= I) return;
    const unsigned char sb = s[(long long)(o >> 5) * nbi + (i0 >> 5)];
    const float scale = __uint_as_float((unsigned int)sb << 23);   // e8m0 byte is the exponent
    const __nv_fp8_e4m3* src = w + (long long)o * I + i0;
    __nv_bfloat16* dst = y + (long long)o * I + i0;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        dst[k] = __float2bfloat16(float(src[k]) * scale);
    }
}
