// FP8 e4m3 GEMV for decode (M=1), block-scaled (32x32), bf16 out.
//
// Replaces the tilelang fp8_gemm for the shared expert: that kernel uses block_M=32, so
// at M=1 thirty-one of the thirty-two tile rows are padding. Measured 152 us for
// [2304, 5120] fp8 = 11.8 MB -> 78 GB/s, while the same shape at 4.25 bit reaches 316
// GB/s through Marlin. The gap is the tile shape, not the arithmetic.
//
// Layout (same idea as moe_gemv6.cu, which measures 246 GB/s on fp4):
//   block = 512 = 16 output rows x 32 lanes
//   each lane owns 16 consecutive k (16 fp8 bytes, one LDG.E.128); a warp therefore
//   covers 512 consecutive k, so every fetch is full 128-byte transactions.
//   A scale group is 32 k, i.e. two lanes, so the group index is (lane >> 1).
//
// e8m0 needs no ldexpf: the byte IS the IEEE exponent field.
// e4m3 decodes through the hardware cvt.rn.f16x2.e4m3x2 (two values per instruction).
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

__device__ __forceinline__ float e8m0(unsigned char b) {
    return __uint_as_float((unsigned int)b << 23);
}

// four packed e4m3 bytes -> four floats
__device__ __forceinline__ void e4m3x4(unsigned int raw, float* o) {
    const __half2_raw hr0 = __nv_cvt_fp8x2_to_halfraw2((unsigned short)(raw & 0xFFFFu), __NV_E4M3);
    const __half2_raw hr1 = __nv_cvt_fp8x2_to_halfraw2((unsigned short)(raw >> 16), __NV_E4M3);
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&hr0));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hr1));
    o[0] = f0.x; o[1] = f0.y; o[2] = f1.x; o[3] = f1.y;
}

}  // namespace

extern "C" __global__ void fp8_gemv_bs(
    const unsigned char* __restrict__ a,    // [K]    fp8 e4m3 activation
    const unsigned char* __restrict__ as,   // [K/32] e8m0 activation scales
    const unsigned char* __restrict__ w,    // [N, K] fp8 e4m3
    const unsigned char* __restrict__ ws,   // [N/32, K/32] e8m0 weight scales
    __nv_bfloat16* __restrict__ y,          // [N]
    int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 16 + (threadIdx.x >> 5);
    if (n >= n_rows) return;

    const int nb = k_dim >> 5;              // scale groups along k
    const unsigned char* wr = w + (long long)n * k_dim;
    const unsigned char* wsr = ws + (long long)(n >> 5) * nb;

    float acc = 0.f;
    const int niter = k_dim >> 9;           // 512 k per warp-iteration
    for (int it = 0; it < niter; ++it) {
        const int kbase = (it << 9) + (lane << 4);
        const int grp = (it << 4) + (lane >> 1);   // 32-k group of kbase

        // activation: 16 bytes, four hardware converts
        const uint4 araw = *(const uint4*)(a + kbase);
        float av[16];
        {
            const unsigned int* u = reinterpret_cast<const unsigned int*>(&araw);
#pragma unroll
            for (int i = 0; i < 4; ++i) e4m3x4(u[i], av + 4 * i);
        }
        const float asc = e8m0(as[grp]);
        const float wsc = e8m0(wsr[grp]);

        // weights: 16 bytes, four hardware converts, dot with the activation
        const uint4 wraw = *(const uint4*)(wr + kbase);
        const unsigned int* u = reinterpret_cast<const unsigned int*>(&wraw);
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float wv[4];
            e4m3x4(u[i], wv);
            part += av[4 * i] * wv[0] + av[4 * i + 1] * wv[1]
                  + av[4 * i + 2] * wv[2] + av[4 * i + 3] * wv[3];
        }
        acc += part * asc * wsc;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}
