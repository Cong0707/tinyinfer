// gemv_fp4_w4a8: fp8 activation x fp4 (E2M1) weight, per-32-element block scales.
//
// Replaces the tilelang gemv_fp4_kernel_v2 (deepseek/gemv_fp4.py) that the MoE serve
// path calls for every routed expert. Removing it is what lets decode run without
// tilelang, which otherwise has to JIT-compile on first use and refuses to import in a
// stripped-down deployment.
//
// Contract, matching the tilelang kernel exactly:
//
//     Y[m,n] = sum_kb  AS[m,kb] * WS[n,kb] * sum_l A[m, kb*32+l] * W[n, kb*32+l]
//
// The scales sit INSIDE the k loop: they are per-32-block, so the inner dot has to be
// reduced first and then multiplied by that block's pair of scales. A single running
// accumulator across blocks would apply one block's scale to every block's product.
// (The tilelang version reaches the same value by accumulating scale-multiplied lanes in
// a per-lane fragment and reducing at the end -- same arithmetic, different order.)
//
// E2M1 has exactly eight magnitudes and they are all representable in e4m3, so the fp4
// table below is exact; the tilelang kernel expresses the same thing as a cast chain
// float4_e2m1fn -> fp8 -> fp32.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

// index = sign(1) : exp(2) : mantissa(1)
__device__ __constant__ float kFp4Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

constexpr int ROWS = 8;   // output rows per block, one warp each

}  // namespace

// A  [M, K]     fp8 e4m3
// AS [M, K/32]  fp32
// W  [N, K/2]   packed E2M1, low nibble = even k
// WS [N, K/32]  fp32
// Y  [M, N]     bf16
extern "C" __global__ void gemv_fp4_w4a8(
    const __nv_fp8_e4m3* __restrict__ A,
    const float* __restrict__ AS,
    const unsigned char* __restrict__ W,
    const float* __restrict__ WS,
    __nv_bfloat16* __restrict__ Y,
    int M, int N, int K) {
    const int nb = K >> 5;
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;
    const int nblocks = (N + ROWS - 1) / ROWS;
    const int bx = blockIdx.x % nblocks;
    const int bm = blockIdx.x / nblocks;
    if (bm >= M) return;

    const int n = bx * ROWS + r;
    if (n >= N) return;

    const __nv_fp8_e4m3* arow = A + (long long)bm * K;
    const unsigned char* wrow = W + (long long)n * (K >> 1);
    const float* asrow = AS + (long long)bm * nb;
    const float* wsrow = WS + (long long)n * nb;

    float total = 0.f;
    for (int kb = 0; kb < nb; ++kb) {
        const int kidx = (kb << 5) + lane;
        const float a = float(arow[kidx]);
        const unsigned char byte = wrow[kidx >> 1];
        const int nib = (kidx & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
        float part = a * kFp4Lut[nib];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
        total += part * asrow[kb] * wsrow[kb];
    }

    if (lane == 0) Y[(long long)bm * N + n] = __float2bfloat16(total);
}
