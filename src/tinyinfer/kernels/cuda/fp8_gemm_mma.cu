// fp8 GEMM on tensor cores, with per-32-k blockwise scales.
//
// This is the M>=2 counterpart to fp8_gemv_bs (which is the M==1 SIMT path). Both
// compute the same thing; this one uses mma.sync.aligned.m16n8k32 so its arithmetic
// matches what the Hopper tilelang kernel does, which is what the reference model's
// outputs are defined by.
//
// The scale has to be applied INSIDE the k loop: it is per 32-element block, so each
// block's dot product is finished, scaled, and only then accumulated:
//
//     C[m,n] = sum_kb  (sum_{k in kb} A[m,k]*B[n,k]) * as[m,kb] * bs[n/32,kb]
//
// Because the mma instruction's own k is 32, one mma call is exactly one scale block.
//
// Layout (verified on device against a known input):
//   A fragment: 4 x uint32  a0=(row g,   k q*4+0..3)  a1=(row g+8, k q*4+0..3)
//                               a2=(row g,   k q*4+16+0..3)  a3=(row g+8, ...)
//   B fragment: 2 x uint32  b0=(col g,   k q*4+0..3)  b1=(col g, k q*4+16+0..3)
//   C fragment: 4 x float   c0,c1 -> (row g, col q*2, q*2+1)  c2,c3 -> (row g+8, ...)
// where g = lane/4 and q = lane%4.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int WARP_M = 16;    // rows per mma
constexpr int WARP_N = 8;     // cols per mma
constexpr int WARP_K = 32;    // k per mma, also one scale block
constexpr int NB = 8;         // mma tiles per warp along N (=> 64 cols)
constexpr int BLOCK_M = 128;  // 8 warps stacked along M

__device__ __forceinline__ float e8m0_val(unsigned char b) {
    // The stored byte IS the biased exponent: value = 2^(b - 127), so the float bits are
    // just b << 23. Adding 127 first (as an earlier version did) turned the neutral scale
    // 127 into an exponent of 254, i.e. +inf, and every output came out inf/nan.
    return __uint_as_float((unsigned int)b << 23);
}

__device__ __forceinline__ void mma_m16n8k32(float d[4], const uint32_t a[4],
                                             const uint32_t b[2]) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

}  // namespace

// A  [M, K]     e4m3
// AS [M, K/32]  e8m0
// B  [N, K]     e4m3
// BS [N/32, K/32] e8m0
// C  [M, N]     bf16
extern "C" __global__ void fp8_gemm_mma(
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ AS,
    const unsigned char* __restrict__ B, const unsigned char* __restrict__ BS,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int q = lane & 3;

    const int m0 = blockIdx.y * BLOCK_M + warp * WARP_M;
    const int n0 = blockIdx.x * (WARP_N * NB);
    if (m0 >= M) return;

    const int nb_k = K >> 5;

    // start a little clear of the end so __shfl and the writes stay in range
    const bool rows_lo = (m0 + g) < M;
    const bool rows_hi = (m0 + g + 8) < M;

    float acc[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    for (int kb = 0; kb < nb_k; ++kb) {
        const int kbase = kb << 5;

        uint32_t a[4];
        if (rows_lo) {
            a[0] = *(const uint32_t*)(A + (long long)(m0 + g) * K + kbase + q * 4);
            a[2] = *(const uint32_t*)(A + (long long)(m0 + g) * K + kbase + q * 4 + 16);
        } else { a[0] = 0u; a[2] = 0u; }
        if (rows_hi) {
            a[1] = *(const uint32_t*)(A + (long long)(m0 + g + 8) * K + kbase + q * 4);
            a[3] = *(const uint32_t*)(A + (long long)(m0 + g + 8) * K + kbase + q * 4 + 16);
        } else { a[1] = 0u; a[3] = 0u; }

        const float as_lo = rows_lo ? e8m0_val(AS[(long long)(m0 + g) * nb_k + kb]) : 0.f;
        const float as_hi = rows_hi ? e8m0_val(AS[(long long)(m0 + g + 8) * nb_k + kb]) : 0.f;

#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const int n = n0 + nb * WARP_N + g;
            if (n >= N) continue;
            uint32_t b[2];
            b[0] = *(const uint32_t*)(B + (long long)n * K + kbase + q * 4);
            b[1] = *(const uint32_t*)(B + (long long)n * K + kbase + q * 4 + 16);

            float d[4];
            mma_m16n8k32(d, a, b);

            // scale for this block; bs is per 32 columns, and this lane's columns
            // are q*2 and q*2+1, which always fall in the same 32-group
            const float bs_n = e8m0_val(BS[(long long)(n >> 5) * nb_k + kb]);
            acc[nb][0] += d[0] * as_lo * bs_n;
            acc[nb][1] += d[1] * as_lo * bs_n;
            acc[nb][2] += d[2] * as_hi * bs_n;
            acc[nb][3] += d[3] * as_hi * bs_n;
        }
    }

#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        const int col = n0 + nb * WARP_N + q * 2;
        if (col >= N) continue;
        if (rows_lo) {
            C[(long long)(m0 + g) * N + col] = __float2bfloat16(acc[nb][0]);
            if (col + 1 < N)
                C[(long long)(m0 + g) * N + col + 1] = __float2bfloat16(acc[nb][1]);
        }
        if (rows_hi) {
            C[(long long)(m0 + g + 8) * N + col] = __float2bfloat16(acc[nb][2]);
            if (col + 1 < N)
                C[(long long)(m0 + g + 8) * N + col + 1] = __float2bfloat16(acc[nb][3]);
        }
    }
}
