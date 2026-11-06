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
#ifndef TI_MMA_NB
#define TI_MMA_NB 4
#endif
// mma tiles per warp along N. This is the knob that sets the grid size: with BLOCK_M=128
// and 8 warps per block the whole M=128 problem is 36 blocks (N/64), i.e. 1.4 blocks per
// SM on a 26-SM part -- 11 warps resident out of a possible 64, 17% occupancy. Halving
// NB to 4 doubles the block count to 72 and lifts occupancy to ~34% without touching the
// arithmetic: each block still computes whole mma tiles, just fewer of them.
constexpr int NB = TI_MMA_NB;
constexpr int BLOCK_M = 128;  // 8 warps stacked along M
constexpr int NWARP = BLOCK_M / WARP_M;

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
//
// Built for the whole Turing..GB300 range, so the body is guarded: mma.sync with e4m3
// operands is an sm_89 feature and ptxas rejects it outright for the sm_75 and sm_86
// passes of a fatbin. On those targets this compiles to an empty kernel; the host side
// gates the call on the device capability (gemm._fp8_native), so it is never reached.
extern "C" __global__ void fp8_gemm_mma(
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ AS,
    const unsigned char* __restrict__ B, const unsigned char* __restrict__ BS,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int q = lane & 3;

    const int m0 = blockIdx.y * BLOCK_M + warp * WARP_M;
    const int n0 = blockIdx.x * (WARP_N * NB);
    // No early return: the loop below has a __syncthreads, and a warp that bailed out of
    // it deadlocks the block. Out-of-range work is masked instead.
    const bool warp_active = (m0 < M);

    const int nb_k = K >> 5;
    const int ncol = WARP_N * NB;          // columns this block owns

    // All BLOCK_M rows of a block read the SAME weight columns: the warps are stacked
    // along M, so before this staging each of the 8 warps fetched the block's weight tile
    // from global on its own, 8x the necessary traffic. One tile is ncol x 32 bytes =
    // 1 KB, small enough to double-buffer and keep the next k-block's fetch in flight.
    // B tile in shared memory, swizzled. ncu on the plain layout measured 5,901,609 bank
    // conflicts -- 50.01% of all shared-load wavefronts, i.e. every load took 2.0
    // wavefronts instead of 1.0 -- because the four lanes of a quad read addresses
    // (g*32 + q*4) mod 128, which only ever touches banks 0..3, 8..11, 16..19, 24..27
    // (16 of the 32). The XOR below folds the row index into the low bits so consecutive
    // rows land on different banks, which is the standard tensor-core swizzle.
    constexpr int ROW_B = 32;
#define TI_SWZ(COL, OFF) ((uint8_t*)(sB[cur] + (COL) * ROW_B + (((COL) & 3) ^ ((OFF) >> 2)) * 4))
    // B is shared by the whole block: the 8 warps are stacked along M and need the SAME
    // weight columns, so staging it per warp multiplies the global reads by 8 (measured:
    // 23.0 -> 7.6 TFLOPS). Block-wide staging plus one __syncthreads per k-block is the
    // cheaper trade, and the barrier is what the double buffer exists to hide.
    __shared__ __align__(16) uint8_t sB[2][WARP_N * NB * ROW_B];

#define TI_STAGE_B(ST, KB)                                                       \
    do {                                                                         \
        const int _kb = (KB) << 5;                                               \
        for (int t = threadIdx.x; t < ncol * 8; t += blockDim.x) {               \
            const int _col = t >> 3;                                             \
            const int _off = (t & 7) << 2;                                       \
            const int _n = n0 + _col;                                            \
            const uint32_t _v = (_n < N)                                          \
                ? *(const uint32_t*)(B + (long long)_n * K + _kb + _off)         \
                : 0u;                                                            \
            *(uint32_t*)(sB[ST] + _col * ROW_B + (((_col) & 3) ^ ((_off) >> 2)) * 4)   \
                = _v;                                                          \
        }                                                                        \
    } while (0)

    // start a little clear of the end so __shfl and the writes stay in range
    const bool rows_lo = warp_active && (m0 + g) < M;
    const bool rows_hi = warp_active && (m0 + g + 8) < M;

    float acc[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    // Loop-invariant row bases (see the note at their use site).
    const unsigned char* a_lo_row = A + (long long)(m0 + g) * K;
    const unsigned char* a_hi_row = A + (long long)(m0 + g + 8) * K;
    const unsigned char* as_lo_row = AS + (long long)(m0 + g) * nb_k;
    const unsigned char* as_hi_row = AS + (long long)(m0 + g + 8) * nb_k;

    int cur = 0;
    TI_STAGE_B(0, 0);
    for (int kb = 0; kb < nb_k; ++kb) {
        __syncthreads();                       // sB[cur] is ready for every warp
        if (kb + 1 < nb_k) TI_STAGE_B(1 - cur, kb + 1);
        const int kbase = kb << 5;

        // A's row bases are loop-invariant: (m0+g)*K does not depend on kb, and the
        // previous form recomputed it (a 64-bit multiply) four times per k-block, 160 times
        // at K=5120. Hoisting them into registers outside the loop removes 8 integer
        // instructions per iteration; at four mma per iteration that is a measurable share
        // of the ALU pipe ncu reported as the busiest one (21.1%).
        uint32_t a[4];
        if (rows_lo) {
            a[0] = *(const uint32_t*)(a_lo_row + kbase + q * 4);
            a[2] = *(const uint32_t*)(a_lo_row + kbase + q * 4 + 16);
        } else { a[0] = 0u; a[2] = 0u; }
        if (rows_hi) {
            a[1] = *(const uint32_t*)(a_hi_row + kbase + q * 4);
            a[3] = *(const uint32_t*)(a_hi_row + kbase + q * 4 + 16);
        } else { a[1] = 0u; a[3] = 0u; }

        const float as_lo = rows_lo ? e8m0_val(as_lo_row[kb]) : 0.f;
        const float as_hi = rows_hi ? e8m0_val(as_hi_row[kb]) : 0.f;

        // Independent accumulators, one per mma tile, so the four mma instructions of this
        // k-block issue back to back instead of each waiting on the previous one's scale
        // FMA. The scale is applied after ALL four have retired: every lane's d[] is a
        // hardware register set written by the mma itself, so there is no dependency
        // between them until the multiply. Doing the FMA inline (the previous form) put a
        // read-after-write on acc[nb] right behind every mma, which the scheduler cannot
        // hide with only four instructions in flight.
        float d0[NB], d1[NB], d2[NB], d3[NB];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const int _col = nb * WARP_N + g;
            const uint8_t* _base = sB[cur] + _col * ROW_B;
            uint32_t b[2];
            b[0] = *(const uint32_t*)(_base + ((_col & 3) ^ q) * 4);
            b[1] = *(const uint32_t*)(_base + ((_col & 3) ^ (q + 4)) * 4);
            float dd[4];
            mma_m16n8k32(dd, a, b);
            d0[nb] = dd[0];
            d1[nb] = dd[1];
            d2[nb] = dd[2];
            d3[nb] = dd[3];
        }
        // One bs per 32-column scale block, and NB*WARP_N == 32 when NB == 4, so every nb in
        // this block shares the SAME weight scale. The previous form decoded it once per
        // nb (four e8m0_val calls where one suffices) and then multiplied by as twice per
        // accumulator, i.e. 32 multiply instructions for what is 2 + 4 per nb. The two
        // products s_lo/s_hi are formed once and reused by all four FMA, which is what a
        // fused multiply-add wants anyway.
        const int _nc = (n0 >> 5) * nb_k + kb;   // column scale-block of this n tile
        const float bs_c = ((n0 >> 5) < (N + 31) / 32 && nb_k > 0)
            ? e8m0_val(BS[(long long)_nc]) : 0.f;
        const float s_lo = as_lo * bs_c;
        const float s_hi = as_hi * bs_c;
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            acc[nb][0] += d0[nb] * s_lo;
            acc[nb][1] += d1[nb] * s_lo;
            acc[nb][2] += d2[nb] * s_hi;
            acc[nb][3] += d3[nb] * s_hi;
        }
        cur ^= 1;
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
#endif
}
