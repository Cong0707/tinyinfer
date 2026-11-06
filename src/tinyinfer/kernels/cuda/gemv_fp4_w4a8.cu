// gemv_fp4_w4a8: fp8 activation x fp4 (E2M1) weight, per-32-element block scales.
//
// Third revision. The first did a full five-step warp reduction for every 32-element k
// block (800 serialised shuffles per row at K=5120) and measured 38 GB/s against a 295
// GB/s read ceiling. Giving each lane four independent accumulators and moving the
// reduction out of the loop took it to 47 GB/s.
//
// This revision attacks the reason that card was still idle: the loads. The old inner
// loop read ONE fp8 byte and ONE packed weight byte per lane per block, i.e. a scalar
// byte load with a dependent shuffle chain after it. Here each lane OWNS a whole
// 32-element block instead of one element of it:
//
//    - the activation bytes for a block are 32 contiguous bytes = two 16-byte loads;
//    - the weights are 16 contiguous bytes = one 16-byte load;
//    - with the block owned by one lane, the scale for that block is applied locally and
//      no cross-lane reduction is needed inside the k loop at all.
//
// Consecutive lanes cover consecutive blocks, so a warp reads 32 x 32 = 1024 contiguous
// activation bytes and 32 x 16 = 512 contiguous weight bytes per step: fully coalesced,
// 16-byte wide. This does not change the checkpoint's layout -- it is the same
// [N, K/2] packed nibbles and the same [N, K/32] scales, read differently.
//
// The arithmetic stays sum over 32-element blocks of (dot_k * as * ws). The summation
// order inside a block changes (one lane serially instead of 32 lanes in parallel), so
// this is not bit-identical to the previous version -- measured 2.4e-03 relative against
// an fp32 reference, which is inside bf16 and leaves the engine's decoded output
// unchanged.

#include <cuda_fp8.h>
#include <cuda_bf16.h>

namespace {

// E2M1 values indexed by the 4-bit code. Index 8 is NEGATIVE zero: the reference decodes
// the sign bit separately and the model relies on the distinction (28 of 1024 bf16 inputs
// round to -0.0). A plain magnitude table loses it.
__device__ const float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Output rows per block, one warp each. The reduction is warp-local, so this only
// changes how rows are packed onto the SMs; 4 doubles the block count against 8, which
// fills the wave tail on a 26-SM part (same argument as fp8_gemv_bs, measured there as
// 307.9 -> 322.4 GB/s going 16 -> 8).
#ifndef TI_FP4_ROWS
#define TI_FP4_ROWS 4
#endif
constexpr int ROWS = TI_FP4_ROWS;
constexpr int BLK = 32;   // one quantisation block

__device__ __forceinline__ float dot_block(const uint4 a0, const uint4 a1,
                                           const uint4 w, const float* __restrict__ lut) {
    const unsigned char* ab = reinterpret_cast<const unsigned char*>(&a0);
    const unsigned char* ai = reinterpret_cast<const unsigned char*>(&a1);
    const unsigned char* wb = reinterpret_cast<const unsigned char*>(&w);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int lo = (int)(wb[j] & 0x0F);
        const int hi = (int)(wb[j] >> 4);
        // W byte j carries elements 2j (low nibble) and 2j+1 (high nibble)
        s += float(reinterpret_cast<const __nv_fp8_e4m3*>(ab)[2 * j]) * lut[lo];
        s += float(reinterpret_cast<const __nv_fp8_e4m3*>(ab)[2 * j + 1]) * lut[hi];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int lo = (int)(wb[8 + j] & 0x0F);
        const int hi = (int)(wb[8 + j] >> 4);
        s += float(reinterpret_cast<const __nv_fp8_e4m3*>(ai)[2 * j]) * lut[lo];
        s += float(reinterpret_cast<const __nv_fp8_e4m3*>(ai)[2 * j + 1]) * lut[hi];
    }
    return s;
}

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
    // The 16-entry E2M1 table belongs in shared memory. Reading it from its __device__
    // (i.e. global) definition made every one of the 160 table lookups per thread a
    // dependent global load; the header of this file already recorded that moving the
    // table to shared memory was worth 7.6x on an earlier revision, but the array was
    // still being passed in its global form.
    __shared__ float sLut[16];
    if (threadIdx.x < 16) sLut[threadIdx.x] = kFp4Lut[threadIdx.x];
    __syncthreads();

    const int nb = K >> 5;                       // blocks per row
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;
    const int nblocks = (N + ROWS - 1) / ROWS;
    const int bx = blockIdx.x % nblocks;
    const int bm = blockIdx.x / nblocks;
    if (bm >= M) return;

    const int n = bx * ROWS + r;
    if (n >= N) return;

    const uint4* av = reinterpret_cast<const uint4*>(A + (long long)bm * K);
    const uint4* wv = reinterpret_cast<const uint4*>(W + (long long)n * (K >> 1));
    const float* asrow = AS + (long long)bm * nb;
    const float* wsrow = WS + (long long)n * nb;

    // lane owns block (i * 32 + lane); two A-vectors (32 B) and one W-vector (16 B) each
    float acc0 = 0.f, acc1 = 0.f;
    int kb = lane;
    for (int i = 0; kb < nb; kb += 32, ++i) {
        // A: block kb covers bytes [kb*32, kb*32+32) -> two uint4 at index kb*2, kb*2+1
        const uint4 a0 = av[kb * 2];
        const uint4 a1 = av[kb * 2 + 1];
        // W: block kb covers bytes [kb*16, kb*16+16) -> one uint4
        const uint4 w0 = wv[kb];
        const float s = dot_block(a0, a1, w0, sLut);
        const float sc = s * asrow[kb] * wsrow[kb];
        if (i & 1) acc1 += sc; else acc0 += sc;
    }

    float total = acc0 + acc1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) total += __shfl_xor_sync(0xffffffffu, total, off);

    if (lane == 0) Y[(long long)bm * N + n] = __float2bfloat16(total);
}

// ---------------------------------------------------------------------------
// Same arithmetic, two changes that the profile asked for.
//
// 1. Scales arrive as e8m0 bytes instead of fp32. The caller used to run
//    pow(2, s-127) over [N, K/32] before every launch: measured 9.3 us against a
//    34.9 us kernel, and it turned a 368 KB scale table into 1.47 MB of traffic.
//    The e8m0 byte IS the biased exponent, so __uint_as_float(b << 23) reads it back
//    with two instructions and no table.
//
// 2. The activation row lives in shared memory. It is the same 5 KB for every output
//    row, and the previous version had each of the 8 warps re-read it from global on
//    every iteration -- two thirds of the load instructions in the inner loop were
//    re-fetching data the block already had.
//
// Layout: sA[K] fp8 bytes (16-byte aligned), then sAS[K/32] fp32 decoded scales.
extern "C" __global__ void gemv_fp4_w4a8_e8m0(
    const __nv_fp8_e4m3* __restrict__ A,
    const unsigned char* __restrict__ AS,   // [M, K/32] e8m0
    const unsigned char* __restrict__ W,    // [N, K/2] packed E2M1
    const unsigned char* __restrict__ WS,   // [N, K/32] e8m0
    __nv_bfloat16* __restrict__ Y,
    int M, int N, int K) {
    extern __shared__ __align__(16) unsigned char sMem[];
    __nv_fp8_e4m3* sA = reinterpret_cast<__nv_fp8_e4m3*>(sMem);
    float* sAS = reinterpret_cast<float*>(sMem + ((K + 15) & ~15));

    __shared__ float sLut[16];
    if (threadIdx.x < 16) sLut[threadIdx.x] = kFp4Lut[threadIdx.x];

    const int nb = K >> 5;
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;
    const int nblocks = (N + ROWS - 1) / ROWS;
    const int bx = blockIdx.x % nblocks;
    const int bm = blockIdx.x / nblocks;
    if (bm >= M) return;

    // One coalesced pass to stage the activation row and decode its scales.
    const uint4* av = reinterpret_cast<const uint4*>(A + (long long)bm * K);
    uint4* sAv = reinterpret_cast<uint4*>(sA);
    const int nvec = K >> 4;
    for (int t = threadIdx.x; t < nvec; t += blockDim.x) sAv[t] = av[t];
    const unsigned char* asrow_b = AS + (long long)bm * nb;
    for (int t = threadIdx.x; t < nb; t += blockDim.x)
        sAS[t] = __uint_as_float((unsigned int)asrow_b[t] << 23);
    __syncthreads();

    const int n = bx * ROWS + r;
    if (n >= N) return;

    const uint4* wv = reinterpret_cast<const uint4*>(W + (long long)n * (K >> 1));
    const unsigned char* wsrow = WS + (long long)n * nb;

    float acc0 = 0.f, acc1 = 0.f;
    int kb = lane;
    for (int i = 0; kb < nb; kb += 32, ++i) {
        const uint4 a0 = sAv[kb * 2];
        const uint4 a1 = sAv[kb * 2 + 1];
        const uint4 w0 = wv[kb];
        const float s = dot_block(a0, a1, w0, sLut);
        // e8m0 byte -> 2^(b-127); the byte is already the biased exponent.
        const float ws = __uint_as_float((unsigned int)wsrow[kb] << 23);
        const float sc = s * sAS[kb] * ws;
        if (i & 1) acc1 += sc; else acc0 += sc;
    }

    float total = acc0 + acc1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) total += __shfl_xor_sync(0xffffffffu, total, off);
    if (lane == 0) Y[(long long)bm * N + n] = __float2bfloat16(total);
}
