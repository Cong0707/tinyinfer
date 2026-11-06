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

// one packed e4m3 byte -> float, for the scalar tail path (K not a multiple of 512)
__device__ __forceinline__ float e4m3_one(unsigned char b) {
    float o;
    e4m3x4((unsigned int)b, &o);
    return o;
}

}  // namespace

extern "C" __global__ void fp8_gemv_bs(
    const unsigned char* __restrict__ a,    // [K]    fp8 e4m3 activation
    const unsigned char* __restrict__ as,   // [K/32] e8m0 activation scales
    const unsigned char* __restrict__ w,    // [N, K] fp8 e4m3
    const unsigned char* __restrict__ ws,   // [N/32, K/32] e8m0 weight scales
    __nv_bfloat16* __restrict__ y,          // [N]
    int n_rows, int k_dim) {
    // Rows per block comes from the launch, not a literal: one warp handles one row and
    // the reduction is warp-local, so the block's warp count only affects how the rows are
    // packed onto the SMs -- the arithmetic is identical for any block size. That makes it
    // a free knob for the wave-tail: N/16 blocks on a 26-SM part is 1.4 waves (the second
    // wave leaves 40% of the SMs idle), while N/8 doubles the block count and fills it.
    const int warps = blockDim.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * warps + (threadIdx.x >> 5);
    if (n >= n_rows) return;

    const int nb = k_dim >> 5;              // scale groups along k
    const unsigned char* wr = w + (long long)n * k_dim;
    const unsigned char* wsr = ws + (long long)(n >> 5) * nb;

    float acc = 0.f;
    // K need not be a multiple of 512: the loop rounds up and the last iteration handles
    // a short tail. That matters because two of the largest projections are NOT 512-
    // aligned -- wq_b is K=1280 (42 MB) and the shared expert's w2 is K=2304 -- and they
    // would otherwise fall through to the block_M=32 GEMM, the slowest thing left in the
    // decode step. A 16-byte vector load past the end of K would walk off the tensor, so
    // an incomplete lane group reads scalar instead.
    const int niter = (k_dim + 511) >> 9;   // 512 k per warp-iteration, rounded up
    // Software-pipelined loads. The loop body is load-then-use, so on a cold run -- which
    // is every decode, since each expert weight is a first touch out of HBM -- the full
    // HBM latency of each load sits exposed between iterations. That is worth a lot:
    // measured on the decode shapes the kernel delivers ~192 GB/s in situ against the
    // 323-395 GB/s it reaches when the data is warm. Issuing the next iteration's two
    // 16-byte vector loads before consuming the current pair lets them overlap with the
    // decode and the FMA of the current one.
    //
    // Nothing about the arithmetic moves: same iterations, same accumulation order, same
    // tail handling. Only the load is hoisted.
    uint4 a_pre, w_pre;
    bool full_pre = false, valid_pre = false;
    int grp_pre = 0;
#define TI_PREFETCH(IT)                                                          \
    do {                                                                         \
        const int _kb = ((IT) << 9) + (lane << 4);                               \
        valid_pre = (_kb < k_dim);                                               \
        grp_pre = valid_pre ? (((IT) << 4) + (lane >> 1)) : 0;                   \
        full_pre = valid_pre && (_kb + 16) <= k_dim;                             \
        if (full_pre) {                                                          \
            a_pre = *(const uint4*)(a + _kb);                                    \
            w_pre = *(const uint4*)(wr + _kb);                                   \
        }                                                                        \
    } while (0)

    TI_PREFETCH(0);
    for (int it = 0; it < niter; ++it) {
        const uint4 araw = a_pre;
        const uint4 wraw = w_pre;
        const int kbase = (it << 9) + (lane << 4);
        // No continue: this loop contains a __shfl_xor_sync, and every __shfl*_sync in the
        // warp requires ALL 32 lanes to reach it. Skipping ahead on the tail lanes hung the
        // kernel outright (the test never returned). The guard is a bool instead, and the
        // tail lanes simply contribute zero.
        const bool valid = valid_pre;
        const int grp = grp_pre;
        const bool full = full_pre;
        if (it + 1 < niter) TI_PREFETCH(it + 1);

        float av[16];
        if (full) {
            const unsigned int* u = reinterpret_cast<const unsigned int*>(&araw);
#pragma unroll
            for (int i = 0; i < 4; ++i) e4m3x4(u[i], av + 4 * i);
        } else {
            for (int j = 0; j < 16; ++j) {
                const int kk = kbase + j;
                av[j] = (kk < k_dim) ? e4m3_one(a[kk]) : 0.f;
            }
        }
        const float asc = e8m0(as[grp]);
        const float wsc = e8m0(wsr[grp]);

        float part = 0.f;
        if (full) {
            const unsigned int* u = reinterpret_cast<const unsigned int*>(&wraw);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float wv[4];
                e4m3x4(u[i], wv);
                part += av[4 * i] * wv[0] + av[4 * i + 1] * wv[1]
                      + av[4 * i + 2] * wv[2] + av[4 * i + 3] * wv[3];
            }
        } else if (valid) {
            for (int j = 0; j < 16; ++j) {
                const int kk = kbase + j;
                if (kk >= k_dim) break;
                part += av[j] * e4m3_one(wr[kk]);
            }
        }
        // Each lane applies its own group's scale, then the warp reduces. Pairing the two
        // lanes into (x1+x2)*s was tried to match tilelang's (sum of 32 k) * scale, and it
        // BROKE the shapes that used to be bit-identical: end to end went from
        // [989,43,4571,5958] to [989,19923,201,19923]. Whatever tilelang's gemm does
        // internally, it is not that, so this stays as the verified form.
        if (valid) acc += part * asc * wsc;
    }
    // Warp reduce. The groups are already summed in the same order as tilelang's
    // k loop within each lane; a fully serial k-order reduce was tried and is far
    // too slow (one shuffle per 32-k group, ~160 per call).
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}