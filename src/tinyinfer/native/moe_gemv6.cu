// Hand-written decode kernels (no torch headers: nvcc -> cubin, loaded by
// tinyinfer/native.py through the CUDA driver API).
//
// A decode layer of the routed MoE used to cost 12 tilelang GEMV launches, 15 e8m0 ->
// fp32 scale copies, and ~7 elementwise launches for clamp/silu/scale. These four
// kernels replace all of it with four launches and no scale conversion:
//
//   moe_gemv6       [E, 2N] = gate_up(act, w1/w3)          fp8 e4m3 activation
//   moe_act6        [E,  N] = silu(clamp(g)) * clamp(u) * w
//   moe_gemv6_bf16  [E,  D] = down(a, w2)                  bf16 activation
//   moe_acc6        y      += sum_e down[e]
//
// Decoding the checkpoint's own formats inline is the point: expert weights are fp4
// e2m1 packed two per byte, expert scales are raw e8m0 bytes, and the routing
// activation arrives as fp8 e4m3. Converting the scales to fp32 first was the single
// largest aten::copy_ item of a decode step (205ms of 790ms).
//
// GEMV layout for both projections: block = 512 (16 rows x 32 lanes), grid.x walks
// rows, grid.y is the expert, grid.z selects gate (0) or up (1). Expert pointers
// arrive packed in a device int64 array as [w1_e0, w3_e0, w1_e1, ...] and scales as
// [s1_e0, s3_e0, ...]; the output is [E, 2*n_rows] bf16, gate rows then up rows.
//
// Each lane owns eight consecutive k of every 256-wide step, so the weight read and the
// activation read are coalesced and every lane stays inside one 32-wide scale group.
// Three details carry the speed:
//   * the fp4 table lives in shared memory. A __constant__ array indexed by a runtime
//     nibble is replayed once per distinct lane value (16 replays of a warp here);
//     shared memory does it in one issue. This alone was 7.6x.
//   * fp8 e4m3 decodes with the hardware cvt.rn.f16x2.e4m3x2 (two values per
//     instruction) instead of eight scalar converts.
//   * e8m0 needs no ldexpf: the byte *is* the IEEE exponent field.
//
// NOTE: the pointer arrays are 'long long'. A plain 'long' is 32-bit on Windows even
// for nvcc, so indexing them truncated the 64-bit device addresses to their low half
// and every launch faulted with an illegal memory access.
//
// k_dim must be a multiple of 256 (the caller checks; 5120 and 2304 both are).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __constant__ float kFp4Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// e8m0 is exactly the IEEE exponent field: 2^(b-127) is the float whose exponent byte
// is b and whose mantissa is zero. One shift instead of a ldexpf per scale read.
__device__ __forceinline__ float e8m0(const unsigned char b) {
    return __uint_as_float((unsigned int)b << 23);
}

// eight packed e4m3 bytes -> eight floats, four hardware converts
__device__ __forceinline__ void e4m3x8(const unsigned char* p, float* o) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const unsigned int raw = *(const unsigned int*)(p + (q << 2));
        const __half2_raw hr0 = __nv_cvt_fp8x2_to_halfraw2(
            (unsigned short)(raw & 0xFFFFu), __NV_E4M3);
        const __half2_raw hr1 = __nv_cvt_fp8x2_to_halfraw2(
            (unsigned short)(raw >> 16), __NV_E4M3);
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&hr0));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hr1));
        o[q * 4 + 0] = f0.x;
        o[q * 4 + 1] = f0.y;
        o[q * 4 + 2] = f1.x;
        o[q * 4 + 3] = f1.y;
    }
}

// eight bf16 -> eight floats, four 16-byte aligned loads of four bytes each
__device__ __forceinline__ void bf16x8(const unsigned char* p, float* o) {
    union {
        uint4 u;
        __nv_bfloat16 h[8];
    } cv;
    cv.u = *(const uint4*)p;
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = __bfloat162float(cv.h[j]);
}

// MODE 0 = fp8 e4m3 activation, 1 = bf16 activation
template <int MODE>
__device__ __forceinline__ float gemv_row(const unsigned char* __restrict__ a,
                                          const unsigned char* __restrict__ as,
                                          const unsigned char* __restrict__ w,
                                          const unsigned char* __restrict__ ws,
                                          const int k_dim, const int lane,
                                          const float* lut) {
    const int niter = k_dim >> 8;   // 256 k per step
    float acc = 0.f;
    for (int it = 0; it < niter; ++it) {
        const int kbase = (it << 8) + (lane << 3);      // 8 consecutive k per lane
        const unsigned int wpack = *(const unsigned int*)(w + (kbase >> 1));
        const int g = (it << 3) + (lane >> 2);          // 32-wide scale group of kbase
        // a bf16 activation carries no block scale of its own
        const float sc = (MODE == 0 ? e8m0(as[g]) : 1.f) * e8m0(ws[g]);
        float av[8];
        if (MODE == 0) {
            e4m3x8(a + kbase, av);
        } else {
            bf16x8(a + kbase * 2, av);
        }
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            part += av[j] * lut[(wpack >> (j << 2)) & 0xF];
        }
        acc += part * sc;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

extern "C" __global__ void moe_gemv6(
    const unsigned char* __restrict__ a,    // [K] fp8 e4m3
    const unsigned char* __restrict__ as,   // [nb] e8m0 bytes
    const long long* __restrict__ w_ptrs,   // [2E]
    const long long* __restrict__ s_ptrs,   // [2E]
    __nv_bfloat16* __restrict__ y,          // [E, 2N]
    int n_rows, int k_dim, int n_experts, int act_stride, int scale_stride) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = kFp4Table[threadIdx.x];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 16 + (threadIdx.x >> 5);
    if (n >= n_rows) return;
    const int e = blockIdx.y;
    const int which = blockIdx.z;
    const unsigned char* w = (const unsigned char*)w_ptrs[e * 2 + which]
                           + (long long)n * (k_dim >> 1);
    const unsigned char* ws = (const unsigned char*)s_ptrs[e * 2 + which]
                            + (long long)n * (k_dim >> 5);
    // gate/up share one activation row and one scale row across experts (stride 0);
    // the down projection consumes a per-expert [E, K] pair (stride K and K/32)
    const float acc = gemv_row<0>(a + (long long)e * act_stride,
                                  as + (long long)e * scale_stride,
                                  w, ws, k_dim, lane, lut);
    if (lane == 0) y[e * 2 * n_rows + which * n_rows + n] = __float2bfloat16(acc);
}

extern "C" __global__ void moe_gemv6_bf16(
    const __nv_bfloat16* __restrict__ a,    // [K] bf16
    const unsigned char* __restrict__ as,   // unused on this path (no block scale)
    const long long* __restrict__ w_ptrs,   // [2E]
    const long long* __restrict__ s_ptrs,   // [2E]
    __nv_bfloat16* __restrict__ y,          // [E, 2N]
    int n_rows, int k_dim, int n_experts, int act_stride) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = kFp4Table[threadIdx.x];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 16 + (threadIdx.x >> 5);
    if (n >= n_rows) return;
    const int e = blockIdx.y;
    const int which = blockIdx.z;
    const unsigned char* w = (const unsigned char*)w_ptrs[e * 2 + which]
                           + (long long)n * (k_dim >> 1);
    const unsigned char* ws = (const unsigned char*)s_ptrs[e * 2 + which]
                            + (long long)n * (k_dim >> 5);
    // bf16 has no per-block scale, so only the activation row moves with the expert
    const float acc = gemv_row<1>((const unsigned char*)a + (long long)e * act_stride,
                                  as, w, ws, k_dim, lane, lut);
    if (lane == 0) y[e * 2 * n_rows + which * n_rows + n] = __float2bfloat16(acc);
}

// gate/up -> activation: one launch instead of the two bf16->fp32 copies, two clamps,
// the silu and the routing scale that the torch form spent per expert.
extern "C" __global__ void moe_act6(
    const __nv_bfloat16* __restrict__ gu,   // [E, 2N]
    const float* __restrict__ wexp,         // [E] routing weights
    __nv_bfloat16* __restrict__ out,        // [E, N]
    int n, float limit, int n_experts) {
    const int e = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gu[e * 2 * n + i]);
    float u = __bfloat162float(gu[e * 2 * n + n + i]);
    if (limit > 0.f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    // silu(g) = g * sigmoid(g); the division form stays finite at both tails
    const float s = g / (1.f + __expf(-g));
    out[e * n + i] = __float2bfloat16(s * u * wexp[e]);
}

// gate/up -> quantised activation, all in one launch.
//
// This has to reproduce the reference arithmetic exactly, because the down projection
// consumes an fp8 e4m3 activation with one e8m0 scale per 32 and a 3-bit mantissa is
// coarse: feeding the fused path a bf16 activation instead moved the final logits by
// 52% (measured) even though gate, up and down were each within a bf16 ulp on their
// own. Two details matter and both are in official_kernel_ref.act_quant:
//   * the value is rounded through bf16 first (the torch form was y.to(bf16) before
//     act_quant read it), so the fused form rounds through bf16 too;
//   * the scale is the power of two at or above amax/448 (round_scale=True), with amax
//     floored at 1e-4 -- not amax/448 itself.
__device__ __forceinline__ float warp_amax(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

extern "C" __global__ void moe_act_q6(
    const __nv_bfloat16* __restrict__ gu,   // [E, 2N]
    const float* __restrict__ wexp,         // [E] routing weights
    unsigned char* __restrict__ aq,         // [E, N] fp8 e4m3
    unsigned char* __restrict__ asc,        // [E, N/32] e8m0 bytes
    int n, float limit, int n_exp) {
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.x;               // one 32-wide quantisation group
    const int e = blockIdx.y;
    const int nb = n >> 5;
    const int i = (g << 5) + lane;

    float v = 0.f;
    if (i < n) {
        float gt = __bfloat162float(gu[e * 2 * n + i]);
        float u = __bfloat162float(gu[e * 2 * n + n + i]);
        if (limit > 0.f) {
            u = fminf(fmaxf(u, -limit), limit);
            gt = fminf(gt, limit);
        }
        v = (gt / (1.f + __expf(-gt))) * u * wexp[e];
        v = __bfloat162float(__float2bfloat16(v));
    }
    // amax over the group, floored as the reference does
    float amax = warp_amax(fabsf(v));
    amax = fmaxf(amax, 1e-4f);
    // s = 2^ceil(log2(amax / 448)), exactly: the float's exponent field gives
    // floor(log2), and the mantissa says whether to step up
    const float t = amax * (1.f / 448.f);
    int ex = (int)(__float_as_uint(t) >> 23) - 127;
    if (__uint_as_float((unsigned)(ex + 127) << 23) < t) ++ex;
    const float s = __uint_as_float((unsigned)(ex + 127) << 23);
    if (i < n) {
        aq[e * n + i] = (unsigned char)__nv_cvt_float_to_fp8(v / s, __NV_SATFINITE,
                                                             __NV_E4M3);
    }
    if (lane == 0) asc[e * nb + g] = (unsigned char)(ex + 127);
}

// sum the per-expert down projections straight into the residual: the six separate
// y += rows[i] it replaces were six read-modify-write passes over the same vector
extern "C" __global__ void moe_acc6(
    const __nv_bfloat16* __restrict__ src,  // [E, stride_e]
    float* __restrict__ dst,                // [n] fp32, accumulated in place
    int n, int n_experts, int stride_e) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = dst[i];
    for (int e = 0; e < n_experts; ++e) acc += __bfloat162float(src[e * stride_e + i]);
    dst[i] = acc;
}
