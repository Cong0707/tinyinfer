// fp4_quant: bf16 -> E2M1 packed (2 values/byte) + per-block scale.
//
// Replaces tilelang fp4_quant_kernel (official_kernel_ref) for two callers:
//   - the indexer, block 32, e8m0 scales (power of two)
//   - the compressed KV cache, block 16, e4m3 scales
// plus their inplace form, which stores the quantised-then-restored bf16 (that value is
// what the reference cache holds, so it is part of the result, not an optimisation).
//
// Layout: one warp per group. A group of G elements is G/2 BYTES, so lane i owns the byte
// covering elements (2i, 2i+1) and writes it once. An earlier draft had lane i own element
// i and do a read-modify-write of a shared byte -- two lanes racing on the same address.
// Lanes past G/2 still take part in the warp reduce (with a zero) and simply do not store.
//
// Rounding rules, both copied from the reference:
//   e8m0: amax floored at 6*2^-126, s = 2^ceil(log2(amax/6))     (bit ops, see act_quant)
//   e4m3: amax floored at 6*2^-9,   s = float(fp8(amax/6))       (cast chain, not a divide)
// Value: clamp(x/s, -6, 6) rounded to E2M1 with ties-to-even. Tie points are 0.25, 0.75,
// 1.25, 1.75, 2.5, 3.5, 5.0 and each resolves to the code with an even LSB.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ int fast_log2_ceil_f(float x) {
    const unsigned int b = __float_as_uint(x);
    const int e = (int)((b >> 23) & 0xFFu);
    const unsigned int man = b & 0x7FFFFFu;
    return e - 127 + (man != 0u ? 1 : 0);
}

__device__ __forceinline__ float fast_pow2_f(int e) {
    return __uint_as_float((unsigned int)((e + 127) << 23));
}

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ unsigned int f2fp4(float v) {
    const float a = fabsf(v);
    unsigned int code;
    if (a <= 0.25f) code = 0u;
    else if (a < 0.75f) code = 1u;
    else if (a <= 1.25f) code = 2u;
    else if (a < 1.75f) code = 3u;
    else if (a <= 2.5f) code = 4u;
    else if (a < 3.5f) code = 5u;
    else if (a <= 5.0f) code = 6u;
    else code = 7u;
    // signbit, not v < 0: E2M1 has a signed zero (code 8) and the reference's cast keeps
    // it. Using a comparison turned -0.0 into +0.0, which showed up as 28 differing bf16
    // values in the restore path even though every printed magnitude matched.
    return signbit(v) ? (code | 8u) : code;
}

__device__ __constant__ float kFp4Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    // -0.0f, not 0.0f: the reference restores float(fp4_code) * s, and code 8 is a
    // SIGNED zero, so the product is -0.0 for a positive scale.
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// scale for one group; every lane returns the same s and code
__device__ __forceinline__ void group_scale(float amax, int use_e8m0,
                                            float* p_s, unsigned int* p_code) {
    if (use_e8m0) {
        amax = fmaxf(amax, 6.0f * exp2f(-126.0f));
        const int e = fast_log2_ceil_f(amax * (1.0f / 6.0f));
        *p_s = fast_pow2_f(e);
        *p_code = (unsigned int)(e + 127);
    } else {
        amax = fmaxf(amax, 6.0f * exp2f(-9.0f));
        const __nv_fp8_e4m3 q8 = __nv_fp8_e4m3(amax * (1.0f / 6.0f));
        *p_s = float(q8);
        *p_code = (unsigned int)q8.__x;
    }
}

}  // namespace

// x [M,K] bf16, y [M,K/2] packed, scale [M,K/gsize] one byte each
extern "C" __global__ void fp4_quant(
    const __nv_bfloat16* __restrict__ x,
    unsigned char* __restrict__ y,
    unsigned char* __restrict__ scale,
    int M, int K, int gsize, int use_e8m0) {
    const int lane = threadIdx.x & 31;
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int ngroups = (M * K) / gsize;
    if (gwarp >= ngroups) return;

    const long long gbase = (long long)gwarp * gsize;
    const int nbytes = gsize >> 1;
    const bool active = lane < nbytes;

    float amax = 0.f;
    if (active) {
        amax = fmaxf(fabsf(__bfloat162float(x[gbase + 2 * lane])),
                     fabsf(__bfloat162float(x[gbase + 2 * lane + 1])));
    }
    amax = warp_max(amax);

    float s; unsigned int code;
    group_scale(amax, use_e8m0, &s, &code);

    if (active) {
        const unsigned int lo = f2fp4(fminf(fmaxf(__bfloat162float(x[gbase + 2 * lane]) / s, -6.0f), 6.0f));
        const unsigned int hi = f2fp4(fminf(fmaxf(__bfloat162float(x[gbase + 2 * lane + 1]) / s, -6.0f), 6.0f));
        y[gbase / 2 + lane] = (unsigned char)((lo & 0x0Fu) | ((hi & 0x0Fu) << 4));
    }
    if (lane == 0) scale[gwarp] = (unsigned char)code;
}

extern "C" __global__ void fp4_quant_dequant(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    unsigned char* __restrict__ scale,
    int M, int K, int gsize, int use_e8m0) {
    const int lane = threadIdx.x & 31;
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int ngroups = (M * K) / gsize;
    if (gwarp >= ngroups) return;

    const long long gbase = (long long)gwarp * gsize;
    const bool active = lane < gsize;

    float amax = 0.f;
    if (active) amax = fabsf(__bfloat162float(x[gbase + lane]));
    amax = warp_max(amax);

    float s; unsigned int code;
    group_scale(amax, use_e8m0, &s, &code);

    if (active) {
        const unsigned int c = f2fp4(fminf(fmaxf(__bfloat162float(x[gbase + lane]) / s, -6.0f), 6.0f));
        y[gbase + lane] = __float2bfloat16(kFp4Lut[c] * s);
    }
    if (lane == 0) scale[gwarp] = (unsigned char)code;
}