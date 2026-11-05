// act_quant: bf16 [N, K] -> fp8 e4m3 [N, K] + e8m0 scale [N, K/32], block=32.
//
// One warp per 32-element group: the 32 lanes ARE the group, so the absmax is one
// shuffle reduce and the scale is computed once, not per element. This is the same
// schedule as vLLM's per_token_group_quant (csrc/quantization/w8a8/fp8/) with the
// UE8M0 template branch taken, which is what this checkpoint needs:
//
//     y_s = absmax / 448;  if ue8m0: y_s = exp2(ceil(log2(max(|y_s|, 1e-10))))
//     q   = clamp(x / y_s, -448, 448)
//
// The previous implementation came from the vendored official kernel.py and therefore
// required tilelang (a JIT that needs nvcc at runtime and compiles on first use).
// Dropping that dependency is the point of this file.
//
// e8m0 stores an exponent only: value = 2^(code - 127), so the code we emit is
// 127 + log2(y_s), and y_s is a power of two by construction.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstring>

namespace {

// ceil(log2(x)) and 2^e by IEEE-754 bit manipulation, copied from the reference
// (official_kernel_ref.fast_log2_ceil / fast_pow2). This is NOT the same as
// ceilf(log2f(x)) / exp2f(e): the intrinsic has rounding error at exact powers of two,
// so a mantissa of 0 vs 1ulp flips the result by a whole binade. Measured: using the
// intrinsics changed the end-to-end decode output completely.
__device__ __forceinline__ int fast_log2_ceil(float x) {
    const unsigned int b = __float_as_uint(x);
    const int e = (int)((b >> 23) & 0xFFu);
    const unsigned int man = b & 0x7FFFFFu;
    return e - 127 + (man != 0u ? 1 : 0);
}

__device__ __forceinline__ float fast_pow2(int e) {
    return __uint_as_float((unsigned int)((e + 127) << 23));
}

__device__ __forceinline__ float round_scale(float amax) {
    // 448 is the largest finite e4m3 value; the reference multiplies by its
    // reciprocal rather than dividing, and that difference is one ulp.
    return fast_pow2(fast_log2_ceil(amax * (1.0f / 448.0f)));
}

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

}  // namespace

// x      [N, K]    bf16, K % 32 == 0
// q      [N, K]    fp8 e4m3
// scale  [N, K/32] fp8 e8m0
extern "C" __global__ void act_quant(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ q,
    __nv_fp8_e8m0* __restrict__ scale,
    int N, int K, float eps) {
    const int lane = threadIdx.x & 31;
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int ngroups = (N * K) >> 5;
    if (gwarp >= ngroups) return;

    const long long base = (long long)gwarp * 32;
    const float v = __bfloat162float(x[base + lane]);

    float amax = warp_max(fabsf(v));
    amax = fmaxf(amax, eps);   // the reference floors at 1e-4
    const float y_s = round_scale(amax);

    const float qv = fminf(fmaxf(v / y_s, -448.0f), 448.0f);
    q[base + lane] = __nv_fp8_e4m3(qv);

    // One store per group, from lane 0. e8m0 is an exponent field only: the stored
    // byte IS the code, value = 2^(code - 127). Passing a float to __nv_fp8_e8m0 would
    // convert it again (it expects the value, and rounds it to the nearest power of
    // two), so the byte is written directly. y_s is already an exact power of two
    // (exp2(ceil(log2(.)))), so log2f(y_s) is an exact integer.
    if (lane == 0) {
        int code = (int)lrintf(log2f(y_s)) + 127;
        if (code < 0) code = 0;
        if (code > 255) code = 255;
        const unsigned char b = (unsigned char)code;
        __nv_fp8_e8m0 v;
        memcpy(&v, &b, 1);
        scale[gwarp] = v;
    }
}
// The inplace form: quantise to e4m3 and immediately dequantise back to bf16, so the
// caller gets bf16 whose values sit on the fp8 grid. That is what the reference's
// inplace=True does (out_dtype = in_dtype), and the KV cache path depends on it: the
// window cache holds the quantised-then-restored values, not the originals.
//
// Kept as a second entry point rather than a flag so the fp8-only path keeps its exact
// code (it is bit-identical to the reference and measured at 398 calls/token).
extern "C" __global__ void act_quant_dequant(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    __nv_fp8_e8m0* __restrict__ scale,
    int N, int K, float eps) {
    const int lane = threadIdx.x & 31;
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int ngroups = (N * K) >> 5;
    if (gwarp >= ngroups) return;

    const long long base = (long long)gwarp * 32;
    const float v = __bfloat162float(x[base + lane]);

    float amax = warp_max(fabsf(v));
    amax = fmaxf(amax, eps);   // the reference floors at 1e-4
    const float y_s = round_scale(amax);

    const float qv = fminf(fmaxf(v / y_s, -448.0f), 448.0f);
    const __nv_fp8_e4m3 q8 = __nv_fp8_e4m3(qv);
    // restore: the stored value is the rounded fp8 code, scaled back up
    y[base + lane] = __float2bfloat16(float(q8) * y_s);

    if (lane == 0) {
        int code = (int)lrintf(log2f(y_s)) + 127;
        if (code < 0) code = 0;
        if (code > 255) code = 255;
        const unsigned char b = (unsigned char)code;
        __nv_fp8_e8m0 vv;
        memcpy(&vv, &b, 1);
        scale[gwarp] = vv;
    }
}