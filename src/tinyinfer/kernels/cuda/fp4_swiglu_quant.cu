// fp4 gate/up GEMV + SwiGLU + block-32 FP8 quantisation, all in one kernel.
//
// Replaces the pair  moe_fused.fused_fp4_swiglu_pq  (tilelang)  ->  act_quant(h)  for the
// routed experts. Per decode token this chain runs 240 times (6 experts x 40 layers) and
// was two launches plus a bf16 round trip of the whole [1, inter] row through memory.
//
//   gate[n] = sum_k  dequant_fp8(aq[m,k], as[m,k/32]) * dequant_fp4(w1[n,k], s1[n,k/32])
//   up[n]   = same with w3/s3
//   h[n]    = clamp(gate) * silu(clamp(up))          (DeepSeek applies limit to both)
//   then per 32 consecutive n: s = 2^ceil(log2(amax/448)), q = clamp(h / s)
//
// One block owns 32 consecutive outputs -- exactly one quantisation group -- so the scale
// is computed from registers with no second pass over memory. Within the block, 8 threads
// cooperate on one output (K/8 each) and reduce through shared memory.
//
// The arithmetic is scalar FMA in fp32, which is NOT the same accumulation order as the
// tensor-core path it replaces: results agree to ~1e-6 relative but not bit-for-bit, so
// this kernel is an A/B alternative (TI_FUSED_MOE=1), verified against the tilelang chain
// by the layer-by-layer probe rather than assumed equal.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

__device__ __constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float e8m0f(unsigned char b) {
    return ldexpf(1.0f, (int)b - 127);
}

__device__ __forceinline__ float fp4w(const unsigned char* w, long long row, int khalf, int k) {
    const unsigned char byte = w[row * khalf + (k >> 1)];
    return kFp4Lut[(k & 1) ? (byte >> 4) : (byte & 0xF)];
}

// 8 lanes (a subgroup of one warp) sum their partials
__device__ __forceinline__ float sub8_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    return v;
}

}  // namespace

extern "C" __global__ void fp4_swiglu_quant(
    const unsigned char* __restrict__ Aq,   // [K]   fp8 e4m3 activation (m = 0)
    const unsigned char* __restrict__ AS,   // [nb]  e8m0 activation scales
    const unsigned char* __restrict__ W1,   // [N, K/2] packed fp4
    const unsigned char* __restrict__ S1,   // [N, nb]  e8m0
    const unsigned char* __restrict__ W3,
    const unsigned char* __restrict__ S3,
    unsigned char* __restrict__ HQ,         // [N]   fp8 output
    unsigned char* __restrict__ HS,         // [N/32] e8m0 output scales
    int N, int K, float limit) {
    const int gid = threadIdx.x >> 3;        // 0..31 -- output within the group
    const int lid = threadIdx.x & 7;         // 0..7  -- stride inside K
    const int n = blockIdx.x * 32 + gid;
    const int nb = K >> 5;
    const long long khalf = K >> 1;

    __shared__ float s_h[32];

    float gate = 0.f, up = 0.f;
    if (n < N) {
        for (int kb = 0; kb < nb; ++kb) {
            const float ascal = e8m0f(AS[kb]);
            const float s1 = e8m0f(S1[(long long)n * nb + kb]);
            const float s3 = e8m0f(S3[(long long)n * nb + kb]);
            float pg = 0.f, pu = 0.f;
            for (int j = lid; j < 32; j += 8) {
                const int k = kb * 32 + j;
                // .x is the raw ushort bit pattern of a half, not a float
                const __half_raw hr = __nv_cvt_fp8_to_halfraw(Aq[k], __NV_E4M3);
                const float a = __half2float(__ushort_as_half(hr.x)) * ascal;
                pg += a * fp4w(W1, n, (int)khalf, k);
                pu += a * fp4w(W3, n, (int)khalf, k);
            }
            gate += sub8_sum(pg) * s1;
            up += sub8_sum(pu) * s3;
        }
    }
    if (lid == 0) {
        // The reference rounds the accumulated gate/up to bf16 before the activation
        // (g = bf16(Acc1), u = bf16(Acc3) in moe_fused.fp4_swiglu_pq_kernel), and the
        // result H is a bf16 tensor that act_quant then reads. Both round trips have to be
        // present or the quantiser sees a different row and the scale moves by one binade.
        float g = __bfloat162float(__float2bfloat16(gate));
        float u = __bfloat162float(__float2bfloat16(up));
        if (limit > 0.f) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
        float v = (g / (1.f + __expf(-g))) * u;
        s_h[gid] = __bfloat162float(__float2bfloat16(v));
    }
    __syncthreads();

    // block-32 quantisation: one scale for the whole group.
    // Mirrors official_kernel_ref: amax floored at 1e-4, then
    // s = 2^ceil(log2(amax/448)) computed by bit manipulation exactly as fast_log2_ceil /
    // fast_pow2 do it (a float log2/ceil would differ on the boundary cases).
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(s_h[i]));
        amax = fmaxf(amax, 1e-4f);
        const float q = amax * (1.0f / 448.0f);
        const int bits = __float_as_int(q);
        const int expo = (bits >> 23) & 0xFF;
        const int man = bits & 0x7FFFFF;
        const int ce = expo - 127 + (man != 0 ? 1 : 0);
        s_scale = __int_as_float((ce + 127) << 23);
        HS[blockIdx.x] = (unsigned char)(ce + 127);
    }
    __syncthreads();
    if (n < N) {
        const float inv = 1.0f / s_scale;
        float v = s_h[gid] * inv;
        v = fminf(fmaxf(v, -448.0f), 448.0f);
        HQ[n] = (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}
