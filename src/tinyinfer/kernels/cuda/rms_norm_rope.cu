// rms_norm + RoPE in one kernel, for the window KV path.
//
//   y[d]      = rms_norm(x)[d]                       for d < D - rot
//   (y0, y1)  = rotate((y0, y1), freqs[s][i])        for the last rot dims, in pairs
//
// The two ops are fused because nothing reads the normalised row in between. Bit
// exactness is mandatory here -- a previous tilelang version of this fusion is in the
// tree but permanently disabled because its reduction order differed from the reference
// by ~4e-3 -- so the reduction copies vllm_rms_norm.cu exactly: 8-wide vectors, thread i
// owns vector i, the same shuffle tree, the same block size the caller passes
// (min(hidden/8, 1024)). RoPE then reads the bf16-rounded rms output and writes bf16,
// matching apply_rope's float->complex->copy_ round trip.
#include <cuda_bf16.h>

namespace {

constexpr int kVec = 8;  // 16 bytes of bf16

__device__ __forceinline__ float block_reduce_sum(float v, float* s) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (threadIdx.x < nwarps) ? s[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
        if (threadIdx.x == 0) s[0] = t;
    }
    __syncthreads();
    return s[0];
}

}  // namespace

extern "C" __global__ void rms_norm_rope_bf(
    const __nv_bfloat16* __restrict__ x,       // [M, D]
    const __nv_bfloat16* __restrict__ weight,  // [D]
    const float2* __restrict__ freqs,          // [S, rot/2], row = position
    __nv_bfloat16* __restrict__ out,           // [M, D]
    int M, int S, int D, int rot, float eps) {
    extern __shared__ float smem[];
    float* s_warp = smem;          // 32 floats

    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const __nv_bfloat16* row = x + (long long)m * D;
    __nv_bfloat16* dst = out + (long long)m * D;

    float variance = 0.f;
    const int nvec = D / kVec;
    for (int i = tid; i < nvec; i += nthreads) {
        const int d = i * kVec;
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
            const float v = __bfloat162float(row[d + j]);
            variance += v * v;
        }
    }
    for (int d = nvec * kVec + tid; d < D; d += nthreads) {
        const float v = __bfloat162float(row[d]);
        variance += v * v;
    }

    const float total = block_reduce_sum(variance, s_warp);
    __shared__ float s_var;
    if (tid == 0) s_var = rsqrtf(total / (float)D + eps);
    __syncthreads();
    const float r = s_var;

    const int base = D - rot;
    // tail (rope) region: handled below, skip it here
    for (int i = tid; i < nvec; i += nthreads) {
        const int d = i * kVec;
        if (d + kVec <= base) {
#pragma unroll
            for (int j = 0; j < kVec; ++j)
                dst[d + j] = __float2bfloat16(
                    __bfloat162float(row[d + j]) * r * __bfloat162float(weight[d + j]));
        }
    }
    for (int d = nvec * kVec + tid; d < base; d += nthreads)
        dst[d] = __float2bfloat16(
            __bfloat162float(row[d]) * r * __bfloat162float(weight[d]));

    // RoPE over the last rot dims, in adjacent pairs, reading the rms output (bf16)
    const int pos = (S > 0) ? (m % S) : 0;
    const int npair = rot / 2;
    const float2* fr = freqs + (long long)pos * npair;
    for (int i = tid; i < npair; i += nthreads) {
        const int d0 = base + 2 * i;
        const int d1 = d0 + 1;
        const float a = __bfloat162float(
            __float2bfloat16(__bfloat162float(row[d0]) * r * __bfloat162float(weight[d0])));
        const float b = __bfloat162float(
            __float2bfloat16(__bfloat162float(row[d1]) * r * __bfloat162float(weight[d1])));
        const float2 f = fr[i];
        dst[d0] = __float2bfloat16(a * f.x - b * f.y);
        dst[d1] = __float2bfloat16(a * f.y + b * f.x);
    }
}
