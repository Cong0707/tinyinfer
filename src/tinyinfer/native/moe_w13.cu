// Hand-written decode kernels (no torch headers: nvcc -> cubin, loaded by
// tinyinfer/native.py through the CUDA driver API).
//
// gemv_fp4_w13: one fp4 GEMV covering BOTH the gate (w1) and up (w3) matrices of one
// expert. The tilelang path launches one kernel per matrix (12 per layer for six
// experts); this reads two weight pointers in one launch. The [w1; w3] torch.cat
// variant was tried before and lost 22ms to 768 cat launches -- this avoids the cat
// entirely by branching on the row index.
//
//   packed fp4 E2M1: two values per byte, low nibble first
//   one e8m0 scale per 32 K, pre-converted to fp32 by the caller
//   activation is fp8 e4m3 with its own per-32 scale (already applied by the caller)

#include <cuda_bf16.h>

// E2M1 -> float. bits: sign(1) exp(2) mant(1).
__device__ __constant__ float kFp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float fp4_at(const unsigned char* w, long n, long k, long khalf) {
    const unsigned char byte = w[n * khalf + (k >> 1)];
    const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
    return kFp4[nib];
}

extern "C" __global__ void gemv_fp4_w13(
    const float* __restrict__ act_a,      // [K]   fp8 values widened to float
    const float* __restrict__ act_s,      // [nb]  activation scales (fp32)
    const unsigned char* __restrict__ w1, // [N, K/2] packed fp4
    const float* __restrict__ s1,         // [N, nb]
    const unsigned char* __restrict__ w3,
    const float* __restrict__ s3,
    __nv_bfloat16* __restrict__ out,      // [2N] bf16: first N = w1, next N = w3
    int n_rows, int k_dim) {
    // one block per 16 rows; 32 lanes split K
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;
    const int nb = k_dim >> 5;
    const int half = n_rows;
    const int n0 = blockIdx.x * 16;
    const int n = n0 + r;
    if (n >= 2 * half) return;

    const unsigned char* w = (n < half) ? w1 : w3;
    const float* ws = (n < half) ? s1 : s3;
    const int row = (n < half) ? n : (n - half);
    const long khalf = k_dim >> 1;

    float acc = 0.f;
    for (int kb = 0; kb < nb; ++kb) {
        float part = 0.f;
        for (int j = lane; j < 32; j += 32) {
            const int k = kb * 32 + j;
            part += act_a[k] * fp4_at(w, row, k, khalf);
        }
        acc += part * act_s[kb] * ws[row * nb + kb];
    }
    // warp reduce over the 32 lanes that split K
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[n] = __float2bfloat16(acc);
}
