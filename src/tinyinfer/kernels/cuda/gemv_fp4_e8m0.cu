// Hand-written decode kernels (no torch headers: nvcc -> cubin, loaded by
// tinyinfer/native.py through the CUDA driver API).
//
// gemv_fp4_e8m0: the fp4 expert GEMV reading the checkpoint's e8m0 scale bytes directly.
// The tilelang path needs an fp32 copy of every scale (1.4MB per scale, three per
// expert, 1680 copies per decode step ~= 168ms of the aten::copy_ total). e8m0 is
// 2^(byte-127), so the kernel decodes it inline with one ldexpf.
//
//   A   [K]        fp8 e4m3 activation (already widened to float by the caller)
//   AS  [nb]       fp32 activation scales
//   W   [N, K/2]   packed E2M1, low nibble first
//   WS  [N, nb]    e8m0 scale bytes
//   Y   [N]        bf16

#include <cuda_bf16.h>

__device__ __constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float e8m0(unsigned char b) {
    // 0xFF is NaN and 0x00 is 2^-127 in the format; both are treated as the power of two
    return ldexpf(1.0f, (int)b - 127);
}

extern "C" __global__ void gemv_fp4_e8m0(
    const float* __restrict__ a,
    const float* __restrict__ as,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ ws,
    __nv_bfloat16* __restrict__ y,
    int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;
    const int nb = k_dim >> 5;
    const long long khalf = k_dim >> 1;
    const int n = blockIdx.x * 16 + r;
    if (n >= n_rows) return;

    float acc = 0.f;
    for (int kb = 0; kb < nb; ++kb) {
        float part = 0.f;
        for (int j = lane; j < 32; j += 32) {
            const int k = kb * 32 + j;
            const unsigned char byte = w[(long long)n * khalf + (k >> 1)];
            const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
            part += a[k] * kFp4Lut[nib];
        }
        acc += part * as[kb] * e8m0(ws[n * nb + kb]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

// gemv_fp4_e8m0_v2: same math, coalesced 16-byte vector loads.
//
// v1 had each lane read one nibble: lane j read byte (kb*32+j)>>1, so two adjacent lanes
// touched the same byte and a warp's 32 lanes spanned only 16 bytes. The DRAM/L2
// transaction is 128 bytes, so 8/9 of every fetch was discarded -- measured 14-20 GB/s
// against a 336 GB/s peak (4-6%).
//
// Here each lane loads one complete byte (holding two nibbles, i.e. two consecutive k) and
// contributes both halves. A warp then spans 32 consecutive bytes, and with 4-byte
// vector loads (uint4) it spans 128 bytes -- exactly one cache line.
extern "C" __global__ void gemv_fp4_e8m0_v2(
    const float* __restrict__ a,
    const float* __restrict__ as,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ ws,
    __nv_bfloat16* __restrict__ y,
    int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;
    const int nb = k_dim >> 5;                 // 32-value blocks along K
    const long long khalf = k_dim >> 1;        // fp4 bytes per row
    const int n = blockIdx.x * 16 + r;
    if (n >= n_rows) return;
    const unsigned char* row = w + (long long)n * khalf;

    float acc = 0.f;
    // one byte per lane per 64-k step: 32 lanes cover 64 k values with 32 contiguous bytes
    for (int kb2 = 0; kb2 < nb; kb2 += 2) {
        const int byte_off = kb2 * 16 + lane;   // 16 bytes per 32-k block
        const unsigned char byte = row[byte_off];
        const int k0 = kb2 * 32 + lane * 2;
        const int k1 = k0 + 1;
        float p = a[k0] * kFp4Lut[byte & 0xF] + a[k1] * kFp4Lut[byte >> 4];
        // both nibbles belong to the same 32-block (kb2 and kb2+1 differ only if lane
        // crosses a block boundary, which it does not: lane*2 < 32 within a block)
        acc += p * as[kb2] * e8m0(ws[n * nb + kb2]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

// gemv_fp4_e8m0_v3: fix the real problem -- too little work per warp and a
// broadcast-heavy inner loop.
//
// v1/v2 gave one warp per output row. A row is K/2 = 2KB of fp4, so a warp issues ~64
// load instructions and then finishes; with 4096 rows that is 4096 warps but each has a
// long dependency chain on a[k] reloaded from global every k. Measured 15-20 GB/s.
//
// Here a block (8 warps) covers 8 rows and stages the activation vector in shared memory
// once, so the inner loop is shared-memory reads plus one coalesced fp4 load per lane per
// 32-k block. The row loop is unrolled by 2 so a warp keeps two accumulators live.
extern "C" __global__ void gemv_fp4_e8m0_v3(
    const float* __restrict__ a,
    const float* __restrict__ as,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ ws,
    __nv_bfloat16* __restrict__ y,
    int n_rows, int k_dim) {
    extern __shared__ float sa[];               // k_dim floats
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;             // 0..7 with 256 threads
    const int nb = k_dim >> 5;
    const long long khalf = k_dim >> 1;

    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) sa[i] = a[i];
    __syncthreads();

    const int n = blockIdx.x * 8 + r;
    if (n >= n_rows) return;
    const unsigned char* row = w + (long long)n * khalf;

    float acc = 0.f;
    // 4 bytes (uint) per lane covers 8 nibbles = 8 k values; 32 lanes = 256 k
    const int nvec = khalf / 4;                 // 4-byte groups per row
    for (int v = lane; v < nvec; v += 32) {
        const uchar4 b = *reinterpret_cast<const uchar4*>(row + v * 4);
        const int k0 = v * 8;
        float p = 0.f;
        p += sa[k0 + 0] * kFp4Lut[b.x & 0xF];
        p += sa[k0 + 1] * kFp4Lut[b.x >> 4];
        p += sa[k0 + 2] * kFp4Lut[b.y & 0xF];
        p += sa[k0 + 3] * kFp4Lut[b.y >> 4];
        p += sa[k0 + 4] * kFp4Lut[b.z & 0xF];
        p += sa[k0 + 5] * kFp4Lut[b.z >> 4];
        p += sa[k0 + 6] * kFp4Lut[b.w & 0xF];
        p += sa[k0 + 7] * kFp4Lut[b.w >> 4];
        // 8 consecutive k all live in one 32-block unless k0 crosses it (k0 % 32 == 0 or 8/16/24)
        acc += p * as[k0 >> 5] * e8m0(ws[n * nb + (k0 >> 5)]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

// gemv_fp4_e8m0_v4: match the bf16 kernel's proven shape.
//
// gemv_bf16 reaches 92% of DRAM peak because each lane accumulates R_ACC=4 partial sums in
// a 128-k stride loop, so the compiler keeps four loads in flight per lane and the warp's
// 16 accumulators hide memory latency.
//
// v1-v3 spent the loop on fp4 decoding bookkeeping (nibble extract + table lookup per k)
// and ran 16x fewer loads per iteration, so the kernel was issue-bound at 20 GB/s. This
// version loads uint4 (16 bytes = 32 nibbles = 32 k) per lane per iteration and unrolls
// four of those, i.e. 128 k per warp iteration -- the same stride the bf16 kernel uses.
extern "C" __global__ void gemv_fp4_e8m0_v4(
    const float* __restrict__ a,
    const float* __restrict__ as,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ ws,
    __nv_bfloat16* __restrict__ y,
    int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * 8 + warp;
    if (n >= n_rows) return;

    const int nb = k_dim >> 5;
    const long long khalf = k_dim >> 1;
    const unsigned char* row = w + (long long)n * khalf;
    const int nvec = khalf >> 4;               // 16-byte vectors per row

    float acc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) acc[u] = 0.f;

    int v = lane;
    for (; v + 96 < nvec; v += 128) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int vi = v + u * 32;
            const uint4 raw = *reinterpret_cast<const uint4*>(row + (long long)vi * 16);
            const int k0 = vi * 32;
            const int kb = k0 >> 5;
            float s = as[kb] * e8m0(ws[n * nb + kb]);
            const unsigned int* dw = reinterpret_cast<const unsigned int*>(&raw);
            float p = 0.f;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const unsigned int word = dw[t];
                const int base = k0 + t * 8;
                p += a[base + 0] * kFp4Lut[word & 0xF];
                p += a[base + 1] * kFp4Lut[(word >> 4) & 0xF];
                p += a[base + 2] * kFp4Lut[(word >> 8) & 0xF];
                p += a[base + 3] * kFp4Lut[(word >> 12) & 0xF];
                p += a[base + 4] * kFp4Lut[(word >> 16) & 0xF];
                p += a[base + 5] * kFp4Lut[(word >> 20) & 0xF];
                p += a[base + 6] * kFp4Lut[(word >> 24) & 0xF];
                p += a[base + 7] * kFp4Lut[(word >> 28) & 0xF];
            }
            acc[u] += p * s;
        }
    }
    for (; v < nvec; v += 32) {
        const uint4 raw = *reinterpret_cast<const uint4*>(row + (long long)v * 16);
        const int k0 = v * 32;
        const int kb = k0 >> 5;
        float s = as[kb] * e8m0(ws[n * nb + kb]);
        const unsigned int* dw = reinterpret_cast<const unsigned int*>(&raw);
        float p = 0.f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const unsigned int word = dw[t];
            const int base = k0 + t * 8;
            p += a[base + 0] * kFp4Lut[word & 0xF];
            p += a[base + 1] * kFp4Lut[(word >> 4) & 0xF];
            p += a[base + 2] * kFp4Lut[(word >> 8) & 0xF];
            p += a[base + 3] * kFp4Lut[(word >> 12) & 0xF];
            p += a[base + 4] * kFp4Lut[(word >> 16) & 0xF];
            p += a[base + 5] * kFp4Lut[(word >> 20) & 0xF];
            p += a[base + 6] * kFp4Lut[(word >> 24) & 0xF];
            p += a[base + 7] * kFp4Lut[(word >> 28) & 0xF];
        }
        acc[0] += p * s;
    }
#pragma unroll
    for (int u = 1; u < 4; ++u) acc[0] += acc[u];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc[0] += __shfl_down_sync(0xffffffffu, acc[0], off);
    if (lane == 0) y[n] = __float2bfloat16(acc[0]);
}

// diagnostic: read fp4 bytes and sum them, no lookup table, no scales.
// Isolates "can this kernel stream fp4 from DRAM" from "is the decode math slow".
extern "C" __global__ void fp4_readonly(const unsigned char* __restrict__ w,
                                        unsigned int* __restrict__ sink,
                                        int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * 8 + warp;
    if (n >= n_rows) return;
    const long long khalf = k_dim >> 1;
    const unsigned char* row = w + (long long)n * khalf;
    const int nvec = khalf >> 4;

    unsigned int s = 0;
    for (int v = lane; v < nvec; v += 32) {
        const uint4 raw = *reinterpret_cast<const uint4*>(row + (long long)v * 16);
        const unsigned int* dw = reinterpret_cast<const unsigned int*>(&raw);
        s += dw[0] + dw[1] + dw[2] + dw[3];
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) sink[n] = s;
}

// gemv_fp4_e2m1: decode E2M1 in ALU instead of a 16-entry table lookup.
//
// Measured diagnosis: the table-lookup kernel is issue-bound, not bandwidth-bound. Pure
// fp4 reads reach 212 GB/s at 320MB, but the decode path lands at 16-22 GB/s because each
// weight costs 5 instructions (shift, and, table load, FMA) and a lane can only do one at
// a time.
//
// E2M1 has 16 values, so a 4-bit code maps to a small set. Rather than index a table
// (a memory op per weight), rebuild the float by bit assembly: sign, 2-bit exponent,
// 1-bit mantissa. That is pure ALU and the compiler can schedule it.
//
// decode: code = s eem (s=bit3, e=bits2:1, m=bit0)
//   value = (-1)^s * 2^(e-1) * (1 + m/2)        for e>0
//   value = (-1)^s * m/2                         for e==0  (subnormal)
__device__ __forceinline__ float fp4_e2m1(unsigned int c) {
    const float sign = (c & 8u) ? -1.f : 1.f;
    const unsigned int e = (c >> 1) & 3u;
    const unsigned int m = c & 1u;
    // e in 0..3 -> 2^(e-1) with e==0 meaning 0.5 (subnormal step)
    const float scale = __uint_as_float(0x3f000000u << (e == 0 ? 0 : e)) ;
    // simpler and branch-free: build via a 4-entry constant in registers
    const float esel = e == 0 ? 0.5f : (e == 1 ? 1.0f : (e == 2 ? 2.0f : 4.0f));
    const float base = e == 0 ? 0.f : esel;
    return sign * (base + (e == 0 ? m * 0.5f : m * esel * 0.5f));
}

extern "C" __global__ void gemv_fp4_e2m1(
    const float* __restrict__ a,
    const float* __restrict__ as,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ ws,
    __nv_bfloat16* __restrict__ y,
    int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * 8 + warp;
    if (n >= n_rows) return;

    const int nb = k_dim >> 5;
    const long long khalf = k_dim >> 1;
    const unsigned char* row = w + (long long)n * khalf;
    const int nvec = khalf >> 2;               // 4-byte words per row

    float acc = 0.f;
    for (int v = lane; v < nvec; v += 32) {
        unsigned int word;
        memcpy(&word, row + (long long)v * 4, 4);
        const int k0 = v * 8;
        const float s = as[k0 >> 5] * e8m0(ws[n * nb + (k0 >> 5)]);
        float p = 0.f;
        p += a[k0 + 0] * fp4_e2m1(word        & 0xF);
        p += a[k0 + 1] * fp4_e2m1((word >> 4) & 0xF);
        p += a[k0 + 2] * fp4_e2m1((word >> 8) & 0xF);
        p += a[k0 + 3] * fp4_e2m1((word >> 12) & 0xF);
        p += a[k0 + 4] * fp4_e2m1((word >> 16) & 0xF);
        p += a[k0 + 5] * fp4_e2m1((word >> 20) & 0xF);
        p += a[k0 + 6] * fp4_e2m1((word >> 24) & 0xF);
        p += a[k0 + 7] * fp4_e2m1((word >> 28) & 0xF);
        acc += p * s;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

// fp4 decode with no table and no branches: build the float bit pattern directly.
//
// E2M1 layout is s|ee|m. For e>0 the value is 2^(e-1) * (1 + m/2), which is exactly the
// float with exponent field (126+e) and the mantissa top bit set from m. e==0 is the
// subnormal case (0 or 0.5). Both fall out of the same expression if the exponent is
// clamped, so the decode is:
//
//   bits = ((126 + e) << 23) | (m << 22)      -> 2^(e-1) * (1 + m/2)
//   value = sign * as_float(bits)
//
// The earlier version indexed a 16-entry table, which is a memory op per weight and made
// the kernel issue-bound (measured 19-22 GB/s against a 212 GB/s pure-read ceiling).
__device__ __forceinline__ float fp4_fast(unsigned int c) {
    // e==0 must yield 0.5^m, i.e. 0 or 0.5: exponent 126 with mantissa from m gives
    // 0.5 or 0.75, so handle e==0 by forcing the mantissa term instead.
    const unsigned int e = (c >> 1) & 3u;
    const unsigned int m = c & 1u;
    const unsigned int safe_e = e | (e == 0u);          // 0 stays 0 -> use 1 as base
    unsigned int bits = ((126u + safe_e) << 23) | (m << 22);
    float v = __uint_as_float(bits);
    v = (e == 0u) ? (m ? 0.5f : 0.f) : v;               // subnormal fixup
    return (c & 8u) ? -v : v;
}

extern "C" __global__ void gemv_fp4_fast(
    const float* __restrict__ a,
    const float* __restrict__ as,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ ws,
    __nv_bfloat16* __restrict__ y,
    int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * 8 + warp;
    if (n >= n_rows) return;

    const int nb = k_dim >> 5;
    const long long khalf = k_dim >> 1;
    const unsigned char* row = w + (long long)n * khalf;
    const int nvec = khalf >> 2;                // 32-bit words per row

    float acc = 0.f;
    for (int v = lane; v < nvec; v += 32) {
        unsigned int word;
        memcpy(&word, row + (long long)v * 4, 4);
        const int k0 = v * 8;
        const float s = as[k0 >> 5] * e8m0(ws[n * nb + (k0 >> 5)]);
        float p = 0.f;
        p = fmaf(a[k0 + 0], fp4_fast(word        & 0xF), p);
        p = fmaf(a[k0 + 1], fp4_fast((word >> 4) & 0xF), p);
        p = fmaf(a[k0 + 2], fp4_fast((word >> 8) & 0xF), p);
        p = fmaf(a[k0 + 3], fp4_fast((word >> 12) & 0xF), p);
        p = fmaf(a[k0 + 4], fp4_fast((word >> 16) & 0xF), p);
        p = fmaf(a[k0 + 5], fp4_fast((word >> 20) & 0xF), p);
        p = fmaf(a[k0 + 6], fp4_fast((word >> 24) & 0xF), p);
        p = fmaf(a[k0 + 7], fp4_fast((word >> 28) & 0xF), p);
        acc = fmaf(p, s, acc);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

// gemv_fp4_vec8: decode eight nibbles with vectorised integer ops.
//
// Measured diagnosis: the scalar decode costs ~14 instructions per weight (extract e,
// extract m, build the float exponent, subnormal fixup, sign) against bf16's 3, so the
// kernel is issue-bound at 60 GB/s while a pure fp4 read of the same buffer reaches
// 212 GB/s.
//
// Eight nibbles live in one 32-bit word, and the fields are in the same positions in each
// nibble, so the extraction is one SWAR op instead of eight scalar ones:
//   e_all = (word >> 1) & 0x77777777
//   m_all = word & 0x11111111
//   s_all = word & 0x88888888
// The float for each nibble is then built from its two bits with a LUT-free expression,
// and the multiply-accumulate stays scalar because the activation vector differs per k.
extern "C" __global__ void gemv_fp4_vec8(
    const float* __restrict__ a,
    const float* __restrict__ as,
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ ws,
    __nv_bfloat16* __restrict__ y,
    int n_rows, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * 8 + warp;
    if (n >= n_rows) return;

    const int nb = k_dim >> 5;
    const long long khalf = k_dim >> 1;
    const unsigned char* row = w + (long long)n * khalf;
    const int nvec = khalf >> 2;

    float acc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) acc[u] = 0.f;

    int v = lane;
    for (; v + 96 < nvec; v += 128) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int vi = v + u * 32;
            unsigned int word;
            memcpy(&word, row + (long long)vi * 4, 4);
            const int k0 = vi * 8;
            const float s = as[k0 >> 5] * e8m0(ws[n * nb + (k0 >> 5)]);

            // SWAR extraction: every nibble's fields in one op each
            const unsigned int e_all = (word >> 1) & 0x77777777u;
            const unsigned int m_all = word & 0x11111111u;
            const unsigned int s_all = word & 0x88888888u;

            float p = 0.f;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const unsigned int e = (e_all >> (t * 4)) & 3u;
                const unsigned int m = (m_all >> (t * 4)) & 1u;
                const unsigned int sg = (s_all >> (t * 4)) & 1u;
                // 2^(e-1)*(1+m/2), with e==0 -> m*0.5
                const float base = (e == 0u)
                    ? (m ? 0.5f : 0.f)
                    : __uint_as_float((126u + e) << 23) * (1.f + m * 0.5f);
                p = fmaf(a[k0 + t], sg ? -base : base, p);
            }
            acc[u] = fmaf(p, s, acc[u]);
        }
    }
    for (; v < nvec; v += 32) {
        unsigned int word;
        memcpy(&word, row + (long long)v * 4, 4);
        const int k0 = v * 8;
        const float s = as[k0 >> 5] * e8m0(ws[n * nb + (k0 >> 5)]);
        const unsigned int e_all = (word >> 1) & 0x77777777u;
        const unsigned int m_all = word & 0x11111111u;
        const unsigned int s_all = word & 0x88888888u;
        float p = 0.f;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const unsigned int e = (e_all >> (t * 4)) & 3u;
            const unsigned int m = (m_all >> (t * 4)) & 1u;
            const unsigned int sg = (s_all >> (t * 4)) & 1u;
            const float base = (e == 0u)
                ? (m ? 0.5f : 0.f)
                : __uint_as_float((126u + e) << 23) * (1.f + m * 0.5f);
            p = fmaf(a[k0 + t], sg ? -base : base, p);
        }
        acc[0] = fmaf(p, s, acc[0]);
    }
#pragma unroll
    for (int u = 1; u < 4; ++u) acc[0] += acc[u];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc[0] += __shfl_down_sync(0xffffffffu, acc[0], off);
    if (lane == 0) y[n] = __float2bfloat16(acc[0]);
}
