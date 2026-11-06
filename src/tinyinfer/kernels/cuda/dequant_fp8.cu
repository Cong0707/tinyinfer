// dequant_fp8: fp8 e4m3 weight + block-32 e8m0 scale -> bf16.
//
// Rewritten twice for bandwidth. The first version gave each thread 16 elements with a
// per-thread scale read: 81 GB/s. The second gave each thread a whole 32-element block:
// 191-207 GB/s. This one attacks what is left, which is the STORE.
//
// The traffic is asymmetric: 33.5 MB read (fp8) against 67 MB written (bf16), and the
// second version emitted the writes as sixteen 4-byte __nv_bfloat162 stores per thread.
// A 4-byte store per thread means a warp covers 128 bytes per instruction; the widest
// store the ISA offers is 16 bytes, so a warp can cover 512. Four __floats2bfloat162
// pairs assembled in registers and written as uint4 cut the store instruction count
// from 16 to 4 and quadruple the bytes per store.
//
// __stcs (streaming) for the store: the output is consumed once by the next kernel and
// should not evict the weight tiles and activations that are still live in L2.

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

extern "C" __global__ void dequant_fp8(
    const __nv_fp8_e4m3* __restrict__ w,   // [O, I]
    const uint8_t* __restrict__ s,         // [O/32, I/32] e8m0 (the byte IS the exponent)
    __nv_bfloat16* __restrict__ y,         // [O, I]
    int O, int I) {
    const int nbi = I >> 5;                  // scale groups per row
    const long long total = (long long)O * nbi;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x) {
        const int o = (int)(idx / nbi);
        const int bi = (int)(idx % nbi);
        const uint8_t sb = s[(long long)(o >> 5) * nbi + bi];
        // e8m0 stores the biased exponent directly: the byte IS (exp + 127).
        const float scale = __uint_as_float((unsigned int)sb << 23);
        const long long off = (long long)o * I + ((long long)bi << 5);
        const __nv_fp8_e4m3* src = w + off;
        __nv_bfloat16* dst = y + off;

        // 32 fp8 -> 32 bf16 -> four 16-byte stores.
        uint4 q[4];
        uint32_t* qw = reinterpret_cast<uint32_t*>(q);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const __nv_bfloat162 lo = __floats2bfloat162_rn(
                float(src[2 * k]) * scale, float(src[2 * k + 1]) * scale);
            qw[k] = *reinterpret_cast<const uint32_t*>(&lo);
        }
        uint4* dv = reinterpret_cast<uint4*>(dst);
        __stcs(dv + 0, q[0]);
        __stcs(dv + 1, q[1]);
        __stcs(dv + 2, q[2]);
        __stcs(dv + 3, q[3]);
    }
}
