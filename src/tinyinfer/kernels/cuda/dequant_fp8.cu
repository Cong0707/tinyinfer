// dequant_fp8: fp8 e4m3 weight + block-32 e8m0 scale -> bf16.
//
// Rewritten for bandwidth. The first version gave each thread 16 consecutive elements
// and read one scale byte per thread: 81 GB/s on [8192, 4096], i.e. 21% of this card's
// 384 GB/s, and it is 235 ms of a decode step because wo_a is dequantised per layer.
//
// What was wrong and what replaced it:
//   * the scale byte was re-read by every thread of a 32-element block. Now one thread
//     owns exactly one 32-element block, so the scale is one byte per thread and the 128
//     bytes a warp needs land in one 32-byte sector per row.
//   * 16 elements per thread is too few to amortise the index arithmetic. 32 halves the
//     thread count and doubles the bytes in flight per thread.
//   * the store was `__float2bfloat16(x * scale)` element by element. bf16 stores are
//     now 2-wide (4 bytes) so each thread issues 16 stores instead of 32.
//   * the fp8->float conversion goes through the hardware converter (`__nv_cvt_fp8_to_
//     halfraw`), not a bit-twiddling emulation, on any arch that has it.
//
// Target: one 32-element block per thread, 128 threads per block of 4096 elements.

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

extern "C" __global__ void dequant_fp8(
    const __nv_fp8_e4m3* __restrict__ w,   // [O, I]
    const uint8_t* __restrict__ s,         // [O/32, I/32] e8m0 (the byte IS the exponent)
    __nv_bfloat16* __restrict__ y,         // [O, I]
    int O, int I) {
    // One thread = one 32-element scale block along I.
    const int nbi = I >> 5;                  // scale groups per row
    const long long total = (long long)O * nbi;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x) {
        const int o = (int)(idx / nbi);
        const int bi = (int)(idx % nbi);
        const uint8_t sb = s[(long long)(o >> 5) * nbi + bi];
        // e8m0 stores the biased exponent directly: the byte IS (exp + 127).
        const float scale = __uint_as_float((unsigned int)sb << 23);
        const __nv_fp8_e4m3* src = w + (long long)o * I + (bi << 5);
        __nv_bfloat16* dst = y + (long long)o * I + (bi << 5);
#pragma unroll
        for (int k = 0; k < 32; k += 2) {
            // two bf16 packed into one 4-byte store
            const float a = float(src[k]) * scale;
            const float b = float(src[k + 1]) * scale;
            __nv_bfloat162 packed = __floats2bfloat162_rn(a, b);
            *reinterpret_cast<__nv_bfloat162*>(dst + k) = packed;
        }
    }
}
