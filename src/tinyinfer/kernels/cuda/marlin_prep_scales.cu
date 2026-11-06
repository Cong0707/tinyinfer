// marlin_prep_scales: the scale-layout rewrite MarlinFp4Weight used to run as three
// torch passes (transpose, 8x8 block permute, 0,2,1,3 sub-block permute). Measured on
// [2304,5120] those three cost 85 us of a 133 us repack, and all of it is index
// arithmetic on a 360KB buffer.
//
// Input  s: [N, K/32] e8m0 bytes (row-major, the checkpoint layout).
// Output o: [K/32, N], element for element the same as the reference chain.
//
// The map was read back from the reference rather than derived: feed it an arange and
// record where each output element came from. What it shows is that BOTH permutations
// land on the n index -- the 8x8 block permute runs along the (transposed) n axis, not
// along k -- and the k index is a plain transpose:
//
//   out[k][n] <- s[n_src(n)][k]
//   n_src(n) = (n>>6)<<6 | (n&1)*16 | ((n>>1)&1)*8 | ((n>>2)&1)*32 | ((n>>3)&7)
//
// (The 0,2,1,3 sub-block permute is folded into the same expression: it and the 8x8
// block permute both act on the low bits of n, which is why a single map covers both.)

#include <cstdint>

extern "C" __global__ void marlin_prep_scales(
    const uint8_t* __restrict__ s,   // [N, k32]
    uint8_t* __restrict__ o,         // [k32, N]
    int N, int k32) {
    const long long total = (long long)N * k32;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        const int k = (int)(i / N);
        const int n = (int)(i % N);
        const int n_src = ((n >> 6) << 6) | (((n >> 3) & 7)) | ((n & 1) * 16)
                        | (((n >> 1) & 1) * 8) | (((n >> 2) & 1) * 32);
        o[i] = s[(long long)n_src * k32 + k];
    }
}
