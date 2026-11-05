// gemv_fp4_w4a8_gate_up: gate and up in ONE launch.
//
// The MoE expert computes silu(gate(x)) * up(x), where gate and up are two separate
// [N, K] fp4 matrices over the SAME activation. Today that is two GEMV launches, and a
// torch.cat of the two weights to fuse them was tried and removed (the cat cost 768
// launches and 22ms of GPU time per token -- more than the 18 -> 12 launch saving).
//
// This kernel needs no cat: it takes both weight pointers and both scale pointers, and
// walks them together. The activation, which is the only operand they share, is read
// once instead of twice.
//
// What it does NOT fuse: the silu and the multiply. Those need the full output of both
// GEMVs, so fusing them would mean holding 2N accumulators in the same block and
// re-reading the activation -- the activation is the small operand here, the weights
// are not, and this kernel is weight-bandwidth-bound.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

// NOT __constant__: a cubin loaded through cuModuleLoadData does not carry the host-side
// initialiser for constant memory, so the array reads as garbage in the kernel (measured:
// every output came out ~1e36, i.e. the fp32 bits reinterpreted). A plain file-scope
// __device__ array is emitted as initialised device data and works under dynamic loading.
__device__ float kFp4Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

constexpr int ROWS = 8;

// one row of the same maths as gemv_fp4_w4a8, over two weight matrices
__device__ __forceinline__ float row_dot(
    const __nv_fp8_e4m3* __restrict__ arow, const float* __restrict__ asrow,
    const unsigned char* __restrict__ wrow, const float* __restrict__ wsrow,
    int nb, int lane) {
    float total = 0.f;
    for (int kb = 0; kb < nb; ++kb) {
        const int kidx = (kb << 5) + lane;
        const float a = float(arow[kidx]);
        const unsigned char byte = wrow[kidx >> 1];
        const int nib = (kidx & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
        float part = a * kFp4Lut[nib];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
        total += part * asrow[kb] * wsrow[kb];
    }
    return total;
}

}  // namespace

// G, U  [M, N]  bf16 outputs
extern "C" __global__ void gemv_fp4_w4a8_gate_up(
    const __nv_fp8_e4m3* __restrict__ A,
    const float* __restrict__ AS,
    const unsigned char* __restrict__ WG,
    const float* __restrict__ WSG,
    const unsigned char* __restrict__ WU,
    const float* __restrict__ WSU,
    __nv_bfloat16* __restrict__ G,
    __nv_bfloat16* __restrict__ U,
    int M, int N, int K) {
    const int nb = K >> 5;
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;
    const int nblocks = (N + ROWS - 1) / ROWS;
    const int bx = blockIdx.x % nblocks;
    const int bm = blockIdx.x / nblocks;
    if (bm >= M) return;

    const int n = bx * ROWS + r;
    if (n >= N) return;

    const __nv_fp8_e4m3* arow = A + (long long)bm * K;
    const float* asrow = AS + (long long)bm * nb;
    const long long woff = (long long)n * (K >> 1);
    const long long soff = (long long)n * nb;

    const float g = row_dot(arow, asrow, WG + woff, WSG + soff, nb, lane);
    const float u = row_dot(arow, asrow, WU + woff, WSU + soff, nb, lane);
    if (lane == 0) {
        G[(long long)bm * N + n] = __float2bfloat16(g);
        U[(long long)bm * N + n] = __float2bfloat16(u);
    }
}
