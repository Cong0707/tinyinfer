// swiglu_scale: the MoE activation tail in one launch.
//
// The decode path used to spell this out in torch:
//     gate = lin(...).float()        bf16 -> fp32 copy
//     up   = lin(...).float()        bf16 -> fp32 copy
//     up   = clamp(up, -lim, lim)    elementwise
//     gate = clamp(gate, max=lim)    elementwise
//     y    = silu(gate) * up         two kernels
//     y    = weights * y             elementwise
//     y    = y.to(bf16)              fp32 -> bf16 copy
// which is 7 launches per layer, 280 per token. On Windows a launch costs ~15 us of
// wall time regardless of size, so this is a launch-count problem, not a bandwidth one:
// the tensors are [1, 2304].
//
// One thread per output element, fp32 maths inside, bf16 in and out.

#include <cuda_bf16.h>
#include <cstdint>

extern "C" __global__ void swiglu_scale(
    const __nv_bfloat16* __restrict__ g,   // gate  [n]
    const __nv_bfloat16* __restrict__ u,   // up    [n]
    const float* __restrict__ w,           // routing weight, 1 element, may be NULL
    __nv_bfloat16* __restrict__ out,       // [n]
    int n, float lim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gf = float(g[i]);
    float uf = float(u[i]);
    if (lim > 0.f) {
        gf = fminf(gf, lim);
        uf = fmaxf(-lim, fminf(uf, lim));
    }
    // silu(gate) * up, then the routing weight
    float y = gf / (1.f + __expf(-gf)) * uf;
    if (w != nullptr) y *= w[0];
    out[i] = __float2bfloat16(y);
}
