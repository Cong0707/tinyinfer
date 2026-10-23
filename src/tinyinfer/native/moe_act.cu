// Hand-written decode kernels (no torch headers: nvcc -> cubin, loaded by
// tinyinfer/native.py through the CUDA driver API).
//
// moe_act: the per-expert SwiGLU chain. torch runs clamp(gate)/clamp(up)/silu/mul/scale
// as five kernels per expert -- six experts per layer, 40 layers, ~1200 launches per
// decode step just here. One paged kernel over the rows.

#include <cuda_bf16.h>

extern "C" __global__ void moe_act_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int n, float limit, float scale, int has_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    if (limit > 0.f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    // silu(g) = g / (1 + exp(-g))
    const float s = g / (1.f + __expf(-g));
    float v = s * u;
    if (has_scale) v *= scale;
    out[i] = __float2bfloat16(v);
}
