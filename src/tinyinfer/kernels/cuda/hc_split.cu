// hc_split: the Hyper-Connections pre/post/comb tail in one launch.
//
// _native_sinkhorn builds its three inputs with torch:
//     pre  = sigmoid(mixes[..., :4] * scale0 + base[:4]) + eps
//     post = 2 * sigmoid(mixes[..., 4:8] * scale1 + base[4:8])
//     comb = (mixes[..., 8:].view(b,s,4,4) * scale2 + base[8:])
// which is 8 kernels per call, twice per layer, on a 24-float row.
//
// The subtlety that makes this fuse-able at all: torch rounds each of those steps
// separately, while an nvcc default build contracts `m*s + b` into one FMA (single
// rounding). A first version let it, measured 1.2e-7 against torch, and that was enough to
// change the decoded tokens after 40 layers. Every arithmetic step below therefore uses an
// explicitly rounding intrinsic (__fmul_rn / __fadd_rn) so the result is bit-identical to
// the unfused spelling. Verified: maxdiff 0.0 for all three outputs.

#include <cstdint>

#define HC 4
// pre(HC) + post(HC) + comb(HC*HC); the torch code slices mixes[..., :hc], [hc:2hc] and
// [2hc:], so the leading sections are 2*HC not 3*HC.
#define ROW (2 * HC + HC * HC)   // 4 + 4 + 16 = 24

extern "C" __global__ void hc_split(
    const float* __restrict__ mixes,   // [n, ROW]
    const float* __restrict__ scale,   // [3]
    const float* __restrict__ base,    // [ROW]
    float* __restrict__ pre,           // [n, HC]
    float* __restrict__ post,          // [n, HC]
    float* __restrict__ comb,          // [n, HC*HC]
    int n, float eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* m = mixes + (long long)i * ROW;
    float* pp = pre + (long long)i * HC;
    float* po = post + (long long)i * HC;
    float* pc = comb + (long long)i * (HC * HC);
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
        // mul, add, sigmoid, add -- four roundings, same as torch
        const float t = __fmul_rn(m[j], s0);
        const float x = __fadd_rn(t, base[j]);
        pp[j] = __fadd_rn(1.f / (1.f + expf(-x)), eps);
    }
#pragma unroll
    for (int j = 0; j < HC; ++j) {
        const float t = __fmul_rn(m[HC + j], s1);
        const float x = __fadd_rn(t, base[HC + j]);
        po[j] = __fmul_rn(2.f, 1.f / (1.f + expf(-x)));
    }
#pragma unroll
    for (int j = 0; j < HC * HC; ++j) {
        const float t = __fmul_rn(m[2 * HC + j], s2);
        pc[j] = __fadd_rn(t, base[2 * HC + j]);
    }
}
