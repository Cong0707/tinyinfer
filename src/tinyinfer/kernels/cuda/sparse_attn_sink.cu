// Sparse (gathered) attention with a learned per-head sink.
//
//   out[q,h,:] = [ sum_i exp(s_i) * v[slot_i] ] / [ sum_i exp(s_i) + exp(sink_h) ]
//   s_i = q[q,h,:] . kv[slot_i] * scale
//
// The sink is a virtual slot: a learned logit with no value. It gives a head a way to
// attend to nothing, which stops the first positions of a sequence being forced onto an
// arbitrary token. The reference (nn.sparse_attn_ref) appends it as one extra score whose
// value vector is zero, so it only ever reaches the denominator.
//
// PARALLEL STRUCTURE
//
//   One CTA per (row, head). The warps split the SLOT list (K is 640 in this model: 128
//   window + 512 index-topk), and inside a warp the 32 lanes split head_dim. Each warp
//   produces a partial (max, sum, weighted-value) triple over its own slots and the CTA
//   merges them with the standard split-KV combine:
//
//     mx    = max(sink, m_0..m_w)
//     denom = exp(sink - mx) + sum_w l_w * exp(m_w - mx)
//     out   = sum_w acc_w * exp(m_w - mx) / denom
//
//   That is the identity vLLM uses in merge_attn_states.cu (arXiv 2501.01005 sec 2.2).
//   Two earlier versions of this kernel got it wrong and are worth recording:
//
//     - the merge was skipped entirely, so only one warp value sum survived;
//     - the sink raised the max but the accumulated values were never rescaled by the
//       same factor, which is wrong exactly when the sink is the largest logit -- i.e.
//       on real weights, where a learned sink is usually large.
//
//   A single-warp version was correct but slow: one warp walking 640 slots serially is
//   latency-bound (982 us at K=640 vs 202 us at K=128). Splitting K across 4 warps is
//   what makes the K=640 case competitive.
//
// No tensor cores: at decode there is one query row, so the work is a 1 x K dot per head
// and the kernel is bound by gather latency, not FLOPs.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// -FLT_MAX breaks exp(): exp(-FLT_MAX - (-FLT_MAX)) is NaN. A large finite negative
// underflows the same way inside the merge and never produces NaN.
__device__ constexpr float kNegBig = -1.0e30f;

}  // namespace

// q    [B, M, H, D] bf16
// kv   [B, N, D]    bf16
// sink [H]          fp32
// idxs [B, M, K]    int32  (-1 = empty slot)
// out  [B, M, H, D] bf16
extern "C" __global__ void sparse_attn_sink(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kv,
    const float* __restrict__ sink,
    const int* __restrict__ idxs,
    __nv_bfloat16* __restrict__ out,
    int B, int M, int H, int D, int N, int K, float scale) {
    const int nhead = B * M * H;
    const int wid = blockIdx.x;
    if (wid >= nhead) return;

    const int head = wid % H;
    const int row = (wid / H) % M;
    const int batch = wid / (H * M);

    const int nwarps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    extern __shared__ float sm[];
    float* s_m = sm;
    float* s_l = sm + nwarps;
    float* s_out = sm + 2 * nwarps;

    const __nv_bfloat16* qr = q + (((long long)batch * M + row) * H + head) * D;
    const int* ir = idxs + ((long long)batch * M + row) * K;

    constexpr int kMaxLocal = 16;
    float acc[kMaxLocal];
#pragma unroll
    for (int i = 0; i < kMaxLocal; ++i) acc[i] = 0.f;
    const int nlocal = (D + 31) / 32;
    if (nlocal > kMaxLocal) return;

    float m_i = kNegBig;
    float l_i = 0.f;
    for (int t = warp; t < K; t += nwarps) {
        const int slot = __ldg(ir + t);
        if (slot < 0 || slot >= N) continue;
        const __nv_bfloat16* kr = kv + ((long long)batch * N + slot) * D;

        float dot = 0.f;
        for (int i = 0; i < nlocal; ++i) {
            const int d = lane + i * 32;
            if (d < D) dot += __bfloat162float(qr[d]) * __bfloat162float(kr[d]);
        }
        dot = warp_allreduce_sum(dot);
        const float sc = dot * scale;

        const float m_new = fmaxf(m_i, sc);
        const float e = __expf(sc - m_new);
        const float rs = __expf(m_i - m_new);
        l_i = l_i * rs + e;
        m_i = m_new;

        for (int i = 0; i < nlocal; ++i) {
            const int d = lane + i * 32;
            if (d < D) acc[i] = acc[i] * rs + e * __bfloat162float(kr[d]);
        }
    }

    if (lane == 0) {
        s_m[warp] = m_i;
        s_l[warp] = l_i;
    }
    for (int i = 0; i < nlocal; ++i) {
        const int d = lane + i * 32;
        if (d < D) s_out[warp * D + d] = acc[i];
    }
    __syncthreads();

    const float sk = sink[head];
    float mx = sk;
    for (int w = 0; w < nwarps; ++w) mx = fmaxf(mx, s_m[w]);

    float denom = __expf(sk - mx);
    for (int w = 0; w < nwarps; ++w) denom += s_l[w] * __expf(s_m[w] - mx);
    const float inv = (denom > 0.f) ? (1.f / denom) : 0.f;

    __nv_bfloat16* orow =
        out + (((long long)batch * M + row) * H + head) * D;
    for (int i = 0; i < nlocal; ++i) {
        const int d = lane + i * 32;
        if (d < D) {
            float v = 0.f;
            for (int w = 0; w < nwarps; ++w) v += s_out[w * D + d] * __expf(s_m[w] - mx);
            orow[d] = __float2bfloat16(v * inv);
        }
    }
}
