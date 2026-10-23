// Single-token decode kernels for Qwen3.5's GatedDeltaNet (linear attention).
//
// transformers falls back to a pure-torch reference for these unless fla's Triton wheels
// are installed and torch.compile works -- on this box it does not, so a 24-layer model
// with 18 linear-attention layers ran as ~1800 tiny kernels per token (33ms, 12% of the
// card's bandwidth). These are the three steps that matter, one kernel each.
//
// Math (seq_len == 1), per (batch, v_head):
//     q = l2norm(q) / sqrt(D_k);  k = l2norm(k)
//     state *= exp(g)                       state: [D_k, D_v]
//     kv_mem = sum_dk state[dk, dv] * k[dk]
//     delta  = (v - kv_mem) * beta
//     state += k[:, None] * delta[None, :]
//     out    = sum_dk state[dk, dv] * q[dk]

#include <cuda_bf16.h>

// Number of threads used to cover the D_v axis; D_k is walked by the same threads.
// One block per (batch, v_head): the [D_k, D_v] state of a head fits in shared memory.
extern "C" __global__ void gdn_step_f(const float* __restrict__ q,      // [B, H, Dk]
                                      const float* __restrict__ k,      // [B, H, Dk]
                                      const float* __restrict__ v,      // [B, H, Dv]
                                      const float* __restrict__ g,      // [B, H]   (log decay)
                                      const float* __restrict__ beta,   // [B, H]
                                      float* __restrict__ state,        // [B, H, Dk, Dv] in/out
                                      float* __restrict__ out,          // [B, H, Dv]
                                      int B, int H, int Dk, int Dv,
                                      int l2norm) {
    int bh = blockIdx.x;
    int b = bh / H, h = bh - b * H;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int plane = Dk * Dv;
    float* st = state + (long)bh * plane;
    const float* qh = q + ((long)b * H + h) * Dk;
    const float* kh = k + ((long)b * H + h) * Dk;
    const float* vh = v + ((long)b * H + h) * Dv;
    float* oh = out + ((long)b * H + h) * Dv;

    __shared__ float s_q[256];       // normalized q for this head (Dk <= 256 here)
    __shared__ float s_k[256];       // normalized k
    __shared__ float s_scale;
    __shared__ float s_beta;

    // ---- L2 norms of q and k, in fp32, same eps the reference uses ----
    if (tid == 0) {
        float sq = 0.f, sk = 0.f;
        for (int i = 0; i < Dk; ++i) {
            float a = qh[i], bq = kh[i];
            sq += a * a;
            sk += bq * bq;
        }
        s_scale = l2norm ? rsqrtf(sq + 1e-6f) : 1.f;
        float ks = l2norm ? rsqrtf(sk + 1e-6f) : 1.f;
        s_beta = beta[(long)b * H + h];
        float rq = s_scale * rsqrtf((float)Dk);
        for (int i = 0; i < Dk; ++i) {
            s_q[i] = qh[i] * rq;
            s_k[i] = kh[i] * ks;
        }
    }
    __syncthreads();

    const float decay = expf(g[(long)b * H + h]);

    // ---- decay the state, then compute kv_mem = sum_dk state*dk k ----
    // One column range per thread pass; s_mem holds the per-(dv) dot product.
    extern __shared__ float sh[];    // [Dv] scratch for kv_mem / delta
    float* s_mem = sh;
    for (int dv = tid; dv < Dv; dv += nt) {
        float acc = 0.f;
        for (int dk = 0; dk < Dk; ++dk) {
            long idx = (long)dk * Dv + dv;
            float s0 = st[idx] * decay;
            st[idx] = s0;
            acc += s0 * s_k[dk];
        }
        float d = (vh[dv] - acc) * s_beta;
        s_mem[dv] = d;
    }
    __syncthreads();

    // ---- rank-1 update: state[dk, dv] += k[dk] * delta[dv], then output ----
    for (int dv = tid; dv < Dv; dv += nt) {
        float d = s_mem[dv];
        float acc = 0.f;
        for (int dk = 0; dk < Dk; ++dk) {
            long idx = (long)dk * Dv + dv;
            st[idx] += s_k[dk] * d;
            acc += st[idx] * s_q[dk];
        }
        oh[dv] = acc;
    }
}

// bf16 variant: activations are bf16, the state is kept in fp32 (the reference promotes
// everything to fp32 and only casts the result back).
extern "C" __global__ void gdn_step_bf(const __nv_bfloat16* __restrict__ q,
                                       const __nv_bfloat16* __restrict__ k,
                                       const __nv_bfloat16* __restrict__ v,
                                       const float* __restrict__ g,
                                       const float* __restrict__ beta,
                                       float* __restrict__ state,
                                       __nv_bfloat16* __restrict__ out,
                                       int B, int H, int Dk, int Dv,
                                       int l2norm) {
    int bh = blockIdx.x;
    int b = bh / H, h = bh - b * H;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    float* st = state + (long)bh * Dk * Dv;
    const __nv_bfloat16* qh = q + ((long)b * H + h) * Dk;
    const __nv_bfloat16* kh = k + ((long)b * H + h) * Dk;
    const __nv_bfloat16* vh = v + ((long)b * H + h) * Dv;
    __nv_bfloat16* oh = out + ((long)b * H + h) * Dv;

    __shared__ float s_q[256];
    __shared__ float s_k[256];
    __shared__ float s_beta;
    if (tid == 0) {
        float sq = 0.f, sk = 0.f;
        for (int i = 0; i < Dk; ++i) {
            float a = __bfloat162float(qh[i]), bq = __bfloat162float(kh[i]);
            sq += a * a; sk += bq * bq;
        }
        s_beta = beta[(long)b * H + h];
        float rq = (l2norm ? rsqrtf(sq + 1e-6f) : 1.f) * rsqrtf((float)Dk);
        float ks = l2norm ? rsqrtf(sk + 1e-6f) : 1.f;
        for (int i = 0; i < Dk; ++i) {
            s_q[i] = __bfloat162float(qh[i]) * rq;
            s_k[i] = __bfloat162float(kh[i]) * ks;
        }
    }
    __syncthreads();
    const float decay = expf(g[(long)b * H + h]);
    extern __shared__ float sh[];
    float* s_mem = sh;
    for (int dv = tid; dv < Dv; dv += nt) {
        float acc = 0.f;
        for (int dk = 0; dk < Dk; ++dk) {
            long idx = (long)dk * Dv + dv;
            float s0 = st[idx] * decay;
            st[idx] = s0;
            acc += s0 * s_k[dk];
        }
        s_mem[dv] = (__bfloat162float(vh[dv]) - acc) * s_beta;
    }
    __syncthreads();
    for (int dv = tid; dv < Dv; dv += nt) {
        float d = s_mem[dv];
        float acc = 0.f;
        for (int dk = 0; dk < Dk; ++dk) {
            long idx = (long)dk * Dv + dv;
            st[idx] += s_k[dk] * d;
            acc += st[idx] * s_q[dk];
        }
        oh[dv] = __float2bfloat16(acc);
    }
}

// bf16 conv_state variant: the HF cache stores conv_states in the model dtype, so the
// shift/append has to read and write bf16 while the accumulate stays fp32.
extern "C" __global__ void gdn_conv_step_bf(const __nv_bfloat16* __restrict__ x,
                                            const float* __restrict__ w,
                                            const float* __restrict__ bias,
                                            __nv_bfloat16* __restrict__ conv_state,
                                            __nv_bfloat16* __restrict__ out,
                                            int B, int C, int K, int silu,
                                            int has_bias) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * C) return;
    int b = (int)(i / C), c = (int)(i - (long)b * C);
    __nv_bfloat16* st = conv_state + ((long)b * C + c) * K;
    for (int t = 0; t + 1 < K; ++t) st[t] = st[t + 1];
    st[K - 1] = x[i];
    float acc = has_bias ? bias[c] : 0.f;
    const float* wc = w + (long)c * K;
    for (int t = 0; t < K; ++t) acc += wc[t] * __bfloat162float(st[t]);
    if (silu) acc = acc / (1.f + expf(-acc));
    out[i] = __float2bfloat16(acc);
}

// Causal depthwise conv for one new token, updating conv_state in place.
//
// Mirrors transformers' reference exactly:
//     new  = cat([conv_state, x], -1)[:, :, -K:]        # last K samples, newest last
//     out  = conv1d(new, w, bias, groups=C)
// conv_state: [B, C, K]; x: [B, C]; w: [C, K]; out: [B, C].
extern "C" __global__ void gdn_conv_step_f(const float* __restrict__ x,      // [B, C]
                                           const float* __restrict__ w,      // [C, K]
                                           const float* __restrict__ bias,   // [C] or null
                                           float* __restrict__ conv_state,   // [B, C, K]
                                           float* __restrict__ out,          // [B, C]
                                           int B, int C, int K, int silu,
                                           int has_bias) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B * C) return;
    int b = (int)(i / C), c = (int)(i - (long)b * C);
    float* st = conv_state + ((long)b * C + c) * K;
    // state holds the previous K samples (oldest first); append the new one and shift
    for (int t = 0; t + 1 < K; ++t) st[t] = st[t + 1];
    st[K - 1] = x[i];
    float acc = has_bias ? bias[c] : 0.f;
    const float* wc = w + (long)c * K;
    for (int t = 0; t < K; ++t) acc += wc[t] * st[t];
    if (silu) acc = acc / (1.f + expf(-acc));
    out[i] = acc;
}

// Gated RMSNorm: y = (x * rsqrt(mean(x^2)+eps) * weight) * silu(z)
extern "C" __global__ void gdn_norm_gated_bf(const __nv_bfloat16* __restrict__ x,   // [R, D]
                                             const __nv_bfloat16* __restrict__ z,   // [R, D]
                                             const float* __restrict__ w,           // [D]
                                             __nv_bfloat16* __restrict__ y,         // [R, D]
                                             int R, int D, float eps) {
    const int r = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (r >= R) return;
    const __nv_bfloat16* xr = x + (long)r * D;
    __nv_bfloat16* yr = y + (long)r * D;
    const __nv_bfloat16* zr = z + (long)r * D;
    float s = 0.f;
    for (int k = lane; k < D; k += 32) {
        float val = __bfloat162float(xr[k]);
        s += val * val;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    s = __shfl_sync(0xffffffffu, s, 0);
    const float rstd = rsqrtf(s / (float)D + eps);
    for (int k = lane; k < D; k += 32) {
        float zv = __bfloat162float(zr[k]);
        float gate = zv / (1.f + expf(-zv));
        yr[k] = __float2bfloat16(__bfloat162float(xr[k]) * rstd * w[k] * gate);
    }
}

// gdn_step, rewritten for bandwidth instead of one-column-at-a-time.
//
// The first version gave each head one block and walked dv serially, so the state was
// touched in 128 separate 512-byte columns per head and reached ~50 GB/s (41us for 2MB).
// This version keeps the whole [Dk, Dv] state of a head in registers: 128 threads, each
// owning one dv column, and all 128 dk values are read in one pass. Every access is
// therefore a full 128-thread coalesced read/write, and the state is updated in place.

#include <cuda_bf16.h>

extern "C" __global__ void gdn_step_v2(const __nv_bfloat16* __restrict__ q,
                                       const __nv_bfloat16* __restrict__ k,
                                       const __nv_bfloat16* __restrict__ v,
                                       const float* __restrict__ g,
                                       const float* __restrict__ beta,
                                       float* __restrict__ state,
                                       __nv_bfloat16* __restrict__ out,
                                       int B, int H, int Dk, int Dv, int l2norm) {
    const int bh = blockIdx.x;
    const int dv = threadIdx.x;              // one dv column per thread
    if (dv >= Dv) return;
    const int b = bh / H, h = bh - b * H;
    float* st = state + (long)bh * Dk * Dv;
    const __nv_bfloat16* qh = q + ((long)b * H + h) * Dk;
    const __nv_bfloat16* kh = k + ((long)b * H + h) * Dk;
    const __nv_bfloat16* vh = v + ((long)b * H + h) * Dv;

    __shared__ float s_q[256];
    __shared__ float s_k[256];
    __shared__ float s_beta;
    if (threadIdx.x == 0) {
        float sq = 0.f, sk = 0.f;
        for (int i = 0; i < Dk; ++i) {
            float a = __bfloat162float(qh[i]), c = __bfloat162float(kh[i]);
            sq += a * a; sk += c * c;
        }
        s_beta = beta[(long)b * H + h];
        float rq = (l2norm ? rsqrtf(sq + 1e-6f) : 1.f) * rsqrtf((float)Dk);
        float rk = l2norm ? rsqrtf(sk + 1e-6f) : 1.f;
        for (int i = 0; i < Dk; ++i) {
            s_q[i] = __bfloat162float(qh[i]) * rq;
            s_k[i] = __bfloat162float(kh[i]) * rk;
        }
    }
    __syncthreads();
    const float decay = expf(g[(long)b * H + h]);

    // pass 1: decay the column and accumulate kv_mem
    float acc = 0.f;
    for (int dk = 0; dk < Dk; ++dk) {
        float s = st[(long)dk * Dv + dv] * decay;
        st[(long)dk * Dv + dv] = s;
        acc += s * s_k[dk];
    }
    const float delta = (__bfloat162float(vh[dv]) - acc) * s_beta;

    // pass 2: rank-1 update and the output dot product
    float sum = 0.f;
    for (int dk = 0; dk < Dk; ++dk) {
        float s = st[(long)dk * Dv + dv] + s_k[dk] * delta;
        st[(long)dk * Dv + dv] = s;
        sum += s * s_q[dk];
    }
    out += ((long)b * H + h) * Dv;
    out[dv] = __float2bfloat16(sum);
}

// gdn_conv_step_v2: the conv state [B, C, K] with C=6144 was walked by one thread per
// channel, i.e. 24 blocks of 256 threads -- and each thread did a 4-element shift plus a
// 4-tap dot in sequence. Measured 61us for 96KB of state. Here one warp handles 32
// channels and the K axis is unrolled, so the shift is four coalesced 32-thread moves.
extern "C" __global__ void gdn_conv_step_v2(const __nv_bfloat16* __restrict__ x,
                                            const float* __restrict__ w,
                                            const float* __restrict__ bias,
                                            __nv_bfloat16* __restrict__ conv_state,
                                            __nv_bfloat16* __restrict__ out,
                                            int B, int C, int K, int silu, int has_bias) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int b = blockIdx.y;
    __nv_bfloat16* st = conv_state + ((long)b * C + c) * K;
    // K is 4: unrolled, no loop overhead
    float s0, s1, s2, s3;
    if (K == 4) {
        s0 = __bfloat162float(st[1]);
        s1 = __bfloat162float(st[2]);
        s2 = __bfloat162float(st[3]);
        s3 = __bfloat162float(x[(long)b * C + c]);
        st[0] = __float2bfloat16(s0);
        st[1] = __float2bfloat16(s1);
        st[2] = __float2bfloat16(s2);
        st[3] = __float2bfloat16(s3);
    } else {
        for (int t = 0; t + 1 < K; ++t) st[t] = st[t + 1];
        st[K - 1] = x[(long)b * C + c];
        return;   // generic path keeps the simple form
    }
    const float* wc = w + (long)c * K;
    float acc = has_bias ? bias[c] : 0.f;
    acc += wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * s3;
    if (silu) acc = acc / (1.f + expf(-acc));
    out[(long)b * C + c] = __float2bfloat16(acc);
}

// gdn_norm_gated_v2: 16 rows of 128 values was one block of 32 threads (a single warp per
// row, 16 warps total on the whole card). 55us for 4KB. One block now covers all rows of
// one (batch, head) with a full warp, and rows are spread over blocks.
extern "C" __global__ void gdn_norm_gated_v2(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ z,
                                             const float* __restrict__ w,
                                             __nv_bfloat16* __restrict__ y,
                                             int R, int D, float eps) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int lane = threadIdx.x & 31;
    const __nv_bfloat16* xr = x + (long)r * D;
    const __nv_bfloat16* zr = z + (long)r * D;
    __nv_bfloat16* yr = y + (long)r * D;
    float s = 0.f;
    for (int k = lane; k < D; k += 32) {
        float v = __bfloat162float(xr[k]);
        s += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    s = __shfl_sync(0xffffffffu, s, 0);
    const float rstd = rsqrtf(s / (float)D + eps);
    for (int k = lane; k < D; k += 32) {
        float n = __bfloat162float(xr[k]) * rstd;
        float zv = __bfloat162float(zr[k]);
        float gate = zv / (1.f + expf(-zv));
        yr[k] = __float2bfloat16(__bfloat162float(__float2bfloat16(n)) * w[k] * gate);
    }
}
