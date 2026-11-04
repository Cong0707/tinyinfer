// Ported from vLLM's csrc/layernorm_kernels.cu (BSD-3-Clause).
// Upstream: rms_norm_kernel<scalar_t, VEC_SIZE, NUM_DIMS, HasWeight>.
//
// Why port it: upstream reads each row through 16-byte vector loads and sizes the
// block from hidden_size, where the original tinyinfer kernel pinned 32 threads to a
// row and did scalar loads. Measured on this card: 84.3us -> 11.5us at M=1, K=4096.
//
// What changed from upstream: no torch host wrapper. A bare __global__ over raw
// pointers, launched from native.py through the CUDA driver API like every other
// kernel in this directory. Device code keeps upstream's vector width rule and
// per-row blocking.
//
// Launch geometry is decided on the host side (vllm_ops.py) and mirrors upstream:
//   vec_size   = gcd(16 / sizeof(bf16), hidden_size)          -> 8 for DeepSeek
//   block_size = min(hidden_size / vec_size,
//                    num_tokens < 256 ? 1024 : 256)
//   grid       = num_tokens

#include <cuda_bf16.h>

namespace {

constexpr int kVec = 8;  // 16 bytes of bf16

struct alignas(16) VecBF16 {
  __nv_bfloat16 val[kVec];
};

// cub::BlockReduce needs CUDA toolkit headers, which this unit avoids so the cubin
// stays portable. Warp shuffle plus a shared broadcast sums the same terms in a
// different order; the fp32 difference is far below the bf16 output step.
__device__ __forceinline__ float block_reduce_sum(float v, float* s) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  if (lane == 0) s[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = (threadIdx.x < nwarps) ? s[threadIdx.x] : 0.f;
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if (threadIdx.x == 0) s[0] = t;
  }
  __syncthreads();
  return s[0];
}

}  // namespace

// One block per row: mirrors upstream rms_norm_kernel<T, VEC, 2, true>. DeepSeek
// always supplies a 1-D weight and a [tokens, hidden] view, so upstream's 3-D/4-D
// and no-weight branches are not ported yet.
extern "C" __global__ void vllm_rms_norm_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int hidden_size,
    float epsilon) {
  __shared__ float s_variance;
  __shared__ float s_warp[32];

  const __nv_bfloat16* row = input + (long long)blockIdx.x * hidden_size;
  float variance = 0.f;

  const int nvec = hidden_size / kVec;
  const VecBF16* v_in = reinterpret_cast<const VecBF16*>(row);
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const VecBF16 t = v_in[i];
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float x = __bfloat162float(t.val[j]);
      variance += x * x;
    }
  }
  for (int i = nvec * kVec + threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float x = __bfloat162float(row[i]);
    variance += x * x;
  }

  const float total = block_reduce_sum(variance, s_warp);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(total / hidden_size + epsilon);
  }
  __syncthreads();

  __nv_bfloat16* out_row = out + (long long)blockIdx.x * hidden_size;
  const VecBF16* v_w = reinterpret_cast<const VecBF16*>(weight);
  VecBF16* v_out = reinterpret_cast<VecBF16*>(out_row);
  const float inv = s_variance;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const VecBF16 src = v_in[i];
    const VecBF16 w = v_w[i];
    VecBF16 dst;
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float x = __bfloat162float(src.val[j]);
      const float wf = __bfloat162float(w.val[j]);
      dst.val[j] = __float2bfloat16(x * inv * wf);
    }
    v_out[i] = dst;
  }
  for (int i = nvec * kVec + threadIdx.x; i < hidden_size; i += blockDim.x) {
    out_row[i] = __float2bfloat16(__bfloat162float(row[i]) * inv *
                                  __bfloat162float(weight[i]));
  }
}
