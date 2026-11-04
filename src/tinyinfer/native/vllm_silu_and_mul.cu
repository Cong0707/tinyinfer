// Ported from vLLM's csrc/activation_kernels.cu (BSD-3-Clause).
// Upstream: act_and_mul_kernel<scalar_t, packed_t, ACT_FN, PACKED_ACT_FN,
//          act_first, use_vec, HAS_CLAMP>.
//
// This is the DeepSeek SwiGLU shape: act_first=true, HAS_CLAMP=true, alpha=1,
// beta=0, which upstream reaches through silu_and_mul_with_clamp(). It clamps the
// gate half from above (fminf) and the up half from both sides before SiLU, which is
// exactly what this model's swiglu_limit asks for.
//
// What changed: torch host wrapper removed, dtype dispatch specialised to bf16, and
// 128-bit loads written out instead of going through PackedTypeConverter/PackedVec.
// The arithmetic is upstream's, term for term, so the result is bit-comparable.
//
//   out[t, i] = silu(min(gate[t,i], limit)) * clamp(up[t,i], -limit, limit)
//
// grid = (num_tokens), block = min(d / VEC, 1024) where d = input.size(-1) / 2.

#include <cuda_bf16.h>

namespace {

constexpr int kVec = 8;  // 16 bytes of bf16

struct alignas(16) VecBF16 {
  __nv_bfloat16 val[kVec];
};

__device__ __forceinline__ float silu_f(float x) {
  // upstream silu_kernel: x / (1 + exp(-x * alpha)) with alpha defaulting to 1
  return x / (1.0f + expf(-x));
}

}  // namespace

extern "C" __global__ void vllm_silu_and_mul_clamp_bf16(
    __nv_bfloat16* __restrict__ out,          // [..., d]
    const __nv_bfloat16* __restrict__ input,  // [..., 2, d]
    const int d, const float limit) {
  const long long token_idx = blockIdx.x;
  const __nv_bfloat16* x_ptr = input + token_idx * 2LL * d;  // gate half
  const __nv_bfloat16* y_ptr = x_ptr + d;                    // up half
  __nv_bfloat16* out_ptr = out + token_idx * (long long)d;

  const int num_vecs = d / kVec;
  const VecBF16* x_vec = reinterpret_cast<const VecBF16*>(x_ptr);
  const VecBF16* y_vec = reinterpret_cast<const VecBF16*>(y_ptr);
  VecBF16* out_vec = reinterpret_cast<VecBF16*>(out_ptr);

  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    VecBF16 x = x_vec[i];
    VecBF16 y = y_vec[i];
    VecBF16 r;
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      // act_first + HAS_CLAMP, upstream compute() branch 1
      float gate = fminf(__bfloat162float(x.val[j]), limit);
      float up = fmaxf(fminf(__bfloat162float(y.val[j]), limit), -limit);
      r.val[j] = __float2bfloat16(silu_f(gate) * up);
    }
    out_vec[i] = r;
  }

  // scalar tail when d is not a multiple of the vector width
  for (int idx = num_vecs * kVec + threadIdx.x; idx < d; idx += blockDim.x) {
    float gate = fminf(__bfloat162float(x_ptr[idx]), limit);
    float up = fmaxf(fminf(__bfloat162float(y_ptr[idx]), limit), -limit);
    out_ptr[idx] = __float2bfloat16(silu_f(gate) * up);
  }
}
