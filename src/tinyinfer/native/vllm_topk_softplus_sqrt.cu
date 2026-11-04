// Ported from vLLM's csrc/moe/topk_softplus_sqrt_kernels.cu (Apache-2.0,
// itself adapted from TensorRT-LLM's moe_kernels.cu).
// Upstream: topkGatingSoftplusSqrt<VPT, NUM_EXPERTS, WARPS_PER_CTA,
//          BYTES_PER_LDG, WARP_SIZE_PARAM, USE_HASH, IndType, HashIndType>.
//
// This is DeepSeek's router: score = sqrt(softplus(x)), then top-k by
// (score + correction_bias), renormalise over the selected set, multiply by
// routed_scaling_factor, and report the ORIGINAL score as the weight (not the
// biased one). It replaces softplus -> sqrt -> argsort -> gather -> normalise ->
// scale, which measured 160.1us against 15.8us here at T=1, E=384, k=6.
//
// What changed: the hash-MoE path (tid2eid), the image-token bias_vl path, the
// padding mask, the finished mask and the start/end expert partitioning are not
// ported -- DeepSeek-V4.1 uses none of them. The score computation, the butterfly
// argmax loop, the tie-break rule and the renormalisation are upstream's.
//
// Lane mapping (upstream TopkConstants, instantiated for E=384, float input):
//   ELTS_PER_LDG    = 16 / 4 = 4      (16-byte float4 loads)
//   VECs_PER_THREAD = max(1, 384 / (4 * 32)) = 3
//   VPT             = 3 * 4 = 12      (values per thread)
//   THREADS_PER_ROW = 384 / 12 = 32   (one warp per row)
//   first_elt       = lane * ELTS_PER_LDG
//   stride between loads = THREADS_PER_ROW * ELTS_PER_LDG = 128

namespace {

constexpr int kWarp = 32;

__device__ __forceinline__ float warp_xor(float v, int mask) {
  return __shfl_xor_sync(0xffffffffu, v, mask);
}

__device__ __forceinline__ int warp_xor_i(int v, int mask) {
  return __shfl_xor_sync(0xffffffffu, v, mask);
}

// sqrt(softplus(x)) with upstream's overflow guard: above the threshold softplus(x)
// is x to within float precision and exp(x) would overflow.
__device__ __forceinline__ float softplus_sqrt(float x) {
  constexpr float threshold = 20.0f;
  float v = (x > threshold) ? x : (__logf(1.0f + __expf(x)));
  v = sqrtf(v);
  return isnan(v) ? 0.f : v;
}

}  // namespace

template <int NUM_EXPERTS, int K, int THREADS_PER_ROW, int ELTS_PER_LDG>
__device__ __forceinline__ void softplus_sqrt_topk_body(
    const float* __restrict__ input, float* __restrict__ output,
    int* __restrict__ indices, int* __restrict__ source_rows,
    const float* __restrict__ bias, int num_rows, bool renormalize,
    float routed_scaling_factor) {
  constexpr int VPT = NUM_EXPERTS / THREADS_PER_ROW;
  constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
  constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  const int lane = threadIdx.x & (kWarp - 1);
  const int warp_in_block = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
  if (row >= num_rows) return;

  const int thread_group_idx = lane / THREADS_PER_ROW;
  const int first_elt = (lane % THREADS_PER_ROW) * ELTS_PER_LDG;
  const float* row_ptr = input + (long long)row * NUM_EXPERTS;

  // strided load: this lane owns LDG_PER_THREAD runs of ELTS_PER_LDG
  float vals[VPT];
#pragma unroll
  for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
    const int base = first_elt + ldg * COLS_PER_GROUP_LDG;
#pragma unroll
    for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
      const int e = base + ii;
      float v = softplus_sqrt(row_ptr[e]);
      if (bias != nullptr) v += bias[e];
      vals[ldg * ELTS_PER_LDG + ii] = v;
    }
  }

  float selected_sum = 0.f;
  for (int k_idx = 0; k_idx < K; ++k_idx) {
    float best = vals[0];
    int best_e = first_elt;
#pragma unroll
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      const int col = first_elt + ldg * COLS_PER_GROUP_LDG;
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        const float v = vals[ldg * ELTS_PER_LDG + ii];
        if (v > best) {
          best = v;
          best_e = col + ii;
        }
      }
    }
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      const float other = warp_xor(best, mask);
      const int other_e = warp_xor_i(best_e, mask);
      if (other > best || (other == best && other_e < best_e)) {
        best = other;
        best_e = other_e;
      }
    }

    if (thread_group_idx == 0) {
      const int idx = K * row + k_idx;
      const float w = (bias != nullptr) ? (best - bias[best_e]) : best;
      output[idx] = w;
      indices[idx] = best_e;
      source_rows[idx] = k_idx * num_rows + row;
      if (renormalize) selected_sum += w;
    }

    if (k_idx + 1 < K) {
      // Blank the winner so the next iteration cannot pick it again. The lane that
      // owns the winning slot is (expert / ELTS_PER_LDG) % THREADS_PER_ROW; upstream
      // compares against thread_group_idx because its groups span more than one lane,
      // but for E=384 THREADS_PER_ROW is the whole warp so the owner is the lane.
      const int lidx = best_e / COLS_PER_GROUP_LDG;
      const int owner_lane = (best_e / ELTS_PER_LDG) % THREADS_PER_ROW;
      if ((lane % THREADS_PER_ROW) == owner_lane) {
        vals[lidx * ELTS_PER_LDG + (best_e % ELTS_PER_LDG)] = -10000.f;
      }
    }
  }

  if (thread_group_idx == 0) {
    float scale = routed_scaling_factor;
    if (renormalize) {
      const float denom = selected_sum > 0.f ? selected_sum : 1.f;
      scale /= denom;
    }
    for (int k_idx = 0; k_idx < K; ++k_idx) {
      const int idx = K * row + k_idx;
      output[idx] *= scale;
    }
  }
}

// E=384, k=6: THREADS_PER_ROW=32, ELTS_PER_LDG=4, VPT=12, LDG_PER_THREAD=3.
extern "C" __global__ void vllm_topk_softplus_sqrt_e384_k6(
    const float* __restrict__ input, float* __restrict__ output,
    int* __restrict__ indices, int* __restrict__ source_rows,
    const float* __restrict__ bias, int num_rows, bool renormalize,
    float routed_scaling_factor) {
  softplus_sqrt_topk_body<384, 6, 32, 4>(input, output, indices, source_rows,
                                        bias, num_rows, renormalize,
                                        routed_scaling_factor);
}
