# Kernel baseline: what vLLM has, what we have

Every kernel in this project is measured against vLLM before it is optimised. vLLM's CUDA
sources are checked out at `repositories/vllm-csrc/` (274 files, pinned to the commit that
`repositories/vllm` is at) so the comparison is against real code, not a description.

## Our kernels and their vLLM counterpart

| tinyinfer | vLLM equivalent | status |
| --- | --- | --- |
| `gemv_bf16` | cutlass `scaled_mm` (no GEMV path) | ours only, 92% of peak |
| `gemv_fp4_e8m0` | `quantization/marlin` (W4A16) | **vLLM 2.21x faster**, measured |
| `head_gemv` | plain GEMM | ours only |
| `moe_gemv6` | `moe/moe_wna16.cu` | not compared yet |
| `moe_w13` | `moe/marlin_moe_wna16/ops.cu` | not compared yet |
| `moe_act` | `activation_kernels.cu` | not compared yet |
| `rms_norm` | `layernorm_kernels.cu` | not compared yet |
| `gdn` | `gdn/fused_gdn_decode_kernel.cu` | not compared yet |
| `qwen_norm`, `qwen_fused` | (Qwen-specific, no direct match) | ours only |
| `hc` | hyper-connections (V4-specific) | ours only |
| `sinkhorn` | none | ours only |

## vLLM has, we have nothing

These are whole optimisation strategies, not single kernels:

- `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` (67 KB) -- q_norm + RoPE + KV insert in
  one launch; we do three separate kernels per layer, 40 layers.
- `cache_kernels_fused.cu` -- fused KV cache write.
- `moe/moe_align_sum_kernels.cu` (49 KB) + `moe/moe_permute_unpermute_op.cu` -- the
  align/permute/unpermute pipeline that makes grouped MoE GEMM possible.
- `moe/grouped_topk_kernels.cu` (59 KB) -- grouped top-k for MoE routing.
- `quantization/fp4/mxfp4_blockwise_moe_kernel.cu` -- true blockwise fp4 MoE.
- `quantization/cutlass_w4a8/*` -- W4A8 grouped GEMM.
- `sampler.cu` (41 KB), `activation_kernels.cu` (64 KB).
- `custom_all_reduce.cu` -- multi-card communication.

## The one measured number so far

fp4 expert GEMV, RTX 5060 Laptop, sm_120, N=16384 K=5120, 40 MB of weights:

| | time | bandwidth |
| --- | --- | --- |
| vLLM Marlin | 232-260 us | 151-169 GB/s |
| ours (hand-written, best of five attempts) | 657 us | 59 GB/s |

Ratio: **vLLM is 2.21x faster.** The difference is the four-stage `cp.async` pipeline,
`ldmatrix` fragment loads, and the k-major interleaved weight repack.

## Scale

| | .cu files | lines of CUDA |
| --- | --- | --- |
| vLLM (pinned commit, non-ROCm) | 90 | 43,404 |
| tinyinfer | 12 | 1,969 |

vLLM carries 22x the CUDA code. That is not a target to match line-for-line -- it is the size
of the surface we are sampling when we say "copy vLLM".

## How the baseline is measured

vLLM 0.33 ships CUDA sources outside the wheel, so `pip install vllm` gives no kernels.
The baseline is built from source at the pinned commit with `pip install . --no-build-isolation`;
each kernel is then driven directly with real tensors on this GPU and timed with CUDA events
after warm-up. Numbers go in this table with the shape and the card, because a bandwidth
figure without the shape means nothing.