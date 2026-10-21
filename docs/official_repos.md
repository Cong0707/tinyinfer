# What we take from the official DeepSeek repositories

Cloned under `repositories/`: DeepInfer, DeepEP, DeepGEMM, deepseek-recipe, DeepJIT,
plus the external references (ds4, flash-moe, llama.cpp, Strata). This note records
what each one offers and how it maps onto tinyinfer, so the porting work has a
written target instead of being rediscovered.

## DeepGEMM — the kernel library (highest value)

Unified tensor-core kernels: FP8/FP4/BF16 GEMM, fused MoE (Mega MoE), MQA scoring for
the lightning indexer, **HyperConnection (mHC / tf32_hc_prenorm_gemm)**.

* `apis/mega_mhc.hpp` — one kernel does: shifted-or-normal **mHC**, i.e.
  `x @ post + comb @ residual`, **generates the next mixes** from `fn` +
  `mix_scales`/`mix_bases`, and applies **RMSNorm** (+ optional FP8 cast).
  Signature: x [T,H], residual [T,hc,H], post [T,hc,1], comb [T,hc,hc],
  fn [hc*(hc+2), hc*H], mix_scales[3], mix_bases[hc*(hc+2)].
  Mapping: our `Block.forward` currently does hc_mixes -> hc_pre -> rms_norm ->
  attention -> hc_post -> hc_mixes -> hc_pre -> rms_norm -> MoE -> hc_post as ~10
  separate tensor ops. The fused form removes the intermediate [T,hc,H] materialisations
  and the per-call overhead (we measured 0.10s + 0.10s per token in the two
  `attn_pre`/`ffn_pre` segments, plus a hard-to-measure hc_post).
* `apis/hyperconnection.hpp` — `tf32_hc_prenorm_gemm`: the HC pre-norm GEMM with
  `sqr_sum` output (RMSNorm partial sums fused into the GEMM).
* `fp8_fp4_mqa_logits` / `fp8_fp4_sparse_mqa_logits` (+ paged variants,
  `get_*_metadata`) — the lightning-indexer scoring kernel: weighted ReLU MQA logits
  over fp8 q against fp4/fp8 paged KV. Our `Indexer.forward` computes the same thing
  with plain torch (q @ k over compressed groups, then top-k); the official kernel
  fuses the weighted-ReLU and the metadata layout.
* `fp8_gemm`/`fp4_gemm` NM layouts + `m_grouped_*` for MoE: group only M, keep N/K
  fixed; our MoE does one small GEMM per expert (6 per layer, M=1) which wastes 31/32
  of a block_M=32 tile. The official shape is one grouped GEMM over the 6 experts.
* `DeepJIT` (below) compiles everything at runtime — no CUDA build step.

Architecture: the shipped kernels target SM90/SM100; this box is SM120 (RTX 5060).
The *algorithms* are portable to tilelang (`T.gemm` lowers to mma on SM120), and the
layout/heuristic structure (block_M/N/K choice, split-K, grouped-M) is what we need.

## DeepJIT — runtime kernel compilation

Lightweight C++/xPU JIT with a low-CPU-overhead module; DeepGEMM and DeepEP compile
their kernels through it. We already have that role filled by tilelang, but two
things are worth copying: (a) kernel cache keyed by the full compile-time parameter
tuple (we hit tilelang cache-key collisions: "W expected 8192 got 4096",
"RES expected M=4 got 1"); (b) compiling on first use behind a lock so parallel
workers do not duplicate work.

## DeepEP — expert-parallel communication

All-to-all dispatch/combine over NVLink/RDMA. Single-box inference does not need the
transport, but two ideas transfer:
* `lb_prefetch_weights` — prefetch expert weights + quant scales **before** the expert
  compute, asynchronously; this is exactly the shape of our `prefetch_to_vram` and
  argues for issuing it a layer earlier rather than for the current layer.
* Engram storage "on GPU or CPU with one wait hook per layer" — our engram tables live
  on disk and are read per token; a CPU-resident hot window with a per-layer hook is
  the same idea.

## deepseek-recipe — prompt/response formatting

Rust + Python bindings that turn Messages/Chat-Completions/Responses into the
official DeepSeek V4/V4.1 prompt and back. This is the authority for the chat format
we hand-build in `scripts/demo_chat.py` (system/user/assistant special tokens,
thinking mode, tool-call protocol, `prefix` continuation). Use it to validate the
prompt instead of guessing.

## DeepInfer — the reference implementation

Already the porting source for the model math (attention, compressor, indexer, MoE,
MTP/DFlash).
