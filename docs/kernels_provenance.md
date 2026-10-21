# Kernel provenance: TileLang first, NVIDIA only as reference

The engine runs **only TileLang and torch** -- no CUDA C++, no CuTe DSL, no Triton,
no FlashInfer / DeepGEMM / b12x / flash-mla binaries. Checked by grepping the package
for those imports (only comments mention them).

## Where each kernel came from

| kernel | source | notes |
| --- | --- | --- |
| sparse_attn | **DeepSeek** `reference/official/kernel.py` (TileLang) | vendored as `official_kernel_ref.py`; level 0-2 default |
| block-FP8 / fp4 GEMM, act_quant, fp4 quant, hc_split_sinkhorn | **DeepSeek** same file (TileLang) | already in the engine |
| block-scaled GEMV (16 features/CTA) | NVIDIA b12x algorithm, **re-implemented in TileLang** | used only above 4096 output features where it wins |
| compress_pool | **DeepSeek** DeepInfer `tilelang_ops/cuda/compress.py` (TileLang) | ported in `deepseek_kernels.py`, replaces the torch pooling |
| engine GEMV / MoE / attention plumbing | our own TileLang | |

## Why DeepInfer over NVIDIA

DeepInfer (DeepSeek's serving stack) writes every operator in TileLang and fans it
out per backend (`tilelang_ops/cuda/`, `tilelang_ops/asc/` for Ascend). That is the
portability the project wants: one source, many accelerators. NVIDIA's b12x is CuTe
DSL and SM120-only, so it is used as an **algorithm reference** and re-expressed in
TileLang rather than imported.

## Authority order for semantics

1. DeepSeek: `config.json` + DeepInfer (`v4_modules/*`, `rotary_embedding.py`).
2. NVIDIA (Flare / b12x / vLLM): execution-structure reference only.

See `parity.md` for the numerical comparison against that authority.

## Still to port from DeepInfer (TileLang, highest value first)

- `fused_transform.py` -- norm + RoPE + Hadamard + fp8/fp4 cast + KV-cache store in
  one kernel (currently 4 separate ops per layer per token).
- `norm.py` -- fused RMSNorm path.
- `index_attn.py` / `get_kv_index.py` -- the indexer's own projections and top-k.
- `fused_sample.py` / `sampler.py` -- sampling without a host round trip.
