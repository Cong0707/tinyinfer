# Toward 1 tok/s on an 8GB card (optimization plan, 2026-10-21)

Baseline measured after the full 510GB model landed at E:\Models:
**0.30-0.33 tok/s** (40 layers, 11 resident + 29 streamed, engram L1+L14 on).

## Where the 3.1s/token goes (torch profiler, 1 decode step, 40 layers)

| item | time | share | note |
|---|---|---|---|
| H2D Memcpy (pageable) | 590ms | 54.8% | 763 copies x 781us; 29 streamed layers x 190MB at 8.3GB/s |
| fp8 GEMM (M=1) | 195ms | 18.2% | 221 calls x 884us; block_M=32 wastes 31/32 of the tile |
| tilelang sparse_attn | 134ms | 12.4% | 32 calls x 4.18ms; M=1 grid is inefficient |
| HC bmm (fp32 einsum) | 91ms | 8.5% | 48 calls x 1.90ms |
| fp8 GEMV (bf16 x fp8) | 91ms | 8.5% | 48 calls x 2.86ms |
| CPU aten::copy_ side | ~406ms | - | per-tensor staging/contiguous work before H2D |

We are not I/O-bound at steady state (warm expert cache); we are **transfer- and
small-kernel-bound**.

## Levers, ordered by expected win

1. **Overlap H2D with compute (up to -590ms).** The single biggest win. Reference:
   ds4's streaming path issues pool preads for the next layer while the GPU still runs
   the current layer, plus cudaMemcpyAsync on an upload stream. We have the v4
   streaming path for compressed layers and the (disabled) v5 double buffer; the 8GB
   card could not hold two full layer buffers (v5 peaked at 9330MB and paged). Options:
   - single 190MB staging buffer + torch.cuda.Stream upload for the *next* layer's
     tensors while the current layer computes (one in-flight layer, no second full copy)
   - or pinned per-tensor buffers limited to ~1GB total (still ~3x pageable bandwidth)
2. **M=1 GEMV for fp8 (-100..150ms).** gemv_fp8.py already contains fused_fp8_gemv
   (173-184) but fp8_linear imported a name that does not exist; wiring it crashed the
   CUDA kernel (illegal access in gemv_bf16_kernel at N=1280/K=5120) - needs a debug
   pass on the rows=16 grid. Also fixes the same waste in shared-expert GEMMs.
3. **tilelang sparse_attn M=1 (-80..100ms).** 4.18ms/call is far above the ~0.3ms the
   FLOP count implies; the kernel is configured for batched prefill. Reference: vllm.cpp
   uses a separate decode kernel with one block per (query, head-chunk).
4. **HC bmm and small ops (-60ms).** 48 x 1.9ms for a 164KB problem is launch overhead
   (measured: CPU does it in 0.22ms, GPU 3.2ms for the same math). Batch the 40 layers'
   HC post/comb into one call, or move the small ones to the CPU (measured slower here
   only because of per-call H2D, so batching is the answer).
5. **Prefill/decode asymmetry**: prefill M=8..64 already works (71-290s first time due
   to JIT, then cached). Long context decode is sparse by design.

## Realistic target

At 8GB the card holds 11 layers; 29 streamed layers x 190MB = 5.5GB/token of H2D.
If (1) fully overlaps H2D (590->~100ms residual), (2) saves ~150ms of GEMM, (3)
saves ~100ms of attention, (4) saves ~60ms, the step drops to roughly **1.6-1.8s
(0.55-0.62 tok/s)**. Reaching 1.0 tok/s needs the H2D to be both overlapped AND
smaller: more resident layers (fp4 attention / smaller KV would free ~1GB -> +2
layers) or a 16GB+ card. Honest statement: **1 tok/s is not reachable on this 8GB
part with the current exact-weight streaming design**; 0.55-0.65 is, and 1+ becomes
reachable on 16GB.
