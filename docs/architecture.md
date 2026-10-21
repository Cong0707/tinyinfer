# Architecture

## Data flow

    token ids ──> embed (host table) ──> [hc-expand] ──> 40 blocks ──> collapse
                                                              │
                                            each block: hc_mixes -> attn_norm
                                            -> MLA (window / compressed) -> hc_post
                                            -> hc_mixes -> ffn_norm -> MoE -> hc_post
                                                              │
                                                         head -> logits

## Modules

| file | role |
| --- | --- |
| engine.py | orchestration: forward/generate, streaming, OOM ladder, sessions, MTP |
| model.py | Block (one decoder layer), weight packing, device moves |
| attention.py | MLA window attention (sliding window KV cache) |
| compressed_attention.py | compressed layers (latent compressor + indexer) |
| compressor.py | Compressor (gated pooling) and Indexer (top-k slot selection) |
| moe.py | gate, shared expert, routed experts, expert serve loop |
| gemm.py / gemv_*.py | fp8/fp4 linear paths (GEMM for prefill, GEMV for decode) |
| loading.py | shard index, expert blob reader, tiered RAM/VRAM expert cache |
| resources.py | hardware planner: resident layers, ring, cache, chunk |
| tiers.py | OptLevel 0-3 and the feature flags each level enables |
| deepseek_kernels.py | TileLang ports of DeepInfer ops (compress_pool, norm_rope, rope_quant) |
| official_kernels.py | b12x GEMV re-expressed in TileLang |
| tp.py | tensor-parallel shard helpers (untested on this box) |
| serve.py | stdlib HTTP server (/health, /generate) |
| bench.py | reproducible benchmark (build/prefill/decode/cache/disk as JSON) |

## Streaming model

The 8GB card cannot hold 40 layers, so the last 36 layers are streamed:

1. At start-up `_flat_setup` builds one pinned host block per streamed layer (13GB) and
   three device ring buffers (463MB each). A layer's weights are one contiguous block,
   so a layer is a single flat H2D copy.
2. Per layer, `_flat_apply(layer, slot)` copies the block into ring slot `slot` and
   points the layer's weight attributes at views of it; `_flat_prefetch` does this on a
   copy stream and `_flat_wait` makes the compute stream wait for its event.
3. Resident layers (4-5) stay on the device; only the tail is copied per token.

The invariant: every weight attribute is either a real tensor (resident/CPU) or a view
into the live ring. `reset()` re-points streamed layers at their CPU sources but keeps
the ring, so no view dangles.

## Expert path

- Router (Gate) is computed on the device; `bincount` turns the indices into the list of
  distinct experts of the layer.
- `ExpertStore.fetch_async` issues disk reads on a 4-thread pool right after the gate;
  `fetch` waits for those futures and copies into VRAM slots (arena) or leaves them in the
  RAM cache (zero-copy).
- The RAM cache is an ARC policy over 18.8MB blobs (14GB on this box).
- The serve loop accumulates `y[idx] += w*expert(x)` in one scatter instead of per
  expert.

## Sessions

`generate()` starts with `reset()`: clears routing memory, per-layer attention state
(shared attention instances are reused by many streamed layers, so ALL layer states are
cleared), compressors/indexers, expert caches, and the per-forward caches on the shared
handoff slot. Weights and caches stay warm.
