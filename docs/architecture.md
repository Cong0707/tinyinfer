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
| kernels/cuda/*.cu | the native kernels: act_quant, fp4_quant, fp8_gemv_bs, gemv_fp4_w4a8, gemv_fp4_gate_up, fp8_gemm_mma, sparse_attn_sink, moe_gemv6, rms_norm, hc_pre_rms_norm, sinkhorn |
| kernels/native.py | loads the cubins through the CUDA driver API (no torch headers, no extension build) |
| kernels/quant.py, fp4q.py | the quantiser entry points that wrap those kernels |
| kcheck/ | conformance cases: every native kernel compared against a reference written in plain torch |
| tp.py | tensor-parallel shard helpers (128-aligned; see the note below) |
| ep.py | expert parallelism: who owns which expert, and the partial-sum reduce |
| server.py | FastAPI OpenAI-compatible service (/v1/models, /v1/completions, /v1/chat/completions, /health), SSE streaming, API keys, per-session slots; driven by cli.py |
| cli.py | `python -m tinyinfer.cli serve|generate` |
| bench.py | reproducible benchmark (build/prefill/decode/cache/disk as JSON) |

## Streaming model

The 8GB card cannot hold 40 layers, so the last 36 layers are streamed:

1. At start-up `_flat_setup` builds one pinned host block per streamed layer (13GB) and
   three device ring buffers (463MB each). A layer's weights are one contiguous block,
   so a layer is a single flat H2D copy.
2. Per layer, `_flat_apply(layer, slot)` copies the block into ring slot `slot` and
   points the layer's weight attributes at views of it; `_flat_prefetch` does this on a
   copy stream and `_flat_wait` makes the compute stream wait for its event.
3. Resident layers stay on the device; only the tail is copied per token. How many is the
   planner's decision, not a constant: 11 of 40 on an 8GB card, all 40 at 24GB.

The zero-copy path above (one flat H2D per layer) only engages when a layer's host bytes
are contiguous. On Windows they are scattered over terabytes of address space by
cudaHostAlloc, so the per-tensor copy_ path runs instead and a layer costs 28 DMAs
rather than one. Same bytes, more launches; see docs/PERFORMANCE.md.

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
## How the engine decides

Nothing below is a hardware table ("if 24GB then ..."). Each decision is solved from the
checkpoint's measured byte sizes and the card's measured bandwidth, so a machine that
does not exist yet still gets an answer -- `scripts/predict.py` prints that answer for any
VRAM figure without needing the hardware.

| decision | made by | input |
| --- | --- | --- |
| resident layers | `resources.plan` | VRAM budget, per-layer bytes, ring size |
| expert slots | `resources.plan` | remaining VRAM / expert blob size, must beat 15% coverage |
| host cache | `resources.plan` | RAM after the pinned slab is reserved |
| MTP on/off | `resources.plan` | whether the draft stages still leave room for a useful resident set |
| streamed layer count | `engine._stream_layers` | the plan's resident count |
| per-layer shard split | `tp.asplit` | capacity-proportional, 128-row aligned |
| which card owns an expert | `ep.ExpertRoute` | contiguous blocks proportional to free VRAM |

The two that are easy to get wrong:

**Expert slots have a floor.** Below ~15% coverage the pool costs more than it saves --
every non-resident expert pays a DMA plus an event wait per use, and a slot still marked
ready from a previous use can hand back the wrong bytes. Measured: 96 slots produced a
different first token, 13 slots made two identical greedy runs disagree. So when the
coverage is too low the planner sets slots to 0 and takes the host-read path instead,
which is a deliberate choice rather than a failure.

**TP cannot split below 128 rows.** fp8/fp4 scales are 32x32 blocks, so a shard boundary
that is not a multiple of 128 would have two cards each read half of one scale block.
Consequence: a 384-row tensor splits into 3 pieces, not 4, and a 100-row tensor does not
split at all. This is why expert parallelism (granularity: one expert) is the multi-card
story for this model, matching DeepSeek's own EP + EPLB deployment.
