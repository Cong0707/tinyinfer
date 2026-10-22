# Environment switches

Zero configuration is the supported default: the planner reads VRAM/RAM/disk and picks
layer residency, ring size, cache size and prefill chunking. These variables exist for
measurement, A/B testing and for promoting a mechanism once it is validated. None of
them is required to run.

## Planning / capacity

| var | meaning |
| --- | --- |
| TI_TIER | optimisation level 0-3 (see tiers.py); level 2 is the supported default here |
| TI_TOPK | routed experts per token (0 = checkpoint top-6; lower trades quality for IO) |
| TI_GPU_LAYERS | force the number of resident layers instead of planning it |
| TI_CHUNK | prefill chunk size (long prompts are stepped down automatically) |
| TI_RING | streaming ring buffers (2-4, planner picks) |
| TI_RAM_FRACTION, TI_RAM_RESERVE_GB | host RAM budget knobs |
| TI_CACHE_POLICY | expert cache policy: arc (default) or lru |
| TI_PIN_CACHE, TI_PIN_CACHE_GB | pin part of the expert cache (direct async DMA) |
| TI_SLAB, TI_FLAT_PIN_GB | pinned layer-slab sizing |
| TI_FLAT | 1 forces flat streaming on, 0 off; unset = planner decides |

## Numerics / semantics

| var | meaning |
| --- | --- |
| TI_YARN_MSCALE | 1 (default) follows DeepInfer's yarn_get_mscale; 0 matches the plain HF table |
| TI_FUSED_KV | 1 enables the fused rope+fp8 KV kernel (verified exact; off by default) |
| TI_BF16_CACHE | 1 keeps up to 8 bf16 expert copies per layer (VRAM cost, use on big cards) |
| TI_NO_MASK_CACHE | 1 disables the indexer visibility-mask cache |
| TI_MULTIROW_GEMV | 1 opts into the multi-row fp4 kernel (exact on synthetic weights, not on real ones) |
| TI_NO_SLOTS | 1 disables the VRAM expert-slot pool: deterministic logits, ~40% slower decode (docs/analysis.md §10) |
| TI_PREFILL_CHUNK | token block size for prefill; 1 = exactly reference-matching, planner's value ~1.6x faster but 0.5%/layer off (docs/analysis.md §13) |
| TI_REF_ATTN | 1 routes sparse attention through the PyTorch reference (diagnostic; slow) |
| TI_DEBUG_ATTN | 1 prints the compressed layer's group bookkeeping per call (diagnostic) |
| TI_MAX_RESIDENT | cap on resident layers: less VRAM pressure, only ~25% vs 12.5% argmax agreement on flat prompts (docs/analysis.md 19) |

## Measurement

| var | meaning |
| --- | --- |
| TI_TIME | per-step decode timing |
| TI_MOE_PROF | MoE phase breakdown (gate/bincount/serve/fetch/shared) |
| TI_WALL_PROF | per-block wall-clock stage timing |
| TI_LAYER_TIME | per-layer total time |
| TI_MEM_TRACE | per-layer GPU memory allocation trace |
| TI_ATTN_PROF | compressed attention phase timing |
| TI_HC_DEBUG, TI_HC_T, TI_HC_GPU, TI_HC_NOFUSE | hyper-connection debug/CPU-fallback switches |
| TI_DEBUG_NAN, TI_DEBUG_FLAT, TI_DEBUG_IDX, TI_SPEC_DEBUG, TI_TL_DEBUG | targeted debug prints |
| TI_EMBED_GPU | keep the embedding table on the GPU (needs VRAM) |
| TI_PREFETCH | gated expert prefetch on/off |
| TI_ASYNC... (MTP/Flare flags live in tiers.py, not here) | |

## Notes

- `TI_FUSED_KV` and `TI_BF16_CACHE` are the two switches with a VRAM/numerics trade-off;
  the defaults are the conservative, aligned ones.
- Everything under Measurement only prints; it never changes results.
- The tier flags (cuda_graph, mtp, speculation, official_gemv, ...) live in
  `tiers.py` and are documented in `docs/level2_mechanisms.md`.
### Switches for measurement or A/B that were missing from the table above

| name | meaning |
| --- | --- |
| TI_CACHE_GB | host expert-cache size in GB; overrides the room-after-slab calculation (docs/analysis.md §22) |
| TI_READERS | disk reader threads (default min(16, cores)); 32 normalises ~5% better on a big-CPU box |
| TI_SERVE_BATCH | experts served per batch in the MoE loop (default 3; 1 is slower, 6 no better) |
| TI_SLOTS | expert VRAM slots; overrides the planner's pool size |
| TI_MTP | 1 forces the MTP draft stages on even when the plan says they do not fit (accept-rate measurement) |
| TI_V | 1 enables the v5 double-buffered streamed layers (off by default: measured +OOM on 8GB) |
| TI_IOCP | 1 switches expert reads to the IOCP reader (measured 2.3x slower in-engine) |
| TI_NO_SYNC | 1 skips the routing host sync (experiment only: routing becomes wrong) |
| TI_SA_BLOCK / TI_SA_HCHUNK | sparse-attention kernel tiling (32 / 16 defaults) |

Everything else in `src/` (TI_DEBUG_*, TI_*_PROF, TI_TIME, TI_LAYER_TIME, TI_MEM_TRACE,
TI_WALL_PROF) only prints diagnostics and changes no value.


| TI_SERVER_PREFIX | 1 makes the HTTP server resume from a slot's previous prefix instead of re-prefilling it (opt-in; the OpenAI API is stateless, so the client must send the growing conversation on the same slot). Experimental: validated by the unit tests and the engine mechanism, not yet measured end to end over HTTP. |
