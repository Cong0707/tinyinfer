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