# Decode speed and session handling

## Current (RTX 5060 8GB + 47GB, level 2, top-6 experts, quality preserving)

| metric | value |
| --- | --- |
| prefill (8 tokens) | 21s |
| **decode step** | **1.66s (0.60 tok/s)** |
| expert fetch | 0.86s |
| layer streaming | ~0.24s (36 layers x 178MB) |
| bincount + serve | 0.22s |
| sessions | two consecutive generate() calls both answer correctly |

History this box: 2.95s (0.34 tok/s) at the start of the round.

## Fixes with the largest effect

1. **RoPE table streamed as a per-layer weight.** `Block._weight_paths()` included
   `attn.freqs_cis`: 1,048,576 x 32 complex64 = 256MB **per layer**, identical in
   every one. 36 layers x 256MB = 9.2GB per token of pure duplication. It is a model
   constant; `rope_cache.py` now hands out one tensor per (dim, base, seqlen, YaRN
   args) and it is no longer in the weight path. Streamed bytes/layer 422MB -> 166MB,
   step 2.95s -> 1.73s, prefill 33s -> 21s, bincount 300ms -> 84ms.

2. **Planner assumed bf16 (2x header bytes).** The non-expert weights are fp8 and
   `_weight_paths` returns them in that form, so the slab halved and the freed RAM
   became expert cache (6.5GB -> 14GB, hit 25% -> 31%).

3. **Session reset was broken** (this is why a second generate used to produce BOS
   tokens or NaNs):
   - `_cpu_refs` was a positional list; dropping the RoPE table from
     `_weight_paths` shifted every later slot by one, so `to("cpu")` restored the
     wrong tensors into hc_*/norm slots. It is now a dict keyed by (owner, name),
     in Block and in all four engine tables (flat setup, slab, stream buffers, slots).
   - The pinned slab's host grant was never released, so a rebuilt slab ran out of
     budget and came up partially initialised (silent numeric corruption). Reset
     re-points streamed blocks at their CPU sources, releases the grant, then drops
     the ring.
   - `generate()` starts with `reset()`.

## Top-k trade-off

| routed experts | step | tok/s | "The capital of France is" |
| --- | --- | --- | --- |
| 6 (checkpoint) | 1.66s | 0.60 | `Paris` |
| 4 | 1.36s | 0.74 | repeats the prompt |
| 2 | 0.81s | 1.23 | `[T] 1.` |

1 tok/s needs dropped experts (quality) or more VRAM/RAM; 4.5GB of expert IO at the
measured 4.5GB/s is already 1.0s.

## MTP status

The draft stages load and do not disturb the main model (plain generate with MTP
loaded still answers "Paris"), but `speculative_generate` drafts BOS for every
position, so the accept rate of 1.0 is vacuous. The draft path needs to be aligned
with DeepInfer `model_executor/models/mtp/dflash.py` (forward_drafts / sample_drafts);
on this card the planner keeps MTP off anyway because its 1.36GB does not fit beside
the streaming ring.

## Round additions

- **Batched small-M GEMV**: fp4_linear looped one GEMV launch per row (8 launches per
  projection at prefill = 144 per layer for six experts) and launch cost dominated the
  expert serve. Rows now run in one launch (grid y = row); verified identical
  (rel 0.000000) and 1.64x faster in isolation.
- **One activation quantisation per layer**: the fp4 quantiser does not depend on the
  expert, so the batch is quantised once and reused (forward_fp4_pq).
- **Scale planes stay packed**: expert scales are per-row [N, nb] in the checkpoint.
- **MTP window seeding**: at start_pos == 0 a draft stage now seeds its main-KV window
  from the target-layer hiddens (official DSparkBlock.forward); draft accept went from
  0.03 to 0.08, still far from useful.
- **LRU vs ARC**: same cold-start hit rate (about 3.7%); the 90.7% seen once was a
  repeated-token artefact of a warm cache, not a policy property.

## Expert cache tuning (late round, disk at 3.6GB/s)

Raising the planner's cache share from 0.80 to 0.85 of the free host RAM (14.6 -> 15.5GB
here) lifted the measured hit rate from 3.7% to 32.9% on the second decode step
(workload: one short prefill then a decode). Fetch still dominates because the
checkpoint volume degraded to ~3.6GB/s after hours of repeated sweeps: step 0 7.0s,
step 1 4.5s at that disk rate, versus 1.66s when the same device sustained 4.5GB/s.
The IO arithmetic in docs/lower_bounds.md is the invariant; absolute tok/s tracks the
current disk rate.
