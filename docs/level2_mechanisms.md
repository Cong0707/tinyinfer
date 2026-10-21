# Level 2 (engine tier) — what is implemented

Default tier is `OptLevel.AGGRESSIVE` (level 2). Tier definitions (user-defined):
0 = reference PyTorch, 1 = production-proven ports (official kernels + NVIDIA Flare
mechanisms), 2 = our engine's features, 3 = approximations.

A machine that cannot hold every layer *refuses to start* below level 2 with an explicit
message: layer streaming and the expert caches are engine (level 2) capabilities.

## Real engineering (not wiring)

**Tensor parallel, end to end.** `tinyinfer/tp.py` provides `Shard.columns/rows`,
`shard_linear` (slices a quantized projection without breaking fp8 128×128 / fp4 32×32
block boundaries) and `all_reduce_sum`.

* `MLAWindowAttention._shard_for_tp` and `MLACompressedAttention._shard_for_tp`:
  q_lora output sliced, `q_norm` sliced with it, heads sliced (`n_local_heads`),
  `sink` sliced, `wo_a` grouped matrix sliced on its o_lora dimension, `wo_b` sliced on
  its input and all-reduced after `wo_b`. `wkv` is replicated because the model has one
  KV head — sharding it needs a different attention layout, and Flare's QSA does the same.
* `MoE` runs expert-parallel: a rank serves only the experts it owns, the shared expert
  runs on rank 0 only, and the partial outputs are summed once at the end.
* `Engine` joins the process group from `RANK`/`world_size` and passes the shard into
  every `Block`; `tp_ranks == 1` leaves every path byte-identical.

**KV windows are pre-allocated.** Both attention classes used to `torch.cat` the KV
history (and `.contiguous()` it) every token — a fresh allocation per layer per token,
which is exactly what blocks CUDA graph capture. They now shift and append inside a
pre-allocated buffer and hand out a *view*; the compressed-KV group buffer is
pre-allocated to a decode-sized run as well. This is the prerequisite for graphs, and it
removes ~40 allocations/token regardless.

**Scheduler as quantum selection.** `tinyinfer/scheduler.py`: a request table with
EEVDF-style virtual time, live-decode priority (a decode stream that waited longer than
one quantum is served first), and quantum choice by useful work per second over the
prepared capacities. It allocates nothing and touches no model.

**Plan-driven speculation gate.** `resources.plan(speculation=...)` prices the draft
stages (measured 1.36GB resident on this checkpoint) and only keeps them when the box can
still hold the streaming ring plus four resident layers — the measured crossover. On the
8GB laptop it therefore reports `speculation=False` and the engine prints:
`MTP off: the draft stages do not fit beside the streaming ring on this card`, then runs
with 5 resident layers instead of 2. On 16GB and above it stays on.

## Ported mechanisms (level 1 already, level 2 on top)

CUDA graph capture (capture/restore around the double forward, self-disabling), prepared
capacity catalog + `warm_graph_catalog`, DSpark feedback-window accumulation, async
expert fetch (reads issued before serving cached experts), batched scatter, contiguous
slot arena, disk prefix checkpoints with atomic rename, resume sampling from the saved
hidden vector.

## Verification

`compileall` clean over the whole package; `OptLevel` table verified; models/graph/slot
counts verified per VRAM size. Level-2 zero-config on this laptop:

```
[tinyinfer] layers=5/40 stream=35 buf=3 slots=16 cache=10.1GB chunk=8 embed=cpu head=gpu
[tinyinfer] MTP off: the draft stages do not fit beside the streaming ring on this card
prefill 34.0s
step 0 4.41s   step 1 8.30s   (fetch 2.37s / 3.17s, reader 3.65-3.74 GB/s)
```

The disk side is noisy on this box (the same configuration has measured 1.5s and 3.2s of
`fetch` in consecutive runs, with `reader` between 3.6 and 4.7GB/s); the mechanism-level
numbers (bincount 0.30s, serve 0.15s) are stable.

## Added after this document was written

| mechanism | where | reference |
| --- | --- | --- |
| fused rope + fp8 KV round trip | `deepseek_kernels.rope_quant_tl` | DeepInfer `fused_transform.py`; verified bit-exact, opt-in `TI_FUSED_KV=1` |
| batched small-M fp4 GEMV | `gemv_fp4.gemv_fp4_m8` | our own; flat grid; exact on synthetic weights, opt-in for real ones |
| grouped expert GEMV | `gemv_fp4.gemv_fp4_grouped` | our own; exact but loses to the row loop once the plane gather is counted |
| one activation quantisation per expert batch | `MoE.forward` + `Expert.forward_fp4_pq` | ours; the quantiser is expert-independent |
| indexer visibility-mask cache | `Indexer.forward` | ours; the mask depends only on (start_pos, s, t, ratio) |
| expert hot tracking | `ExpertStore.hot_experts/hot_prefetch` | ours; explicit, never inline (256 experts = 4.7GB) |
| dynamic prefill chunk for long prompts | `Engine._forward_impl` | ours; long prompts step down to chunk=1, which is the exact path |
| HTTP + OpenAI-compatible endpoints | `tinyinfer/serve.py` | ours; stdlib only |
| reproducible benchmark | `tinyinfer/bench.py` | ours; JSON with disk measured in the same run |

Each of these is behind a tier flag or an env switch (docs/ENV.md); the tier-2 defaults
are the ones that passed their numeric checks.
