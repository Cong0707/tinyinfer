# Hardware matrix: what the engine plans and what it predicts

`tools/plan_matrix.py` runs the planner against a table of hypothetical machines (no GPU
needed) and applies the IO arithmetic from `docs/lower_bounds.md`. The point is that a
second machine can be checked before it is bought or borrowed, and verified after.

## The table (measured constants, measured checkpoint geometry)

| machine | resident | stream | expert slots | chunk | host cache | decode floor | predicted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5060 8GB / 47GB (this box) | 4 | 36 | 12 | 8 | 13.5GB | 1.28s | 0.8 tok/s |
| RTX 3060 12GB / 32GB | 15 | 25 | 15 | 8 | 7.9GB | 1.18s | 0.8 tok/s |
| RTX 4090 24GB / 64GB | 40 | 0 | 298 | 64 | 34.7GB | 0.06s | ~18 tok/s |
| RTX 5090 32GB / 128GB | 40 | 0 | 502 | 64 | 88GB | 0.06s | ~18 tok/s |
| A100 80GB / 256GB | 40 | 0 | 2332 | 64 | 128GB | 0.06s | ~18 tok/s |
| 2x RTX 4090 48GB / 128GB | 40 | 0 | 1055 | 64 | 88GB | 0.06s | ~18 tok/s (per GPU, no TP) |
| M3 Max 48GB unified | 40 | 0 | 563 | 64 | 2.5GB | 0.06s | ~18 tok/s |
| GTX 1650 4GB / 16GB | - | - | - | - | - | - | refused (needs ~2.9GB of skeleton) |
| CPU only | 0 | 40 | 0 | 8 | 28.6GB | - | runs, very slowly |

Measured on this box: 1.66s/token (0.60 tok/s) at 4.5GB/s disk, i.e. 75% of the 1.28s
floor; the rest is fetch variance and bincount.

## Why 8GB and 24GB differ by 50x

Two terms dominate a token:

1. **streamed layers**: 36 x 166MB H2D on 8GB, zero when every layer is resident (24GB+).
2. **expert misses**: a token can route to `topk x n_layers = 240` distinct experts
   (4.5GB of blobs). The planner now sizes the VRAM slot pool to cover **all** of them
   when the card allows it, and moves the 1.32GB embedding table back to the host to make
   room (a 10KB row lookup per token costs ~1ms, the slots save ~1s).

At 24GB the pool reaches 298 slots, every expert the token can ask for is already in
VRAM, and the decode floor collapses to compute: 40 layers x (18 expert GEMVs at
M=1 plus the dense GEMMs) is estimated at ~56ms/token, i.e. ~18 tok/s. That compute
figure is an estimate from the 8GB measurements, not a big-card run.

## What each class of machine is bound by

| class | bound | lever |
| --- | --- | --- |
| 6-12GB | disk (expert misses) | faster volume, larger host cache, LRU/ARC; slots barely help (15 slots = 6% coverage) |
| 16GB | disk, partially | residency grows, slot coverage still low |
| 24GB | compute + launch overhead | full coverage; CUDA graph would matter here (not implemented: see below) |
| 32GB+ | compute + launch overhead | as above; chunk 64 for prefill |
| CPU only | host bandwidth | runs at minutes/token; only useful for parity checks |

## Status of the mechanisms that only matter on bigger boxes

- **Multi-GPU (tensor parallel)**: sharding is implemented and now numerically verified
  (`tests/test_tp.py`: block-aligned shards, per-layout scale slicing). It needs
  `torch.distributed` initialized; no multi-GPU box has been exercised, so treat the
  end-to-end TP path as unproven.
- **CUDA graph**: capture exists but is keyed on `start_pos`, and decode advances the
  position every step, so it never replays. A real decode graph needs device-side
  position handling in the attention kernels; until then the tier flag is a no-op
  (`docs/ROADMAP.md`).
- **MTP**: needs the batch-vs-row MoE path to be exact; currently off by default on 8GB
  and unproven elsewhere.

## Verifying a different machine

    # before: what the planner would choose and the IO floor it implies
    PYTHONPATH=src python tools/plan_matrix.py

    # after: the same numbers measured on that box
    PYTHONPATH=src python -m tinyinfer.bench --model <dir>

Compare `decode_tok_per_s` with the `floor` column above and with
`docs/PERFORMANCE.md`. If a box is far below its floor, the usual causes are a slow
checkpoint volume (read `disk_gb_s` in the same bench output) or another process
holding VRAM.
## Choosing a machine for this checkpoint

Measured constants: 4.5GB of expert reads per token, 6.0GB of streamed layers on an 8GB
card, 26.5GB/s pinned H2D.

| budget | verdict |
| --- | --- |
| 6-8GB VRAM | works, ~0.8 tok/s floor; the disk is the bottleneck, a faster NVMe helps most |
| 12GB | works, same disk bottleneck; a few more layers resident |
| 16GB | works; residency grows, expert coverage still low |
| 24GB | the first class where every routable expert fits in VRAM: the IO term vanishes and the card becomes compute-bound (~10-20 tok/s estimated) |
| 32GB+ | same, more headroom for KV and batch |
| <6GB | refused with an explicit message (needs ~2.9GB for ring + workspace alone) |
| 47GB+ RAM | required for the pinned slab (~13GB) plus the expert cache; 32GB runs with a small cache |

Two things do not depend on the card: the checkpoint is 510GB, so the volume must be
fast (4.5GB/s measured best, 1.5GB/s after long sweeps) and the host needs ~13GB pinned for
the streaming slab. A machine with a small SSD will be slower than the 8GB laptop.

## What the planner decides automatically (and why)

These are the rules that make the same command line behave correctly on every box; all of
them are driven by measured geometry, not by a tier name.

| decision | rule | measured reason |
| --- | --- | --- |
| keep the expert slot pool? | only if it covers >= 15% of `topk*n_layers` (240) experts | below that, every use pays a DMA and an event wait: -15% decode and slower prefill on the 8GB card (analysis.md §16) |
| enable MTP/speculation? | only with >= 16GB VRAM | accept rate 0.08 here; the draft stages cost 40s per short answer and 3-4 resident layers (§16) |
| how big is the host expert cache? | host_room - slab_need - 3GB, capped by the planner's number | the old flat 35% of host room under-sized it by 27% (§22: hit 16.3% -> 25.5%, decode +7.7%) |
| how many resident layers? | as many as fit after ring, tables and workspace | each one removes a 166MB H2D per token (~6ms) |
| how many expert slots at all? | `spare * 0.70 / blob`, at least 12 before trading a layer | a handful of slots is worse than one fewer layer |
| prefill chunk | 8, step down for long prompts | 8 is the workspace sweet spot; chunk=4 is 1.58x faster than token-wise (§13) |

On a 24GB card the same rules give: pool on with 298 slots (all 240 experts), MTP on,
everything resident, 64-token prefill chunks -- i.e. the IO term disappears and the card
becomes compute-bound.

