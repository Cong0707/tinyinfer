# Round outcome (one page)

## Speed

| what | before | after |
| --- | --- | --- |
| fp4 GEMV (decode, per expert) | 0.071ms | 0.064ms (1.11x, bit-identical) |
| decode on this box | 0.49 tok/s | 0.49 tok/s (disk 1.58GB/s; the disk decides) |
| decode on a 24GB card (plan) | 5 tok/s (197 slots) | ~18 tok/s (298 slots, all 240 experts resident) |
| multi-GPU sharding | numerically wrong scale slices | exact, 128-block aligned |
| MoE batch serving | one row per expert (a bug) | all routed rows |

## Portability

The planner is tested against nine machines (tests/test_planner_hardware.py) and prints
them with predicted floors (tools/plan_matrix.py): a 4GB card is refused with the minimum
size, a 6GB card runs, CPU-only boxes keep the tables on the host, unified memory books
one pool, and a 24GB card gets full expert coverage.

## Prefill (this round's second half)

| prompt | path | time | vs token-by-token |
| --- | --- | --- | --- |
| 12 tokens | chunk=1 (exact) | 18.3s | 1.00x |
| 12 tokens | chunk=4 | 11.6s | **1.58x** |
| 63 tokens | default chunk (8) | 79.7s | **1.7x** (136s before) |

The error of the chunked path is an accumulation (~0.95% at position 0, 57% by position 3
after 40 layers), not a broken step: compressor, MoE, indexer score and attention each
measure at their noise floor. Short prompts answer correctly (Paris, pi, multi-turn);
long prompts should use `TI_PREFILL_CHUNK=1`. Two real defects were fixed on the way
(candidate blocks not clipped to per-query visibility; index-list width varying with the
chunk), and the remaining work is to make the compressed path bit-identical.

## Honesty items

- **The session prefill drift (18-20 logits) is not fixed.** Five theories were tested and
  reverted; the last measurement (MoE forward on 96 rows is exactly reproducible) rules
  out nondeterministic index_add. It stays an open item with a failing (xfail) test.
- **The VRAM slot pool makes runs non-reproducible.** `TI_NO_SLOTS=1` gives exact repeats
  at ~40% slower decode. Earlier batch/chunk deltas in docs/analysis.md were measured
  through that noise and are marked as such.
- **The 24GB compute floor (~0.06s/token) is an estimate**, not a run: no such card is
  available here.
- **Tensor parallel has never run on real multi-GPU hardware.**

## Where the next hour of work belongs

1. Deterministic accumulation path for prefill (the drift) -- blocks chunked prefill and
   therefore the 8x prefill a long prompt needs.
2. Slot-pool race (reproducibility).
3. Compressed-layer parity 0.87%.