# Round report: acceleration and portability

This round was about making the engine faster, and about making that true on machines
other than the 8GB laptop. Everything below is measured on this box unless it says
otherwise.

## What changed and what it bought

| change | effect | verified by |
| --- | --- | --- |
| planner: CPU-only boxes keep the tables on the host | no more "GPU head" with no GPU | tests/test_planner_hardware.py |
| planner: unified memory books one pool | a 48GB Mac planned 46GB VRAM + 36GB cache before | same |
| planner: refuses a card too small for the streaming skeleton | no OOM after the first chunk; says how much it needs | same |
| planner: trades the 1.32GB embed table for expert slots at 24GB | 197 -> 298 slots, all 240 routable experts resident, decode floor 0.19s -> compute-bound | same + tools/plan_matrix.py |
| fp4 GEMV: pipelined k loop | 1.11x, bit-identical | probe_gemv_pipe.py |
| tp: block-aligned shards, per-layout scale slicing | multi-GPU sharding was numerically wrong | tests/test_tp.py |
| moe: route every row, not just the first | a batch silently served one row per expert; single-token decode was exact, which hid it | probe_fp4_stage.py, decode re-checked |

## The numbers

| machine | resident layers | expert slots | decode floor | note |
| --- | --- | --- | --- | --- |
| this 8GB box | 4 | 12 | 1.28s (0.8 tok/s) | measured 1.66s at 4.5GB/s disk |
| 12GB card | 15 | 15 | 1.18s | disk-bound |
| 24GB card | 40 | 298 | compute-bound (~0.06s) | all routable experts resident |
| 32GB+ | 40 | 240+ | compute-bound | same |
| 4GB card | - | - | refused | needs ~2.9GB just for ring + workspace |
| CPU only | 0 | 0 | host path | minutes/token, parity work only |

## The bug behind the batch-vs-token mystery

The fp4 serve path indexed the activation with `idx[0]` -- the *first row* routed to an
expert -- while the expert's weight vector was applied to that one row. A single token
routes exactly one row per expert, so decode was exact; any batch lost all but one row
per expert and the error grew with the number of shared experts. Both the serve and the
cached path were fixed to use the (rows, cols) pair from `torch.where`.

After the fix the expert module is exact for multi-row input. The remaining chunk-vs-token
difference (42%) is in the compressed attention path: `compress_len` is computed once per
chunk from the chunk's end position, so the first positions of a chunk can see compressed
groups that belong to later positions. Chunked prefill therefore stays at chunk=1; that is
the next real prefill speedup (8x) once the compressed path is made position-exact.

## What is still unproven

- Tensor parallel on real multi-GPU hardware (the math and the slicing are verified, the
  distributed run is not).
- CUDA graphs: capture is keyed on `start_pos`, which advances every decode step, so the
  captured graph never replays. A real decode graph needs device-side positions.
- The 24GB compute floor is an estimate from the 8GB measurements, not a run.
