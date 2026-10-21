# Performance in one page

## What a token costs (measured, 4.5GB/s disk)

| term | bytes | time |
| --- | --- | --- |
| routed experts (6 x 40 x 18.8MB) | 4.5GB | 1.00s at 4.5GB/s |
| streamed layers (36 x 166MB H2D) | 6.0GB | 0.23s at 26.5GB/s |
| attention + MoE + head | - | ~0.4s at this size |
| **total** | | **1.66s = 0.60 tok/s** |

The floor is 1.23s (0.81 tok/s) before compute. Nothing in software goes below it without
fewer routed experts or more VRAM; docs/lower_bounds.md has the arithmetic.

## What changes it

| lever | effect |
| --- | --- |
| disk speed | linear on the 1.0s expert term; a 1.4GB/s volume gives 2.5-3s steps |
| resident layers (+1) | -6ms/token (one fewer 166MB H2D) |
| expert cache hit rate | each hit removes 18.8MB of disk read |
| top-k 6 -> 4 | ~-0.3s/token, visibly degraded output (not usable) |
| prefill chunk | long prompts use chunk=1 (exact) until the batch path is fixed |

## Where time is not spent any more

- RoPE table: was 256MB/layer streamed (9.2GB/token), now a shared constant.
- Small-M GEMV: one launch per row -> one launch per batch (1.6x, exact).
- Expert activation quantisation: once per layer instead of once per expert per row.
- Indexer mask: cached per (start_pos, s, t, ratio) instead of rebuilt per layer.
- Session reset: no longer rebuilds the pinned slab (it leaked/freed the host grant).

## How to measure it yourself

    PYTHONPATH=src python -m tinyinfer.bench --model <dir>

The output includes the disk rate measured in the same run; compare tok/s only against
runs with a similar disk rate, otherwise compare the ratio to the 1.23s floor.
