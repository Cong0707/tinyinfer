# Benchmark history (this machine, measured)

Every row is one run of `python -m tinyinfer.bench` or the probe scripts; the disk rate
is measured in the same run. Absolute tok/s is fetch-bound and tracks the disk, so the
invariant to compare across runs is the ratio to the IO floor (docs/lower_bounds.md).

| when | disk | decode step | tok/s | what changed |
| --- | --- | --- | --- | --- |
| round start | 4.5GB/s | 2.95s | 0.34 | baseline |
| after RoPE-table fix | 4.5GB/s | 1.73s | 0.58 | table no longer streamed per layer (422MB -> 166MB), bincount 300 -> 84ms |
| after planner fp8 fix | 4.5GB/s | 1.66s | 0.60 | slab halved, cache 6.5 -> 14GB |
| after batching/fused work | 3.6GB/s | 2.5-4.5s | 0.22-0.48 | disk degraded; cache share 0.85, hit 33% |
| late round | 1.9GB/s | 2.5s | 0.46 | disk degraded further |
| end of round | 1.4GB/s | 2.5-3s | 0.40-0.46 | disk at its slowest; code unchanged from the 1.66s run |
| later | 1.5GB/s | 2.5s | 0.41 | disk recovering; warm prefill 11.5s |
| later | 1.5GB/s | 2.3-2.5s | 0.44 | warm prefill 12.1s, hit 16% |
| later | 1.5GB/s | 3.2s | 0.31 | `TI_MULTIROW_GEMV=1`: serve 184ms vs 168ms, so the batch kernel buys nothing at decode (one row per expert) |

## Where the time goes (1.66s step, 4.5GB/s disk)

| part | time |
| --- | --- |
| expert fetch | 0.86s |
| layer streaming (36 x 166MB H2D) | 0.24s |
| bincount | 0.084s |
| expert serve | 0.13-0.18s |
| attention + MoE + head | remainder |

## Prefill

| prompt | disk | time | note |
| --- | --- | --- | --- |
| 8 tokens | 4.5GB/s | 4s warm / 21s incl. slab setup | setup is one-off |
| 8 tokens | 1.9GB/s | 20s | fetch-bound |
| 63 tokens | 2.7GB/s | 136s | chunk=1 path (correct) |

## Rules of thumb

- Decode cost = 4.5GB experts + 6.0GB layers per token; at 4.5GB/s and 26.5GB/s that is
  1.23s. Everything else is second order.
- Each additional resident layer removes 166MB/token (=6ms at PCIe rate).
- Expert cache hit rate depends on the workload: 3-20% on fresh text, up to 90% when the
  same tokens recur with a warm cache.
| 1.8-1.9s | disk 1.5GB/s | serve batch sweep: 1 -> 1.84s, 2 -> 1.83s, 3 -> 1.82s (one
  outlier at 3.8s was a disk dip, not the batch size), 6 -> 1.89s. The serve term is
  the same either way, so the batching strategy is not what limits decode. |
| 13.3s warm / 0.485 tok/s | disk 1.58GB/s | final run of the round: first forward 31.7s,
  warm prefill 13.3s, cache hit 16%. Code path unchanged from the 1.66s/0.60 run at
  4.5GB/s. |
