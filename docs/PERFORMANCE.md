# Performance in one page

Every number here was measured on the development box (RTX 5060 Laptop, 8GB VRAM,
47GB RAM, PCIe 4.0 x8) with `scripts/bench.py` and the profiles noted inline. Numbers
that were NOT measured say so.

## Current state

    decode    807 ms/token = 1.24 tok/s   (782-878 across runs)
    output    [223, 19, 17, 20] for the reference prompt, stable across runs

## Where a decode token goes

    component              ms     note
    disk reads            672     2.52 GB at 3.75 GB/s, 47.7% RAM-cache hit
    compute                62     all kernels together
    PCIe to device          ~0     experts are consumed from pinned host bytes
    host sync / rest       55

The step is disk-bound. Measured with the expert store's own counters over one decode
step:

    expert bytes read from disk   2.52 GB   (47.7% served from the RAM cache)
    bytes crossing PCIe           0.00 GB   (experts are read straight from pinned
                                             host memory by the kernel)

2.52 GB at the measured 3.75 GB/s is 672 ms of the 789 ms step. The GPUs are idle waiting
for the disk, not for the link.

A profiling run shows ~1.2 s attributed to `Memcpy HtoD` over the same step. That number
is misleading: an isolated pinned->device copy costs 3-4 us of fixed overhead plus
~28 GB/s of bandwidth (0.3 MB in 14.8 us, 18.8 MB in 695 us), so the attributions are
queued time on the copy stream while the disk read is in flight, not transfer time.

## What the link can actually do

    pinned -> device     28.5 GB/s   (hardware ceiling on this box)
    pageable -> device   12.4 GB/s
    copy_ dispatch       40-56 us per call, at any size
    _async_h2d           56 us for 1MB, then bandwidth-bound

Small transfers are NOT slower per byte: 240 separate 18.8MB copies run at 28.5 GB/s,
the same as one 4.5GB copy. Fragmentation costs launch overhead, not bandwidth. A
measured consequence: coalescing each layer's 28 tensors into 3 runs would cut DMA
count 9x, but it needs the host addresses to be contiguous, and on Windows they are not
(see below).

## Known waste: 240 expert visits become 480 DMAs

`TI_SERVE_BATCH=3` serves the 6 routed experts of a layer in two batches of three. The
fused branch checks `_expert_cache` before materialising an expert; the per-expert
branch does not, so each batch re-runs the DMA. Measured: 12 `_expert_from_rec` calls
per layer against 6 experts.

Adding the lookup measured 819ms -> 793ms, which is inside run-to-run noise. It was
NOT kept: an Expert's fp4 tuple holds views into the 4-deep `_blob_stage` ring, so a
cached entry can outlive its bytes. That is the same hazard the planner cites when it
refuses a small VRAM slot pool (12 slots made two identical greedy runs disagree).

## Platform limits found (Windows, not design flaws)

**Host addresses are scattered.** A layer's 28 tensors span 2.4 TB of virtual address
space after `cudaHostAlloc`, so `pool_base + offset == data_ptr` is false and the
zero-copy path never engages. On Linux the mapping is contiguous and the path works.

**`__constant__` arrays do not survive `cuModuleLoadData`.** The initialiser lives in a
host ELF section the cubin does not carry, so the kernel reads garbage. Measured once as
every output being ~1e36. Plain `__device__` file-scope arrays work.

**`cudaHostAlloc` above ~24GB fails.** The pinned pool caps there regardless of RAM.

## Hardware the planner targets

`python scripts/predict.py --model <dir>` prints this table from the checkpoint's real
sizes and the planner's own solver:

    card   layers  stream  slots  mtp  expertGB/tok  H2D ms   bound
       8    11/40    29       0   off          4.20     340      IO
      24    40/40     0     241   on           0.00       0  compute
      32    40/40     0     561   on           0.00       0  compute
      80    40/40     0    2480   on           0.00       0  compute
     141    40/40     0    4919   on           0.00       0  compute

A decode token touches at most topk(6) x 40 layers = 240 distinct experts. At 24GB the
slot pool covers 241, so the 296GB expert table is served without crossing PCIe at all.
vLLM needs the whole 303GB resident for the same model, i.e. eight H200s.

## Measured trade-offs, with the losing side kept

| change | result | kept? |
| --- | --- | --- |
| host cache 16GB -> 8GB (slab 0/30 -> 30/30) | 787 -> 857 ms | no, cache wins |
| tensor-core GEMM at M=1 | 4.2-6.5x slower than SIMT | no, gated at M>=128 |
| CPU-side MoE routing | 788 -> 879 ms | no, the D2H sync costs more than the matmul |
| fused gate/up GEMV, M=1 | 374 -> 249 us per expert | **yes** |
| coalesced layer transfer (Windows) | cannot engage | no, needs Linux |

## Expert routing has no strong hot spot

6000 expert visits over 20 tokens, 383 distinct experts:

    top 10% of experts   22.3% of visits
    top 20%               38.4%
    top 30%               51.6%
    top 50%               72.7%

A typical MoE puts 60-80% of visits in its top 20%; this model is unusually flat, which
is why the host cache saturates at an 87% hit rate and why prefetching buys little.
Adjacent-token route overlap (Jaccard) is 0.068 against a 0.0068 random baseline --
real but far too small to predict the next token's experts from.

## MTP (speculative decode)

Drafting `1 + 5a` tokens per main forward, at acceptance `a`, divides the expert traffic
by that factor:

    a=0.3  -45%    a=0.5  -61%    a=0.7  -69%    a=0.9  -75%

On an 8GB card the three draft stages do not fit (measured: CUDA allocator assert), and
the planner turns speculation off below 16GB. It is on by default at 24GB and above.

**The acceptance rate has not been measured here, and cannot be on this box.** A draft
takes its input from the hidden states of the last layers (see
`dspark_target_layer_ids`), so `_draft` returns an empty list unless all 40 layers ran --
running the engine with 2 or 10 layers therefore exercises the fallback, not the
speculation. Verified with 2 layers: `speculative_generate` returns the same 6/6 tokens
as plain decoding, which pins the draft/verify/truncate logic but says nothing about `a`.
The percentages above are therefore arithmetic from the `1 + 5a` structure, not a
measurement. Measuring `a` needs a card that holds the full model plus the draft stages.

## Kernel conformance

`python scripts/check_kernels.py` runs 29 cases covering the quantisers, the three GEMV
kernels, the tensor-core GEMM, sparse attention and the VRAM slot state machine. Every
native kernel that runs on the decode path has a reference comparison there.
