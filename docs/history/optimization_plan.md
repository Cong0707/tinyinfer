# Toward 1 tok/s on an 8GB card (optimization plan, 2026-10-21)

Baseline measured after the full 510GB model landed at E:\Models:
**0.30-0.33 tok/s** (40 layers, 11 resident + 29 streamed, engram L1+L14 on).

## Where the 3.1s/token goes (torch profiler, 1 decode step, 40 layers)

| item | time | share | note |
|---|---|---|---|
| H2D Memcpy (pageable) | 590ms | 54.8% | 763 copies x 781us; 29 streamed layers x 190MB at 8.3GB/s |
| fp8 GEMM (M=1) | 195ms | 18.2% | 221 calls x 884us; block_M=32 wastes 31/32 of the tile |
| tilelang sparse_attn | 134ms | 12.4% | 32 calls x 4.18ms; M=1 grid is inefficient |
| HC bmm (fp32 einsum) | 91ms | 8.5% | 48 calls x 1.90ms |
| fp8 GEMV (bf16 x fp8) | 91ms | 8.5% | 48 calls x 2.86ms |
| CPU aten::copy_ side | ~406ms | - | per-tensor staging/contiguous work before H2D |

We are not I/O-bound at steady state (warm expert cache); we are **transfer- and
small-kernel-bound**.

## Levers, ordered by expected win

1. **Overlap H2D with compute (up to -590ms).** The single biggest win. Reference:
   ds4's streaming path issues pool preads for the next layer while the GPU still runs
   the current layer, plus cudaMemcpyAsync on an upload stream. We have the v4
   streaming path for compressed layers and the (disabled) v5 double buffer; the 8GB
   card could not hold two full layer buffers (v5 peaked at 9330MB and paged). Options:
   - single 190MB staging buffer + torch.cuda.Stream upload for the *next* layer's
     tensors while the current layer computes (one in-flight layer, no second full copy)
   - or pinned per-tensor buffers limited to ~1GB total (still ~3x pageable bandwidth)
2. **M=1 GEMV for fp8 (-100..150ms).** gemv_fp8.py already contains fused_fp8_gemv
   (173-184) but fp8_linear imported a name that does not exist; wiring it crashed the
   CUDA kernel (illegal access in gemv_bf16_kernel at N=1280/K=5120) - needs a debug
   pass on the rows=16 grid. Also fixes the same waste in shared-expert GEMMs.
3. **tilelang sparse_attn M=1 (-80..100ms).** 4.18ms/call is far above the ~0.3ms the
   FLOP count implies; the kernel is configured for batched prefill. Reference: vllm.cpp
   uses a separate decode kernel with one block per (query, head-chunk).
4. **HC bmm and small ops (-60ms).** 48 x 1.9ms for a 164KB problem is launch overhead
   (measured: CPU does it in 0.22ms, GPU 3.2ms for the same math). Batch the 40 layers'
   HC post/comb into one call, or move the small ones to the CPU (measured slower here
   only because of per-call H2D, so batching is the answer).
5. **Prefill/decode asymmetry**: prefill M=8..64 already works (71-290s first time due
   to JIT, then cached). Long context decode is sparse by design.

## Realistic target

At 8GB the card holds 11 layers; 29 streamed layers x 190MB = 5.5GB/token of H2D.
If (1) fully overlaps H2D (590->~100ms residual), (2) saves ~150ms of GEMM, (3)
saves ~100ms of attention, (4) saves ~60ms, the step drops to roughly **1.6-1.8s
(0.55-0.62 tok/s)**. Reaching 1.0 tok/s needs the H2D to be both overlapped AND
smaller: more resident layers (fp4 attention / smaller KV would free ~1GB -> +2
layers) or a 16GB+ card. Honest statement: **1 tok/s is not reachable on this 8GB
part with the current exact-weight streaming design**; 0.55-0.65 is, and 1+ becomes
reachable on 16GB.

## Measured I/O findings (2026-10-21, the "why is the disk idle" round)

These are hardware facts on this box (RTX 5060 laptop 8GB, 47GB RAM, NVMe rated
11GB/s sequential / 1.3GB/s 4K). They change what is worth optimising.

1. **Windows cached reads serialise badly under threads.** Reading 18.8MB expert
   blobs through `os`/buffered IO with 6 threads measured **0.26GB/s**; the same
   reads with `FILE_FLAG_NO_BUFFERING` (`ShardReader(direct=True)`) measured
   **6.3GB/s** at 6 threads and 4.6-6.3GB/s in-engine at 12 threads. The expert
   stream now takes the direct path (`ExpertStore.from_store(direct=True)`).
   Step time went 4.3s -> 2.45s from this one change. Any future work on the read
   path (IoRing / BypassIO / DirectStorage) builds on `_WindowsDirect`.
2. **NO_BUFFERING needs sector-aligned, in-file ranges.** Rounding the span up to
   4KiB overran EOF on the last tensor of a shard (short read + ERROR_INVALID_
   PARAMETER). The reader now clamps to the aligned file size and falls back to a
   cached read for that final partial page.
3. **Keeping the RAM cache pinned is a trap.** Admitting the pinned read buffers
   meant ~630 live `cudaHostAlloc` blocks (11GB); every fetch then allocated fresh
   pinned memory for 18.8MB and the pinned pool became the bottleneck. The cache now
   stores a pageable copy (`d.numpy().copy()`), pinned memory is reserved for the
   layer weights.
4. **Expert routing has almost no token-to-token locality.** Measured cache hit
   rate 9.6-18.5% at 600-630 entries (11GB), i.e. the predicted prefetch is wrong
   ~85% of the time. Enabling the predictor prefetch measured *slower* (0.36 vs
   0.40 tok/s) because it doubles disk reads for the misses. `TI_PREFETCH=1` is
   opt-in and off by default; a speculative/MTP decode is the only way to amortise
   the 3.7GB/token that routing forces us to read.
5. **Mixing allocation sizes blows the 8GB card.** `Block.to()` copies 442MB and
   719MB layers; the caching allocator cannot reuse one for the other, so reserved
   memory climbed to 15.7GB (past the 8.1GB physical card -> WDDM paging). Two
   fixes, both kept: (a) `resources.plan()` keeps the stock per-tensor path;
   (b) `TI_FLAT=1` gives every streamed layer one shared 719MB device buffer pair
   (`_flat_setup/_flat_apply/_flat_prefetch`), zero extra host memory because it
   reuses the pinned refs `Block` already holds -> peak 6973MB, 0.40 tok/s.
6. **Steady state on this box**: 2.45s/token = 0.41 tok/s (stock path) with
   3.7GB/token of expert reads at 4.6GB/s, 15.5GB/token of layer H2D. Disk is at
   ~42% of its sequential rating; the next step is IoRing/BypassIO, not more
   threads. The 8GB card is the binding constraint: every extra resident layer
   removes 442MB of H2D per token.

## Pipeline round: what actually overlaps now (2026-10-21, second half)

Instrumented the decode loop with two timers — time spent waiting for the layer H2D
event (`Engine._h2d_wait_ms`) and time spent waiting for expert disk reads
(`MoE._disk_wait_ms`) — and built a three-stage pipeline:

1. **Triple-buffered layer streaming** (`TI_FLAT=1`): three rotating 719MB device
   buffers; layer k computes while k+1 and k+2 copy in on a side stream. Two buffers
   were not enough: a 719MB H2D takes 27ms and a layer only 15ms, so with two the
   transfer can never hide.
2. **Six-way expert fetch** (`MoE.forward`): one `fetch_async` per expert, consumed
   in two batches of three. The GPU starts on the first three while the last three
   are still on disk.
3. **Zero-residency option** (`TI_GPU_LAYERS=0`): every layer streams through the
   flat pair. Measured: reserved 4198MB (vs 10426MB with per-tensor layer copies),
   no paging at all.

Measured wait times at 1.54s/token (0.65 tok/s, best run):

| stage | per step | note |
|---|---|---|
| layer H2D wait | **9-13ms** | 15GB/token is now hidden by the triple buffer |
| expert disk wait | **1.0-1.5s** | the remaining bottleneck |
| GPU compute | ~0.6s | |
| total | 1.5-2.0s | varies with page-cache state |

So the PCIe side is solved: 15GB of weights arrive without the compute thread ever
blocking. Everything left is disk: 3.8GB of expert blobs per token at an effective
~4.6GB/s, and the variance between 1.5s and 3.3s runs tracks the OS page cache.

Next, in order of leverage:
* **IoRing / deeper async queue** — one 18.8MB read measures 9.4GB/s alone, six
  concurrent threads do not add up to that, and 8-12 threads already plateaued at
  6.3GB/s. The gap is queue depth, not the device.
* **Resident-layer budget** — VRAM sits at 7856/8151MB; every layer moved to
  residency removes 442MB of H2D, but H2D is already free, so this only helps by
  shrinking the *host* traffic, not the step.
* **Gemv for fp8 at M=1** — 195ms of the 600ms GPU time is GEMM doing one row.
* **MTP / speculative decode** — the only way to amortise the 3.8GB/token that
  top-6 routing across 384 experts forces us to read.

