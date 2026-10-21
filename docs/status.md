# Where the engine stands (measured, 2026-10-21)

## Decode

| configuration | tok/s | note |
|---|---|---|
| best warm run, 6 resident layers | **0.56** (1.78-1.84s) | TI_FLAT triple buffer, 8GB pinned weights, 4GB pinned expert cache |
| same, first steps after load | 0.18-0.33 | expert cache cold, expert reads dominate |
| 2 resident layers + MTP weights | 0.28 | MTP needs 1.5GB VRAM, which costs 4 resident layers |
| speculative decode (accept 25%) | 0.13 | negative: draft 0.86s + verify 2.6s for ~2 tokens |

Stage breakdown at 0.56 tok/s (per token, 6 resident layers):
```
attn_pre 0.10s   attn 0.08s (submitted)   hc1 0.00s
ffn_pre  0.10s   moe  1.37s               hc2 0.00s
```
The MoE segment is dominated by one host sync per layer (bincount over the routing).
The CPU waits for the GPU to finish that layer's attention, then submits the fp4 GEMV,
so GPU and CPU run serially. That is the structural reason the card sits near 50%
utilisation.

## Prefill

chunk=8: 165s / 21 tokens (safe, peak 7.7GB). chunk=16: 63s (peak 9.5GB, brief
paging). chunk=32: 64s (peak 10.9GB, decodes slower afterwards). Default stays 8.

## IO

Disk reads are at the device ceiling: 18.8MB blobs measure 7.3GB/s at QD=8 (the
device saturates at eight outstanding 9.4MB requests; QD=32 is slower). Contiguous
18.8MB blocks measure no faster than the real scattered 1.1MB+17.7MB pairs (6.85 vs
7.16GB/s), so re-laying-out the checkpoint would buy nothing.

Layer H2D (15GB/token at 34 streamed layers) is fully hidden: 7-23ms of wait per
token with the triple buffer.

## Open items, in order of expected gain

1. **The per-layer routing sync.** Every layer must tell the host which experts to
   read, which is a hard sync. Only a batch (or an all-GPU expert set) removes it.
2. **Speculative decode.** Draft top-1 now matches the main argmax 60-75% of the
   time, but accept is 25% at draft_len=4. Official DFlash reports 60-80%. With 2
   resident layers the economics do not close.
3. **MTP with 6 resident layers** needs 1.5GB of VRAM that is not there. Dropping the
   third flat buffer or the expert slots would free it, at a measured cost of
   +0.5-0.8s/token.
4. **Official kernels.** DeepGEMM targets SM90/SM100 and is batch-shaped (one CTA per
   token, split-K across SMs); at T=1 its fused mHC collapses to a single CTA and
   loses to the torch path (0.04ms). The reference/official tilelang kernels we
   already call are the official ones for this shape.
