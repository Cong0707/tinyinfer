# Lower bounds for this machine (measured constants, arithmetic only)

Measured on the box: pinned H2D 26.5GB/s, scattered expert reads 4.5GB/s, expert blob
18.8MB (fp4, incompressible), streamed layer 166MB after the RoPE table became a
shared constant (fp8 payloads), resident layers 4-5, decode prefetch hit rate ~35%.

## Decode, one token

Every token moves, at minimum:

- routed experts: 6 experts x 40 layers x 18.8MB = **4.5GB** from disk
  (checked: the fp4 blob is the checkpoint's own storage; routing is a property of the
  token, so the set cannot be predicted from the previous token -- measured overlap of
  adjacent-token expert sets is 0.000, 0/360);
- streamed layers: 36 layers x 166MB = **6.0GB** over PCIe to a resident set the card
  cannot hold (the 8GB card holds 4-5 of 40 layers).

Time floor: 4.5GB / 4.5GB/s + 6.0GB / 26.5GB/s = **1.00s + 0.23s = 1.23s/token**,
i.e. **0.81 tok/s** before any compute. Add ~0.2s of attention/MoE compute and
synchronisation and the practical ceiling is ~1.4s/token (0.7 tok/s).

Measured now: **1.66s/token (0.60 tok/s)**, i.e. 75% of that ceiling. The remaining
0.4s is fetch variance (0.85-1.2s against a 1.0s floor) and the 84ms bincount.

Consequences:

- **1 tok/s is not reachable on this machine without dropping routed experts** (quality
  loss) or adding VRAM (each extra resident layer removes 166MB/token = 6ms).
- The only way to lower the 1.00s expert term is to read fewer experts. Top-4 measured
  1.36s/token with visibly degraded output, top-2 0.81s/token and unusable.

## Prefill, 8 tokens

- routed experts: 8 tokens x 6 x 40 layers with ~40 unique experts per layer (the eight
  tokens route to different experts) = **30GB** from disk -> **6.7s** at 4.5GB/s;
- streamed layers: 6.0GB -> 0.23s.

Floor **~7s**. Measured (warm, no setup): 16s total, of which expert fetch 8.8s and the
rest attention/MoE/synchronisation. The gap is the per-layer indexer/compressor launches
(~30ms/layer at sequence length 8), not IO.

Note: the **first** forward additionally pays the pinned-slab setup (36 x 360MB
cudaHostAlloc = ~13s on Windows). That is a process-start cost, not a per-prompt one.

## What this means for the roadmap

1. The decode IO path is within 25% of its floor; the remaining gains are in the
   non-IO part (bincount 84ms, serve 130-180ms, attention).
2. Prefill is IO-bound at 8.8s of expert reads; the 30GB is real (eight tokens routing
   to ~40 experts per layer), so prefill speed here means fewer unique experts or a
   smaller checkpoint, not a better loop.
