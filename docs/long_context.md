# Long-context prefill (hundreds of K tokens) — design

Target: accept a prompt of 128K-512K tokens and prefill it in reasonable time on this
laptop (8GB VRAM, 47GB RAM, 510GB weights on NVMe).

## Current path and why it does not scale

`Engine.forward(ids)` splits the prompt into `prefill_chunk` pieces (plan gives 8-21)
and runs all 40 layers per chunk:

    for chunk in prompt:          # N/8 chunks
        for layer in 40:          # streaming layer -> per-token H2D/expert reads
            forward(chunk)

Measured: prefill 32 = 180s, prefill 64 = 314s (chunk 8, vram 7227MB). Extrapolating
to 512K tokens gives ~25 hours: the per-chunk cost is dominated by re-streaming the
same 40 layers and re-reading the same hot experts once per chunk, so the work grows
linearly with the number of chunks instead of with the bytes of new tokens.

## Proposed layer-major prefill

Classic "layer-major" execution, which is what vLLM/DeepInfer do for long prompts:

    resident_layers = 9 (per the hardware plan)
    for layer in 0..39:
        load layer once (streamed layers: one H2D + one expert pass for the whole prompt)
        for chunk in prompt:      # the chunk loop becomes inner
            apply layer to chunk     # activation written to the layer's output buffer
        swap(activation_in, activation_out)
    # KV caches for all layers updated in one pass; decode then starts from the tail

Why this is much faster here:
- the 31 streamed layers are moved H2D **once per prompt** instead of once per 8-token
  chunk (H2D drops from O(chunks) to O(layers));
- expert reads happen **once per layer** over the whole prompt instead of once per
  chunk; the expert cache learns the prompt's routing in one sweep;
- the only memory that grows with prompt length is the activation buffer
  (T x 5120 x 2 bytes = 10MB per 1K tokens) plus the KV caches, both of which fit in
  host RAM for 128K-512K tokens (compressed KV is ~64MB/1K tokens at ratio 2).

## Memory budget (per prompt length)

| item | 128K tokens | 512K tokens |
|---|---|---|
| activation buffer (T x dim, bf16) | 1.25 GB | 5.0 GB |
| compressed KV (kv_source layers, ratio 2) | ~8 GB | ~32 GB |
| window KV (128 x 4 layers, fp8) | ~1 GB | ~1 GB |

At 512K the compressed KV alone is ~32GB, so the design must also stand up the KV
quantization tier (kv_bits=8 in the AGGRESSIVE tier halves it) and stream KV pages
like experts. 128K is comfortable with the current RAM budget.

## Chunk budget (measured on this 8GB card)

256-token prefill with different max_chunk:

| T | time | throughput | VRAM |
|---|---|---|---|
| 32 | 316.3s | 0.81 tok/s | 9804MB |
| **64** | **220.8s** | **1.16 tok/s** | 11043MB |
| 128 | 516.9s | 0.50 tok/s | 11963MB (pages badly) |

T=64 is the sweet spot: the larger T amortizes the per-layer H2D better, but beyond
64 the activation growth pushes VRAM into WDDM paging and throughput collapses. The
same rule sets the long-context ceiling on a bigger card: pick T so that VRAM stays
under the physical limit, ~4x the resident-layer footprint.

## Status

- [x] per-kv-source handoff slots: SharedAttentionRuntime instances are now handed to
      Compressor/Indexer per layer (Engine._rt_for picks the nearest kv_source). The
      chunk-major path keeps the process singleton; layer-major uses one instance per
      source.
- [x] per-chunk candidate blocks: layer-major interleaves chunks across layers, so a
      single candidate slot went stale ("tensor a(32) vs b(16)"). Candidates are now
      stored per chunk (candidates_by_chunk) and consumed by the matching chunk index.
- [x] **layer-major prefill works and is faster**: 32 tok 99.5s (0.32 tok/s), 64 tok
      162.2s (0.39 tok/s) vs chunk-major 180s/314s -> **1.8-1.9x**, and the advantage
      grows with prompt length (per-layer H2D is amortized over the whole prompt).
- [ ] 128K prefill measurement (running; projected ~10-15 min at 0.3-0.4 tok/s,
      vs hours for chunk-major)
- [ ] prefill T selection from resources.plan (T=32 works in 7.9GB)
