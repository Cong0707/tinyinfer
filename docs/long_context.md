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

## Chunk budget TBD

Layer-major prefill still chunks the token dimension (T=4096 is 40MB of activation,
well inside RAM); the difference is the loop order, not the chunk size. Once the
loop is swapped we can pick T by the RAM budget (e.g. 8192 -> 80MB activation).

## Status

- [ ] loop-order swap in Engine._forward_chunk (per-layer iteration with an activation
      ring), keeping the existing KV cache machinery per layer
- [ ] prefill T selection from resources.plan (RAM budget rather than VRAM)
- [ ] 128K prefill measurement (target: minutes, not hours)
