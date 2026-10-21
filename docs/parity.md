# Numerical parity with the model author's implementation

Authority order (user-defined):

1. **DeepSeek** -- the released `config.json` and DeepSeek's own serving stack
   (`DeepInfer`, including `model_executor/models/v4_modules/*`) is the highest
   authority for semantics. Levels 0-2 must match it.
2. NVIDIA (Flare / b12x / vLLM) is an **optimisation reference only** -- useful for
   structure and speed, never for numerical semantics.

## Results

Reference: `reference/official/model.py`, adjusted where DeepInfer differs from it
(see `mscale` below). Same layer, same checkpoint tensors, same bf16 activation.

| component | max |rel| |
| --- | --- |
| Layer-1 attention output (full forward) | **0.000000** |
| fp4 routed expert (w1/w2/w3 + SwiGLU + routing weights) | **0.000000** |
| Gate (weights + top-k indices) | **0.000000** |
| hc_split_sinkhorn / hc_mixes / hc_pre / hc_post | **0.000000** |
| RoPE cache vs DeepInfer `YaRNScalingRotaryEmbedding` | 9.7e-06 |
| Compressed-KV attention (layers 2, 8) | 0.0087 / 0.0142 |

## Deviations that had to be fixed

1. **sparse_attn kernel** -- ported the official `kernel.py::sparse_attn`; our own
   TileLang kernel is a fallback (4.7% off before).
2. **Window KV quantisation** -- fp8 round-trip **after** RoPE (skipping it: 4.7%.
   Doing it before RoPE: also wrong).
3. **o quantisation** -- fp8 round-trip before the grouped `wo_a` einsum (4.8%).
4. **Compressed latent** -- rotate at the group position and quantise to fp4
   (block 16, E4M3 scales) before storing in the shared cache (43% otherwise).
5. **Indexer input** -- the reference computes index-K from the **pre-RoPE** latent
   (`KProject` runs before rotation). We rotated in place, so the indexer saw
   rotated keys.
6. **Compressor normalisation order** -- `norm(kv.to(bf16))`: round first, then
   normalise (0.5%, amplified to 17.8% by the fp4 quantiser).
7. **RoPE parameters from `rope_scaling`** -- the reference `Attention.__init__` reads
   flat `rope_factor` / `original_seq_len`; the released config expresses the same as
   the nested `rope_scaling` (yarn, factor 16, original_max 65536). The engine reads
   the config.
8. **YaRN magnitude scaling (`mscale`)** -- DeepInfer's `YaRNScalingRotaryEmbedding`
   multiplies the cos/sin cache by `yarn_get_mscale(factor) = 0.1*ln(factor)+1`
   (= 1.2773 at factor 16). `model.py` omits it; DeepInfer is the authority, so the
   engine applies it. `TI_YARN_MSCALE=0` restores the model.py behaviour for
   comparison.

## Remaining

Compressed layers are within 0.9-1.4%. Their intermediates match to ~0.3% (query
projection, window KV) and the latent matches exactly, so the residual is the
amplification of that 0.3% through the indexer's top-k / softmax path rather than a
missing mechanism. Everything else in the layer graph is exact.

## Optimisation references (not semantics)

CUDA graph capture, the prepared capacity catalog, the async expert fetch ordering,
the contiguous slot arena, prefix checkpoints and the tensor-parallel sharding shape
come from NVIDIA Flare / b12x and vLLM. They are engine-level execution choices and do
not change numbers; the parity table above is measured with them enabled.