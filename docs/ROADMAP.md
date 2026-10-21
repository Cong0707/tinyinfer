# Roadmap (ordered by measured value, not ambition)

## 1. Close the batch-vs-row MoE difference

Blocker for MTP and for chunk>1 prefill. Evidence so far: identical rows are exact,
distinct rows differ by ~26-70% at layer 0; attention, gate, activation quantisation,
single-row GEMV and the batch GEMV each verify 0.000000 in isolation, and the VRAM-slot
path is identical to the RAM-record path. Ruled out: kernels, quantisation, gate, shared
expert, slot pressure. Remaining suspects: the serve loop's 3-expert batching
(`_val_acc`/`_idx_acc` concatenation order feeding one `index_add`) and record lifetime.
Next probe: one expert per serve batch and compare.

## 2. Compressed-layer parity 0.87%

The largest numeric residual against DeepInfer and the leading suspect for (1) and the
session drift. Port DeepInfer's `fused_transform.py` norm leg as well, or compare the
compressor's pooling order against the reference in float64.

## 3. MTP verify path

Depends on (1). Once the batch path is exact the verify pass can rely on a T=5 forward
and MTP becomes real (acceptance currently 0.12).

## 4. Prefill speed

Long prompts currently use chunk=1 because chunk>5 has the (1) bug. Layer-major prefill
(one H2D per layer) has a separate boundary bug (last chunk size). Both unblock the same
way: fix the batch path.

## 5. Hardware headroom

Every extra resident layer removes 6ms/token; a 16GB card would put the 1.23s IO floor
within reach and could hold the MTP stages (1.36GB). The code paths are already sized by
the planner, so this is a matter of VRAM, not new code.

## Not worth doing (measured)

- grouped 6-expert GEMV incl. plane gather: 0.76x
- multi-row GEMV on real weights: not exact -> default off
- expert prediction from the previous token: 0% adjacent overlap
- top-k below 6: visibly degraded output
- KV fp8: KV is ~10MB total, no VRAM to gain