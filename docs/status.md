# Status of tinyinfer (measured, not projected)

## What runs

| item | state |
| --- | --- |
| Load 510GB checkpoint directly from HF safetensors | works (build ~2s + slab setup) |
| Prefill / decode / streaming | works |
| Multi-session (generate twice on one engine) | works (logit drift is a known open item) |
| HTTP server `/health` + `/generate` | verified end to end |
| Benchmark / examples | `python -m tinyinfer.bench`, `examples/*.py` |
| Test suite | passes (2 xfail: session drift, one-shot vs stepwise) |

## Measured performance

| metric | value | note |
| --- | --- | --- |
| decode (this box, disk 4.5GB/s) | 1.66s/token = 0.60 tok/s | quality-preserving top-6 |
| decode (this box, disk 1.5-1.9GB/s after long sweeps) | 2.5-4.5s/token | fetch-bound |
| IO floor | 1.23s/token = 0.81 tok/s | arithmetic in docs/lower_bounds.md |
| prefill, 8 tokens, warm | 20s at 1.9GB/s disk (4s at 4.5GB/s) | fetch-bound |
| prefill, 63 tokens | 136s (chunk=1 path, correct) | chunk>5 has a numeric bug |
| process start | ~13s (13GB pinned slab) | one-off |

The disk volume measured 4.5GB/s at the start of this work and 1.5-3.6GB/s after
hours of repeated checkpoint sweeps, so absolute tok/s tracks the current disk rate,
not code changes. The IO arithmetic in docs/lower_bounds.md is the invariant.

## Numerics (authority: DeepInfer, the model author's stack)

| component | max relative difference |
| --- | --- |
| window-layer attention | 0.000000 |
| fp4 expert FFN | 0.000000 |
| gate (weights + indices) | 0.000000 |
| hyper-connection ops | 0.000000 |
| RoPE vs DeepInfer | 9.7e-06 |
| compressed layers | 0.87% / 1.42% |
| fused rope+quant kernel | 0.000000 (bit-exact) |

## Known open items

1. **One-shot T>1 forward vs token-by-token** differs (a synthetic MoE batch of 8 rows
   vs eight single-row calls: ~26% on layer 0). Routing, activation quantisation and the
   per-expert kernels each verify exact in isolation, so the accumulation is suspect.
   `generate()` only reads the last position of a causal prefill, so answers are
   unaffected; the speculative verify pass and chunked prefill are.
2. **Session drift**: a second prefill in the same session moves logits by >1. Some
   state survives reset; the caches checked so far are not it.
3. **MTP** loads and seeds, acceptance 0.12, but output degenerates because item 1
   breaks its verify pass.
4. **Compressed-layer parity** (0.87%) is the largest numeric residual and the leading
   candidate for item 1.

## What the output looks like

The checkpoint is a **base (pre-SFT) model** with no chat template: its own top-1 for
'The capital of France is' is '#' (13.25) over 'Paris' (12.5), so templated
continuations are its behaviour, not an engine defect (docs/analysis.md).