# FAQ (answers measured on this machine)

## Why does the model answer with markdown headings / template text?

The checkpoint is a base (pre-SFT) model with no chat template. Its own top-1 for
'The capital of France is' is '#' (13.25) over 'Paris' (12.5); the templated
continuations are its behaviour. The engine reproduces the model's distribution
(component parity 0.000000). A chat-aligned variant would change the output.

## Why is it slower than the 4.5GB/s numbers in the docs?

The checkpoint volume measured 4.5GB/s at the start and 1.5-1.9GB/s after hours of
repeated sweeps. Decode is fetch-bound: 4.5GB of expert reads per token at 1.9GB/s is
2.4s before anything else. The IO arithmetic (docs/lower_bounds.md) is the invariant;
run `python -m tinyinfer.bench` to see the disk rate measured in the same run.

## Why 8GB/47GB and not a bigger card?

Layer streaming + expert caches are the level-2 features that make it work at all; the
planner sizes everything from what the machine reports. The design does not depend on
the card size, only the speed does.

## Why not 1 tok/s?

Decode needs 4.5GB of experts + 6.0GB of streamed layers per token. At the measured rates
that is 1.23s (0.81 tok/s) before compute; 1 tok/s would need more VRAM (each extra
resident layer saves 6ms/token) or fewer routed experts (top-4 measured visibly worse).

## Can I use the OpenAI client?

Yes: `python -m tinyinfer.serve` exposes `/v1/completions` and `/v1/chat/completions`
(verified). One request at a time, because there is one KV session.

## What is not perfect yet?

docs/STATUS.md lists the open items: session logit drift, the batch-vs-row MoE
difference (does not affect decode), MTP verify, and the compressed-layer 0.87% residual.
