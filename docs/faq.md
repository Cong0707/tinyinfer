# FAQ (answers measured on this machine)

## Why does the model answer with markdown headings / template text?

The checkpoint is a base (pre-SFT) model with no chat template. Its own top-1 for
'The capital of France is' is '#' (13.25) over 'Paris' (12.5); the templated
continuations are its behaviour. The engine reproduces the model's distribution
(component parity 0.000000). A chat-aligned variant would change the output.

Prompt shapes matter more than with a chat model (measured):

    'The capital of France is'                 -> '# 1. 使用 Python 的 `'
    'Question: What is the capital of France?\nAnswer:' -> ' France is a country in Europe. ...'
    '<|User|>What is ...?<|Assistant|>'        -> ' 1.1.1.1.1'

The question/answer framing is the one that yields a sensible continuation; the
chat-marker framing does not, because the base model never saw it as an instruction.
Chinese question prompts instead land in its exercise-with-answer-key distribution
('问题：...\n答案：' -> 'A\n\n解析：...'); English prompts answer directly.

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

difference (does not affect decode), MTP verify, and the compressed-layer 0.87% residual.
difference (does not affect decode), MTP verify, and the compressed-layer 0.87% residual.
## Do Chinese prompts work?

Not as answers: every Chinese shape tested lands in the checkpoint's exercise/key
distribution ('问题：...\n答案：' -> 'A\n\n解析：...', '请用一句话回答：...' -> 'A 点\n\n解析：由题意').
English question prompts answer correctly ('Paris', '3.14'). The engine reproduces the
checkpoint; Chinese answers need a chat-aligned checkpoint.


## Does a multi-turn chat work?

The checkpoint has no chat template, but a flat transcript in the question/answer shape
continues sensibly (measured):

    Question: What is the capital of France?
    Answer: Paris
    Question: What is the population of that city?
    Answer: -> ' 2,000,000'

so multi-turn use works by appending turns to one transcript and calling `/v1/completions`.


### How capable is the base model?

Measured: it answers some factual questions ('Paris', '3.14', Paris population) but fails
simple arithmetic and trivia ('What is 2+2?' -> ' 2.', 'largest planet?' -> ' 8'). That is
the checkpoint's capability, not the engine's: the engine reproduces the model's
distribution bit-comparably (docs/parity.md). A chat/SFT variant is what makes it reliable
at instruction-style questions.


Creative prompts likewise follow a math-answer template ('Write a haiku about the moon'
-> ' The answer is 1, but the obvious proof is not much at'), consistent with the
capability note above.


- Factual questions are inconsistent rather than uniformly right: 'capital of Japan'
  produced a sentence about Japan's monarchy, 'who wrote Romeo and Juliet' fell into the
  multiple-choice template. Same base-model capability, same conclusion.

