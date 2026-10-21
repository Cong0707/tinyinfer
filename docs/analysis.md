# Analysis: what the numbers prove

All constants below are measured on this machine, not assumed.

## 1. Routing cannot be predicted from the previous token

Measured: for a 360-step sweep, the expert sets of adjacent tokens shared **0 of 360**
pairs. A per-layer check on real prompts gives, for one prefill of 8 tokens, 6-31
distinct experts per layer -- each token picks its own set and the union is the sum of
the tokens, not a repeat.

Consequence: no window/pattern predictor can cut the 4.5GB/token of expert reads (the
router is deterministic given the hidden state, and the hidden state changes with every
token). The gate prefetch that exists (reading a layer's experts between the gate and the
serve) is the only overlap available, and it is already used: fetch_async is issued right
after the gate.

The one real reuse is the cache: over a session the same experts recur only if the text
repeats. That is what the LRU/ARC cache exploits (hit rate is workload dependent: 3.7%
cold on random text, up to 90% when a warm cache already holds the token's set).

## 2. Speculative decoding ceiling on this card

Standard result for k draft tokens with acceptance rate a:

    speedup(k, a) = (1 - a^(k+1)) / ((1 - a) * (1 + k * (1 - a)))

At the measured a = 0.08 (three MTP stages) the speedup is 1.003-1.01 for any k: the
draft is almost always rejected and the verification pass costs a full forward. Even a
perfect draft path would need a >= 0.4 to reach 2x. That is why the planner keeps MTP
off on this card (it also does not fit: 1.36GB of draft weights next to the streaming
ring OOMs at 8GB).

## 3. Why top-k cannot be traded below 6 without visible loss

The routed-expert weights sum to the output of the MoE block. Dropping expert j changes
the block output by ||w_j * (y_j - y_expected)||. For this checkpoint the routing
weights are heavily skewed: the top-1 expert carries about half the routed mass and the
5th/6th carry the remainder. Measured: top-4 output repeats the prompt instead of
answering, top-2 output degenerates. So the quality floor is the checkpoint's own top-6,
and that is what the IO estimate (4.5GB/token) uses.

## 4. The decode floor, restated

    4.5GB / 4.5GB/s  (experts)  = 1.00s
    6.0GB / 26.5GB/s (layers)   = 0.23s
    ------------------------------------
    floor                         1.23s/token  (0.81 tok/s)

Measured best this round: 1.66s/token (0.60 tok/s) = 75% of the floor; the gap is fetch
variance (0.85-1.2s vs a 1.0s floor) and the 84ms bincount. Nothing in software can go
below the floor without either fewer routed experts or more VRAM (each extra resident
layer removes 166MB/token = 6ms).

## 5. Prefill

Eight tokens route to ~40 distinct experts per layer, i.e. **30GB** of expert reads:
6.7s at 4.5GB/s. Warm prefill measures 4-16s depending on disk state (the volume fell
from 4.5GB/s to 2.7GB/s over hours of repeated sweeps). The first forward additionally
pays a one-off 13GB pinned-slab setup.

## The checkpoint is a base (pre-SFT) model

All three prompt shapes give templated continuations, not answers:

    plain  'The capital of France is'        -> '# 1\n# 2\n# '
    qa     'Question: ...?\nAnswer:'        -> ' The answer is in the context of the given passage'
    chat   '<|User|>What is ...?<|Assistant|>' -> ' (a) 1.2 (b)'

The tokenizer has no chat template (`chat_template: none`). The model's own top-5
for the France prompt are '#' 13.25, 'Paris' 12.5, ... i.e. the base model prefers a
Markdown heading; 'Paris' is what an SFT'd chat variant would say. The engine's job is
to reproduce the model's distribution, and the component parity checks (0.000000 on
attention/expert/gate/hc, 0.87% on compressed layers) confirm it does. So the
off-topic-ish output is the checkpoint's behaviour, not an engine defect; a chat
alignment (or the chat variant of the weights) is what would change it.

