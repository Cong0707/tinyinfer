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


## 6. Multi-row expert batches (open)

With eight identical rows a batched MoE call matches eight single-row calls exactly; with
distinct rows it does not (layer 0, up to ~70% of the output range). Each component was
then checked in isolation on real weights: layer-0 attention 0.000000, gate weights and
indices identical, activation quantisation bit-identical, `gemv_fp4_m8` vs `gemv_fp4_pq`
0.000000 on expert 34's real w1/w2/w3. The difference therefore lives in how rows share
state inside the batch (slot/record reuse in the serve loop), not in arithmetic. The
single-row path is what decode uses, which is why answers are unaffected, and it is the
default; the batch kernel is opt-in (`TI_MULTIROW_GEMV=1`).


## 7. Per-row diff is uniform (new data)

On an 8-row batch every row differs from its single-row reference by 1.3-1.9 (row 5,
whose experts are least shared, by 0.54), including rows 0/1/2/7 that route to *no*
shared expert. So the batch error is not caused by multi-row experts; it is uniform.
The batch routes ~40 experts while the VRAM slot pool holds 16; the extra records fall
back to RAM records (`acquire` returns -1 and `rec.tensor` then reads the RAM buffer),
which is a different code path from the slot view. That path difference is the leading
suspect now. Decode (T=1) routes 6 experts, always fits the pool, and is exact.


## 8. Why the window (128) and ratio (2) are not the bottleneck

Both are model constants (config.json), not engine choices, so this is a sanity bound
rather than a tuning knob.

- Sliding-window attention keeps the last `w` positions. For a decaying attention profile
  the mass outside the window is O(exp(-w/g)) for an effective span g; at w=128 and the
  measured attention profiles of this model the dropped mass is far below the fp8
  quantisation step, which is why the window-layer parity against DeepInfer is 0.000000
  even though both compute the same truncation.
- Compression pools `r` token latents into one; the pooled row is a gated average, so the
  reconstruction error is O(1/r) in the same norm the model uses for its residual. With
  r=2 the residual is well under the fp8/fp4 storage step of the compressed cache, again
  consistent with the 0.87% residual measured against DeepInfer being the largest single
  numeric gap in the model rather than an engine defect.

Both statements are bounds, not measurements; the measured part is the parity in
docs/parity.md, and the open item is to close that 0.87% (docs/ROADMAP.md, item 2).


- Ruled out: the VRAM-slot path vs the RAM-record path are identical (rel 0.000000 with
  `store.slots = None`), so the batch difference is not slot pressure.


- Row-uniformity was confirmed again (rows with no shared expert differ by 1.6 as well),
  so the batch error is not a multi-row-expert artefact. The VRAM-slot vs RAM-record path
  was ruled out (0.000000 with `store.slots = None`).


- Serve batch size is now `TI_SERVE_BATCH` (default 3). Setting it to 1 changed the batch
  difference from 26% to 36% -- i.e. accumulation order matters far more than fp32
  rounding should allow, which is itself evidence that some rows are being accumulated
  with the wrong value rather than in a different order.


- Batch size changes the answer (3 -> 26%, 1 -> 36%), far more than fp32 ordering can
  explain, so some rows are accumulated with a wrong value rather than in a different
  order. Next: log the (row, expert) pairs fed to index_add per batch.


### Chinese prompts

Question-shaped Chinese prompts land in the model's *exercise-with-answer-key* distribution
rather than answering:

    问题：法国的首都是哪里？\n答案： -> 'A\n\n解析：解析：A项，由“甲'
    问题：1+1等于几？\n答案：     -> ' 1\nD B C A 4 3'

English question prompts work ('Paris', '3.14'), so this is the base model's Chinese
training distribution, not an engine difference. Inference is correct either way; only the
continuation style differs.


- Chinese prompt variants were tried too; all land in the exercise/key distribution:
  '请回答：...' -> ' 我猜你大概会用到包括住家', '请用一句话回答：...' -> 'A 点\n\n解析：由题意'. The
  engine is computing the checkpoint's distribution correctly; producing Chinese answers
  needs a chat-aligned checkpoint, not a prompt trick.


- Follow-up measurement: `gemv_fp4_m8` vs `gemv_fp4_pq` on **real** expert-34 weights is
  rel 0.000000 for w1 (2304x5120), w2 (5120x2304) and w3 (2304x5120). So the multi-row
  kernel is exact after the flat-grid fix; the remaining one-shot-vs-stepwise MoE
  difference (22.9% on that synthetic batch) therefore does not come from the GEMV
  kernels, and the default stays row-at-a-time until the actual source is found.


## 9. Where the chunk-vs-token difference actually lives

`_forward_impl` with chunk=8 vs eight single-token calls on the same prompt differs by
rel 0.42 (per position 1.3 -> 11.2). The expert FFN is not the source: with the row-index
fix (`_aq[idx]` instead of `_aq[idx[0]]`, which fed only the first routed row through the
kernel) the expert module is exact for multi-row input, and the MoE batch difference that
remains is the compressed-layer path: `compress_len = end_pos // ratio` is computed once
per chunk, so position 0 of a chunk can see compressed groups that belong to later
positions, and the indexer's visibility mask is built for the chunk's end position.

That is why chunked prefill is still restricted to chunk=1: it is not a kernel problem,
it is the compressed-attention semantics at s>1 (docs/ROADMAP.md item 4).


- Layer-major prefill no longer crashes on the variable compressed-slot width (it is
  right-padded to `index_topk` with -1, the kernel's empty marker), but it still does not
  match chunk-major (rel 1.17), so it stays off. The padding itself is a strict
  improvement: the sparse kernel now always sees one width.


### Layer-by-layer: the 42% is accumulated, not a single bug

Comparing the chunked forward against token-by-token with per-layer activation dumps:

- layer 0's block on its own differs by **0.6%** (per-position 0.0-0.06),
- the difference grows monotonically with depth (layer 29: 33%, layer 39: 53%),
- so 40 layers of ~0.6% compound into the 42% seen in the logits.

The single-layer 0.6% comes from the MoE batch path (the same `moe.forward` that differs
by up to 26% on an artificial 8-row batch; on real activations far fewer experts are
shared so the per-layer error is small). Every MoE *component* verifies 0 against the
reference -- gate, activation quantisation, single-row GEMV, multi-row GEMV, shared
expert -- so the residual is in how the per-expert values are combined for several rows
at once, which is also where the batch-size sensitivity pointed.

Consequence: chunk=1 stays the correct prefill path. A chunked prefill needs that 0.6%
to be 0 first, because the error compounds with depth.


- Nailed down further: on 8 tokens x 4 hc (32 rows) the gate indices are identical and
  its weights differ by <=6e-8 versus the same rows run alone, and `quantize_act` is
  bit-identical. The remaining per-layer 0.6% is therefore in the scatter/combination
  (`index_add` of several experts at once versus one at a time), which is far larger than
  fp32 association can explain; it stays the open lead for chunked prefill.


## 10. CORRECTION: the engine is not bit-reproducible with VRAM slots on

Running the exact same call sequence twice gives different logits:

| configuration | two identical runs differ by |
| --- | --- |
| tier 2 (VRAM slot pool) | 0.63-0.78 relative |
| slot pool disabled (`store.slots = None`) | **0.000000** |

So the slot pool has a read/write race: a slot can be re-targeted at an expert while the
GEMV that reads it is still queued on the compute stream. Four fixes were tried and all
failed to remove it (device-wide sync before serve, recording a compute-stream event at
release, keeping that event across eviction, blocking instead of async H2D copies); one
of them (the sync) also cost 35% of decode, so all of them were reverted.

**This invalidates the earlier "batch differs from token-by-token by 26-42%" measurements**
in this file: those comparisons were run with the slot pool on and are dominated by this
noise, not by a routing/matmul difference. The same applies to the layer-0 0.6% and the
chunk 42% figures. With slots off, chunk=1 and token-by-token agree exactly, which is the
honest current statement. Decode output quality is unaffected in practice (top-1 stays
stable across runs), but the logits are not reproducible.

Open item for the roadmap: find and fix the slot race, then re-run the batch/prefix
comparisons (they may well be exact, which would unlock chunked prefill).


### The session-prefill drift: what was tried and what is left

A second prefill of the same prompt in one session drifts by ~20 logits (`forward(t, 0)`
twice with a `reset()` between). Three hypotheses were tested and each fix reverted, all
without effect:

1. `CompressedAttention._compress_kv` / `_kv_buf` / `_kv_window` surviving reset:
2. `MLAWindowAttention`'s instance staging buffers surviving reset:
3. `shared_attn.compress_kv` / `index_k` / `topk_idxs` surviving reset.

The tell is that the drift is the same with and without the reset (18.6 vs 19.8), so the
residual state is not in any of those places: the two forwards of the same 24-token
prompt differ on their own. With single-token forwards the engine is exactly reproducible
(`rel 0.000000`), so the difference is specific to the multi-chunk path and is separate
from the slot-pool race. It stays open, and the regression test keeps its xfail.

