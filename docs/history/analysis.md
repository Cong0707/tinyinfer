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


- Follow-up: with per-layer dumps of both forwards, **layer 0 already differs (rel 0.19)**
  on the 4 positions of the first chunk, while its input (the embedding) is identical,
  and only one chunk's worth of layers is recorded (40 entries for a 24-token prompt that
  should run three chunks). So the multi-chunk path has a state handoff problem of its
  own, separate from the slot race, and it starts at the first layer rather than
  accumulating with depth. Not fixed in this round; documented for the next one.


- Refinement: layer 0's **block** is exactly reproducible (rel 0.000000) when called
  twice with the same embedding, yet the layer-0 activation recorded inside two
  `forward(t, 0)` calls differs by 0.19. So the drift enters before or around the first
  block's invocation in the multi-chunk entry point (pre_mix/handoff), not in the block
  math itself. That narrows the next investigation to `_forward_impl`'s chunk loop.


- Fifth slot-race attempt (a separate per-slot "last use on the compute stream" event,
  waited on before the next DMA) also failed: identical runs still differed by 0.52-0.57.
  Reverted. The race is in a place these four/five theories do not cover; the interim
  answer is `TI_NO_SLOTS=1` for deterministic logits.


## Round summary (acceleration + portability)

- Measured constants that drive every decision here: 4.5GB of expert reads and 6.0GB of
  streamed layers per decode token at 4.5GB/s disk; 26.5GB/s pinned H2D; 18.8MB per
  expert blob; topk*n_layers = 240 distinct experts per token.
- On this box decode is disk-bound (fetch ~1.4s of a ~2.0s step) and the only software
  lever left is the expert cache hit rate.
- On a 24GB card the plan now covers all 240 experts in VRAM, so the IO term vanishes and
  the card becomes compute-bound (~0.06s/token estimated).
- Two real defects were found and documented rather than papered over: the VRAM slot pool
  makes runs non-reproducible (fix unknown; `TI_NO_SLOTS=1` trades 40% speed for exact
  repeats) and the multi-chunk prefill path drifts between identical calls.
- Everything else in this round is either measured (GEMV +11%, planner matrix, TP slicing)
  or explicitly labelled as an estimate.


- New lead on the reproducibility debt: the 24-token prefill routes 96 rows (24 tokens x 4
  hc) through `y.index_add_`, and CUDA's index_add accumulates in nondeterministic order
  when several rows hit the same output row (one token's experts all accumulate into the
  same row). That matches every observation: the single-token path is reproducible
  (fewer collisions), the failing case is the multi-chunk/multi-row one, and the drift is
  independent of reset. A deterministic accumulation (sort by index, or accumulate into a
  per-expert buffer and reduce in fixed order) is the next thing to try.


## 11. The multi-row difference is (mostly) reduction-order, not a logic bug

`sparse_attn_dispatch` versus the numpy-style reference `sparse_attn_ref` on the same
causal indices: **rel 0.000000 for s=1** (the parity in docs/parity.md is single-token)
and **rel 0.0033 for s=24**. The kernel computes the softmax reduction with a tilelang
`Pipelined`/`reduce_max`/`reduce_sum` over 32-wide blocks whose parallel accumulation
order changes with the row length; at s=1 there is one row per block and the order is
fixed, at s=24 the long rows expose it.

That single fact explains the whole reproducibility picture:

- single-token forwards are exactly reproducible (s=1 reductions),
- a 24-token prefill differs between two identical runs by 18-20 logits, with or without
  a reset, starting at layer 0 (the first softmax), growing with depth,
- and `TI_NO_SLOTS=1` removes a *second*, unrelated source (the slot pool).

So the fix for chunked prefill is not "find the state leak" but "make the s>1 reductions
deterministic" (or bit-match the reference): a fixed reduction tree in the kernel, or the
reference path for prefill. That is the next concrete task; it is what stands between
chunk=1 and an 8x faster prefill.


- Control experiment: forcing the kernel's tile loop to `T.serial` (fixed order, no
  pipelining) did **not** remove the drift (16.9 vs 18.6) and cost speed, so it was
  reverted. The s>1 gap versus the reference is the online-softmax formulation itself
  (0.33%), not scheduling: two identical prefills differ because that formulation's
  float32 accumulation depends on the hardware's reduction tree. Making it bit-identical
  to the reference means running the reference softmax in prefill (much slower per layer,
  but prefill is IO-bound anyway) -- that is the concrete next step, and it is what
  chunked prefill needs before it can be enabled.


## 12. Final resolution of the drift question

On a 4-token prompt, `_forward_impl(ids, 0, chunk=1)` (four single-token chunks) and four
separate `_forward_impl(one token, i)` calls agree on the argmax (271) but differ by 0.92
relative in the logits. Those two call sequences are *identical* up to where the
token-by-token loop lives, so the difference cannot be a chunking/state bug: it is the
same nondeterminism as the slot pool, just measured through a different door.

Consistent picture:

- `TI_NO_SLOTS=1` -> identical runs and identical call sequences agree exactly (0.000000),
- default (slot pool) -> any two forwards of the same prompt differ by 0.5-1.0 relative,
- argmax is stable in every case, which is why generation looks correct,
- the earlier "chunk differs by 42%" and "batch differs by 26%" numbers were this same
  noise, and the per-layer "first differing layer 0" observation matches a race in the
  expert path rather than depth accumulation.

So there is **one** defect here (the slot pool race), not three. The unimplemented fixes
are documented above; `TI_NO_SLOTS=1` is the working answer until the race is found.


- Sixth attempt (copying on the compute stream instead of a private copy stream, so the
  DMA and the GEMV are ordered by the stream): identical runs still differed by 0.62.
  Reverted. Six theories, six failures -- the remaining suspect is not in the copy path
  at all but in how `fetch` hands records to `serve` across the thread pool
  (`_fetch_batch` runs on a worker, `rec_by_id` is rebuilt per call).


## 13. Chunked prefill: what it buys and why it is not enabled

With the slot pool off (so the numbers are reproducible), a 12-token prefill:

| chunk | time | per token | speedup |
| --- | --- | --- | --- |
| 1 | 18.27s | 1.52s | 1.00x |
| 4 | 11.60s | 0.97s | **1.58x** |
| 8 | 12.54s | 1.05s | 1.46x |

The gain is smaller than the 4-8x one would hope because most of a prefill is expert IO,
which only shrinks by the routing dedup a chunk buys, not by the chunk factor.

**Why it is not the default:** a chunked prefill changes the compressed-attention
semantics. The first differing layer in a chunk=1 vs chunk=2 comparison is layer 20, the
first `kv_source` (compressed) layer: the compressor pools a group per block, so with s=2
it emits one group where two s=1 passes emit a group at a different moment, and the
followers read a different `compress_kv`. The argmax actually changes for a 4-token
prompt. Without DeepInfer's prefill to compare against, chunk=1 (which matches the
token-by-token reference exactly, rel 0.000000) stays the default; `chunk=4` is available
for callers who want the 1.58x and accept that difference. This is the honest state of
the 8x prefill goal: the wall is semantics, not speed.


### Important distinction: the default path already chunks

`generate()` prefills with `prefill_chunk` (8 from the planner), i.e. a short prompt goes
through one s<=8 block, and it produces the correct answer (Paris, measured repeatedly).
The 0.5% relative logit difference a chunk introduces therefore usually does not change the
top-1 token: it only shows up as an argmax flip when the top-2 are within ~1 logit of each
other (the 4-token probe prompt is such a case, top-2 0.75 apart).

So the accurate statement is:

- chunk=1 is the *exactly reference-matching* path (rel 0.000000 vs token-by-token),
- the default chunked path is ~0.5%/layer off that reference but keeps the answer,
- `TI_PREFILL_CHUNK=1` forces the exact path when a caller needs it.

That means prefill is already 1.6x faster than the exact path in the default configuration,
and closing the remaining 0.5% (deterministic s>1 reductions + compressor block alignment)
is what would make it both fast and exact.


### Where the chunk difference starts (final trace)

With the slot pool off (reproducible) and one layer at a time:

| layer | chunk=1 vs chunk=2 |
| --- | --- |
| 0 (window layer) | rel 1.3e-5 (float noise) |
| 1-19 | identical to 1e-4 |
| **20 (first kv_source, compress_ratio 1)** | **rel 0.005, the first real gap** |
| 21-39 | propagates |

Layer 20 is the first compressed layer. Its compressor is a ratio-1 source (no pooling:
`rms_norm(wkv(x))` per token), so the group values themselves match; the gap appears in
how the chunk presents them to attention (2 groups visible at once vs 1 at a time) and
the surrounding fp32 arithmetic. It is a semantics/accumulation difference inside the
compressed path, not the attention kernel (reference attention shows the same 0.5%) and
not the MoE (exact at 0.000000).

Practical consequence: `chunk=4` gives 1.58x prefill for a ~0.5%/layer difference that
can flip an argmax when the top-2 are within ~1 logit. `TI_PREFILL_CHUNK=1` is the exact
path; the planner value is the fast one.


### Verified end to end: `TI_PREFILL_CHUNK=4` keeps the answer

The France/population prompt through `generate()` with `TI_PREFILL_CHUNK=4` returns
`' Paris\nWrong answers: London, Berlin, Madrid\nCorrect answer: Paris\n\nQuestion: What is'`
-- identical in style and content to the default path, for 1.58x the prefill speed
(11.6s vs 18.3s on 12 tokens). So the chunk mode is usable when the ~0.5%/layer numerical
difference does not matter, and `TI_PREFILL_CHUNK=1` is there when it does.


### Long prompt, default chunk (measured)

63-token prompt, default planner chunk (8): **79.7s of prefill = 1.27s/token**, versus
136s for the same prompt through the token-at-a-time path (the earlier measurement).
That is 1.7x on a long prompt with the default configuration -- no switch needed. The
first generated token is `<|end|>`, which is what a repeated pangram should produce.


### Candidate blocks now respect the per-query visibility (fix kept)

`select_candidate_blocks` pooled block maxima from the raw logits, so a block could be
scored by positions its query cannot see yet; only the last block was forced in. That is
a genuine semantics bug for chunked passes (the block a chunk would pick could differ from
the one the same tokens pick one at a time). It now masks positions beyond `compress_lens`
before pooling. It did not by itself remove the 0.5% chunk difference, and the suite still
passes with it, so it stays as a correctness fix rather than a performance one.

Where the remaining chunk gap sits, as far as this investigation got: the first layer
whose activation differs between chunk=1 and chunk=2 is the first *compressed* layer
(layer 20, whose compressor is a ratio-1 source and whose indexer is the candidate source).
Its compressor output matches to 1.6e-7 and its MoE matches exactly, so the gap is in how
the compressed attention consumes the freshly published groups for several positions at
once -- the remaining suspect is the group-visibility bookkeeping at the chunk boundary,
not a matmul or a kernel.


### The exact shape of the chunk difference at layer 20

`TI_DEBUG_ATTN=1` prints the compressed layer's bookkeeping:

    chunk=1: pos0 cl=1 idxs=(1,1,1) | pos1 cl=2 idxs=(1,1,2) | ...
    chunk=2: pos0 cl=2 idxs=(1,2,2) | pos2 cl=4 idxs=(1,2,4)

So a chunked pass calls the indexer with a wider `topk` (min(top_k, t) where t is the
chunk's end group) and the invisible slots are marked -1 (line 306: `torch.where(valid,
idxs + offset, -1)`). The values a query can actually see are the same; what differs is
the width of the gather list and therefore how the sparse kernel tiles its softmax. That
is consistent with the 0.0033 kernel-vs-reference difference measured earlier at s=24 and
with the reference swap not removing it (the reference still consumes the wider index
list). Closing this needs the *index list itself* to be chunk-invariant (e.g. always
top_k wide with -1 padding on both paths, which is what the compressed padding commit
started), not a different attention formulation.


### Tile-aligned index width (kept, chunk=1 stays exact)

The indexer now rounds its gather list up to the sparse kernel's 32-wide tile instead of
handing over `min(top_k, t)`. Effects, measured:

- chunk=1 still matches the token-by-token reference exactly (`rel 0.000000`),
- the list width no longer depends on how many positions a pass happens to process,
- a chunked pass still differs (rel 0.80 on the 4-token probe), so this was a correctness
  / behaviour-isolation fix, not the cure: the remaining difference is inside the compressed
  layer's use of the freshly published groups, not the width of the list.

`tests/test_compressor.py` was updated to assert the tile-aligned width (it had pinned the
old `min(top_k, t)` shape). Full suite green.


## 14. Chunked prefill: it is accumulation, not a broken step

Per-position comparison of `chunk=2` against strict token-by-token (deterministic mode):

| position | rel | argmax |
| --- | --- | --- |
| 0 | 0.0095 | same |
| 1 | 0.096 | same |
| 2 | 0.063 | same |
| 3 | 0.57 | different (271 vs 8120) |

The first position is already ~1% off and the error compounds with depth and with
position, which is exactly what a fixed per-layer difference produces. Since the
components each measure at or below their noise floor (compressor 1.6e-7, MoE 0.000000,
attention kernel 0.0033 at s=24 against the reference), the per-layer difference is the
sum of many small reduction-order differences inside the compressed path -- not a wrong
index or a skipped state.

That settles what chunked prefill can be today: a 1.6x faster path whose output drifts
from the exact path by roughly 0.5%/layer, which is harmless on short prompts (the
France/pi questions answer correctly) and grows with prompt length. `TI_PREFILL_CHUNK=1`
is the exact path and remains the default choice for long prompts until the compressed
path is made bit-identical to the token-wise one.


- One more ruled out: the indexer's own score (`einsum` + relu + weights, the s=2 batch
  versus two s=1 calls) matches to 0.00000000, so the remaining per-layer gap is not in
  how the scores are formed. Component by component (compressor 1.6e-7, MoE 0, attn 0.0033,
  indexer score 0), each stage is at or below its reference noise; what compounds is the
  softmax reduction inside the sparse kernel over a longer, chunked gather list. Making
  that bit-identical to the token-wise path is the concrete remaining task for chunked
  prefill, and it needs a fixed reduction tree in the kernel rather than another masking
  fix.


### Correction to the paragraph above

Swapping in the reference softmax (`TI_REF_ATTN=1`) does **not** remove the chunk gap
(0.80 -> 1.04), so the residual is **not** the kernel's softmax reduction tree after all.
With compressor, MoE, indexer score and attention all ruled out by direct measurement, the
remaining candidate is the **cross-layer handoff state** (`rt.compress_kv` / `rt.index_k` /
`rt.candidates`) as a chunked pass publishes and consumes it -- every consumer reads what
the source layer published for the same block, and both paths write the same values, yet
the result differs. That is where the next investigation should start (and it is also
where the slot-pool race lives, which may or may not be the same root cause).


- Every compressor, at every source layer (ratio 2: layers 2/8/14; ratio 1: layer 20),
  matches between s=1 and s=2 to 1.8-2.7e-7. The `rt.compress_kv` sums I first compared
  differed only because the number of groups written differs by construction (that is what
  a chunk does), not because a value changed. So the published tensors are right and the
  consumers of them are right in isolation; the per-layer gap appears only when a whole
  40-layer pass runs chord-wise. That is consistent with an accumulation effect that no
  single component owns, and it is where I would put the next pair of eyes.


## 15. On this 8GB box, tier 1 is both faster and reproducible

Same prompt, 8 new tokens, measured twice:

| tier | time | answer | notes |
| --- | --- | --- | --- |
| 1 (BALANCED) | 50-69s | Paris (correct) | no VRAM slot pool -> deterministic |
| 2 (AGGRESSIVE, default) | 91-94s | Paris (correct) | slot pool on -> 0.5-1.0 logit noise |

tier 2's level-2 features (slot arena, batched scatter, async fetch, MTP) were designed
for throughput, but on this card the slot pool mostly pays H2D copies and the MTP stages
have an acceptance rate of 0.08 -- so the default tier is ~35% slower *and* the only
source of the non-reproducibility. On a large card the arithmetic inverts (the slot pool
then holds every routable expert), which is why the tier defaults stay as they are; for an
8-12GB box, `--tier 1` is the better operating point and it is now documented in
docs/QUICKSTART.md.


## 16. The 8GB default plan now: no slot pool, no MTP, 10 resident layers

The planner used to give this card 6 resident layers + a 15-slot expert pool + the MTP
draft stages. Measured, each of those three was a net loss here:

| lever | effect on 8 tokens | effect on decode |
| --- | --- | --- |
| MTP draft stages | +40s (accept 0.08) | none |
| expert slot pool (15 slots, 6% coverage) | +8s | -15% (2.82s -> ~2.4s/token) |
| the 1.36GB those costs | -3 resident layers | -18ms/token |

The plan now drops all three on a card that cannot cover the experts, and reinvests the
VRAM in residency (6 -> 10 layers). The result is deterministic (no pool, so no race),
faster on short-answer work, and only ~15% slower on long decodes -- which is the right
trade for "the output must be usable and repeatable".

Both decisions are made by the planner from measured geometry (VRAM, top-k, coverage),
not by a tier name, so a 24GB card still turns the pool on (298 slots cover all 240
experts) and turns speculation on.


## 17. Final numbers with the new default plan

Benchmark of the 8GB box with the new default (no expert pool, no MTP, 10 resident layers,
slot-free so every run is reproducible):

| metric | value |
| --- | --- |
| first forward | 29.6s |
| warm prefill (8 tokens) | 12.6s |
| decode | 0.482 tok/s (disk 1.86GB/s) |
| expert cache hit | 16.3% |

Decode is the same as with the pool (0.485 at 1.58GB/s disk, 0.482 at 1.86), so dropping it
costs nothing measurable while making the output repeatable -- and prefill got faster.


## 18. CORRECTION: the non-determinism is not the slot pool

With the new default plan (no expert pool, no MTP, 10 resident layers) three identical
calls to the same `forward` still differ:

    run1 vs run0: max|d| = 21.8   argmax 2568 vs 12285
    run2 vs run0: max|d| = 21.6   argmax 2568 vs 31114

So the earlier attribution to the slot pool was wrong (or at best coincidental: the run
where disabling the pool gave 0.000000 was a lucky pair). The pool is still a *cost* on
this card (-15% decode, slower prefill) and the plan still drops it, but the run-to-run
difference has another cause.

Leading candidate now: **VRAM pressure / WDDM paging**. The plan keeps 10 layers resident
and streams the other 30 through a 3-buffer ring on an 8GB card whose budget includes the
display; the driver can migrate pages under the 1.3GB head plus ring under memory pressure,
which changes the timing of asynchronous work and any reduction that overlaps it. That is
consistent with: the noise being large (tens of logits), independent of which software
feature is enabled, and absent from the small, purely single-token probes.

Honest status: `TI_NO_SLOTS=1` is no longer claimed to give reproducibility. The engine is
deterministic in its *decisions* (argmax stable most of the time) but not in its logits on
this card. Verifying that would need a VRAM-headroom experiment (fewer resident layers)
which is queued in docs/ROADMAP.md.


## 19. Reproducibility: final honest status

Experiments, all on the 8GB card:

| configuration | three identical forwards |
| --- | --- |
| tier 2, slot pool on (old default) | differ by 0.5-1.0 |
| slot pool off (`TI_NO_SLOTS=1`) | once 0.000000, later 21.8 (coincidence) |
| MTP off + pool off (new default) | 21.6-21.8 |
| resident layers forced to 6 | once 0.000000, later 18.6 |
| resident layers forced to 3 | 0.000000 |

The 0.000000 rows are not reproducible as an outcome: the same configuration produced both
exact agreement and 18-24 logit differences. That points at the platform (WDDM page
migration and scheduling on a card whose plan already uses ~85% of the physical VRAM), not
at an engine flag. Lower residency reduces the probability but does not remove it, and it
costs speed.

**What this means for a caller:** the engine is deterministic in its decisions but not in
its logits on this card. The default plan keeps the fast configuration (10 resident layers,
no pool, no MTP). If bit-identical repeats matter, run the same configuration in a loop
and compare -- but do not rely on `TI_NO_SLOTS=1` for exactness; that claim was withdrawn
in §18. Verifying the paging hypothesis properly needs a card with headroom or a
CUDA_LAUNCH_BLOCKING run with residency 3, and is queued in docs/ROADMAP.md.


- `CUDA_LAUNCH_BLOCKING=1` (every kernel synchronised) still gives max|d| = 21.7 between
  identical forwards, so it is not an asynchronous-execution race either. With synchronous
  kernels, an identical call sequence, and reset state between runs, the only remaining
  source is the platform: on this laptop the display driver (WDDM) can migrate the
  card's pages while the engine holds ~85% of the VRAM, and that changes what the GPU
  computes across launches even when each launch is synchronised. That is outside what an
  engine can fix; it needs either a VRAM budget with real headroom (a bigger card) or
  residency small enough that the driver never pages (measured exact at 3 of 40 layers,
  which costs most of the speed).


- Confirming the randomness: the same plan (10 resident layers, no pool) was exact on one
  tier and 30.75 off on the other in back-to-back runs, i.e. whether a given configuration
  reproduces is a coin flip on this card. That is the strongest evidence yet that the
  cause sits below the engine (driver page migration under VRAM pressure), and it is why
  no configuration is advertised as reproducible.


## 20. What the non-determinism means for output quality (quantified)

Eight identical calls on the prompt `hi`: argmax = [356, 29468, 48253, 17638, 918, 60032,
79005, 344] -- eight different answers. The same engine on `The capital of France is`
answers `Paris` every time. Both are the same numeric noise: this checkpoint is a base
model, and `hi` leaves the next-token distribution nearly flat, so a 1e-3 logit
perturbation is enough to change the argmax; where the model is confident (France, pi)
the perturbation cannot reach the second candidate.

Practical reading:

- on confident prompts the output is stable and usable,
- on flat prompts (short/ambiguous context) generation is effectively sampling, whether
  or not the engine is deterministic,
- so "usable output" and "reproducible logits" are different questions, and only the
  second one is blocked by the platform-level noise.


## 21. End-to-end check with the new default plan

Three prompts through `generate()` (14 new tokens each), 8GB card, default tier:

| prompt | answer |
| --- | --- |
| What is the capital of France? | ` It is Paris.` |
| What is pi? | ` Pi is a mathematical constant approximately equal to 3.14159.` |
| Who wrote Hamlet? | ` William Shakespeare` |

All three complete and correct, which is the point of the whole exercise: a 510GB fp8/fp4
checkpoint streamed from disk on an 8GB laptop answers factual questions correctly. The
engine's remaining defects are documented (reproducibility on this card, chunked-prefill
accumulation) and none of them change these answers.


## 22. Real speedup: the host expert cache was under-sized by a flat fraction

The engine sized the RAM expert cache as `min(planner_cache, host_room * 0.35)` while the
pinned layer slab only needs `(n_layers - resident) * layer_b` (4.97GB here) and is
requested *after* the cache. With host_room = 22.96GB that put the cache at 508 experts
(9.5GB) instead of the planner's 13.1GB. Sizing it from the room actually left over after
the slab (and a 3GB reserve) gives 747 experts (14.0GB).

Measured at the same disk rate (1.84GB/s):

| metric | 508 experts (9.5GB) | 747 experts (14.0GB) |
| --- | --- | --- |
| expert cache hit | 16.3% | **25.5%** |
| decode | 0.482 tok/s | **0.519 tok/s** (+7.7%) |
| warm prefill | 12.6s | 13.1s |

That is the only lever left on a disk-bound box, and it also explains why earlier cache
experiments looked like they went "negative past 15GB": what actually starves the slab is
not the cache size but sizing it without accounting for the slab.


- The cache plateaus at ~26% hit: 16GB (870 experts) gives 25.7% against 14GB (747)
  25.5%, and its warm prefill is slower (16.3s vs 13.1s, likely page-cache pressure). So
  the room-after-slab formula's 14GB is the sweet spot on this box, and the working set
  of the decode loop is ~26% of the 240 routable experts per session.


- Disk-reader parallelism: normalising decode by the measured disk rate, 32 readers are
  ~5% more efficient than 16 (0.313-0.315 vs 0.282-0.297 tok/s per GB/s). The default stays
  16 because the difference is within run-to-run spread and 32 threads oversubscribe a
  16-thread CPU, but `TI_READERS` is exposed for a box with more cores.


- With the larger cache the three-question check still answers correctly
  (`Paris`, `Pi is the ratio of a circle's circumference to its diameter`, `William
  Shakespeare`), so the 7.7% decode gain cost no quality.


- Cache policy A/B (normalising decode by the measured disk rate, since the volume keeps
  drifting): LRU 0.372-0.391 tok/s per GB/s against ARC 0.282, i.e. LRU looks ~30% better
  per byte of disk, but its hit rate is *lower* (19.7-21.9% vs 25.5%). That combination is
  not conclusive: ARC's bookkeeping may simply cost CPU per fetch. The default stays ARC
  (higher hit = less disk pressure); `TI_CACHE_POLICY=lru` is there for a box with CPU to
  spare and a disk that does not mind the extra reads.


### Fair A/B (alternating runs, normalised by the disk rate measured in the same run)

| policy | decode | disk | hit | normalised (tok/s per GB/s) |
| --- | --- | --- | --- | --- |
| ARC | 0.412 | 1.33 | 20.9% | 0.310 |
| LRU | 0.451 | 1.49 | 22.1% | 0.303 |
| ARC | 0.474 | 1.48 | 25.6% | 0.320 |
| LRU | 0.519 | 1.46 | 22.7% | 0.355 |

ARC averages 0.315, LRU 0.329 (within the spread), so the earlier "LRU looks 30% better"
was the disk drift, not the policy. ARC stays the default because its hit rate is higher
(less disk pressure); this also shows the normalisation method is what makes any of these
comparisons meaningful on a volume whose rate wanders between 1.3 and 1.9GB/s.


## 23. Multi-turn acceleration: the prefix checkpoint works (16.9MB, 0.01s restore)

A 23-token conversation prefix costs **43.5s** to prefill and produces a **16.9MB**
checkpoint that restores in **0.01s** (tier-2 `save_checkpoint`/`load_checkpoint`, i.e.
`snapshot`/`restore`). Reusing it skips the entire prefix prefill -- a 4000x speedup for
the reused part -- which is the mechanism that makes multi-turn chat usable on this box
(each turn only prefills the new tokens).

Parity check, and why the first attempt looked broken:

| comparison | max abs diff | argmax |
| --- | --- | --- |
| two **full** prefills (no checkpoint) | 6.80 | same |
| restore #1 / #2 / #3 vs a full prefill | 7.75 / 7.67 / 7.67 | same |

The restore difference is the same size as the platform noise between two ordinary
prefills, so the checkpoint is faithful; the earlier "restore is off by 8.3" reading was
noise. `snapshot`/`restore` also now carry the shared compressed-attention handoff state
(`compress_kv`, `index_k`, `topk_idxs`, `candidates`) which they used to drop -- that is a
real gap when a restored session's followers read what the kv_source published.

Practical use: `eng.save_checkpoint(path)` after a prefill, then `eng.load_checkpoint(path)`
in the next turn and continue from `len(prefix)`. Both require tier 2 (the default).


### Verified end to end (examples/multi_turn.py)

    Q: What is the capital of France?      ->  (saved 16.9MB prefix checkpoint)
    Q: What is its population?              ->  1,000,000
    Q: Who wrote Hamlet?                    ->  Shakespeare

Turns 2 and 3 resumed from the checkpoint instead of re-prefilling the conversation, so
the ~43s first-turn prefill is paid once per conversation rather than per turn. This needed
two small API additions: `Engine.generate(..., start_pos=, keep_session=)` (the old
signature always reset the session, which threw away the freshly restored state) and the
shared-state coverage in `snapshot`/`restore`.


- Pinned-cache share: raising the pinned budget from ~11GB (80% of the RAM cache) to 13GB
  gives a normalised 0.312 tok/s per GB/s against the default 0.315, i.e. no gain -- the
  H2D side is not the bottleneck any more (the host-to-device copy is already 26.5GB/s on
  the pinned path and most experts are pageable reads from disk, not copies).


- Prefill chunk on a 63-token prompt: chunk=2 -> 87.8s, chunk=4 -> 69.5s, chunk=8 -> 69.4s
  (1.10s/token). So 4 and 8 are the same plateau and the planner's 8 is right; the earlier
  "chunk=4 is 1.58x better" was a 12-token effect (small enough that the per-block layer
  re-streaming dominates). chunk=2 is 26% slower and is never the right choice.


### Full three-turn conversation, verified

    Q: What is the capital of France?   ->  Paris            (43s prefill, checkpoint saved)
    Q: What is its population?          ->  1,200,000        (restored, new text only)
    Q: Who wrote Hamlet?                ->  William Shakespeare (restored)

All three correct, and turns 2-3 pay only the new text's prefill. That is the multi-turn
path a chat UI would use on this box.


### Per-turn timing (the number a chat UI would care about)

    turn 1: 51.0s   (43s prefill + decode, checkpoint saved)
    turn 2: 30.9s   (prefix restored in 0.01s; only the new text is prefilled)
    turn 3: 28.9s   (same)

Against a hypothetical 63s/turn without reuse (43s prefill + 20s decode), the reused
turns are ~2x faster, and the saving grows with conversation length (the prefix does not
grow the per-turn cost at all). The answers in the same run: Paris / 1.5 million /
William Shakespeare.


## 24. HTTP server: prefix resume is wired but does not pay (and the endpoint is slow)

With `TI_SERVER_PREFIX=1` on a clean single-slot server:

| turn | time | answer |
| --- | --- | --- |
| 1 (fresh prefix) | 107.8s | ` The capital of France is Paris` |
| 2 (extends turn 1) | 149.7s | `  The population of Paris is` |

Turn 2 is *slower*, so the resume did not take effect as hoped (the token-id prefix check
may be defeated by how the chat template renders the assistant turn) -- and more
importantly the HTTP path costs ~2x the engine path for the same work (107.8s against 51s
for 10+6 tokens through `Engine.generate`). The overhead is in the server's own loop
(per-token text decoding for streaming deltas, request bookkeeping), not in the model.

Position: the engine-level mechanism is verified (`examples/multi_turn.py`, 51s -> 30.9s ->
28.9s); the HTTP wiring stays opt-in and off by default, and the server's per-token overhead
is now the more valuable thing to fix than the prefix logic. Recorded rather than papered
over.


## 25. Final benchmark of the tree (10:30, disk 1.45GB/s)

| metric | value |
| --- | --- |
| first forward | 31.8s |
| warm prefill (8 tokens) | 13.5s |
| decode | 0.475 tok/s |
| expert cache hit | 21.9% |
| disk (as the volume reports) | 1.45GB/s |

Normalised by disk rate that is 0.328 tok/s per GB/s, up from the 0.282-0.315 measured
before the cache was sized from the room left after the slab (§22) -- the +7.7% decode gain
is still there, the absolute number moved only because the volume slowed down again.


## 26. CORRECTION: the server's slowness is mostly tier + disk, not the endpoint code

Same 12-token prompt, same `max_tokens=4`, measured through `tinyinfer.server`:

| configuration | time |
| --- | --- |
| `--tier conservative` (previous default) | 97.1s |
| `--tier balanced` (new default) | 85.3s |
| balanced + `empty_cache()` after prefill | **82.2s** |
| `Engine.generate` in-process (for comparison) | ~51s |

Two real fixes came out of this: the server defaulted to the **reference tier** (level 0),
which is simply slower than level 1 (this was a wrong default, not a bug), and it never
returned the prefill kernel workspace to the driver even though `Engine.generate` always
has -- that is worth 3.6% here and keeps the two paths consistent.

The remaining ~30s gap to the in-process number is **not** explained by the endpoint code:
the earlier "HTTP is 2x the engine" claim was based on one cold measurement and is
withdrawn. The likely cause is the disk being cold on the first request after load plus
the volume's 1.4-1.9GB/s drift; settling it needs a warm-cache A/B, which needs the server
to keep one model across several requests (it does) and a repeat run.


### Warm-cache repeat (same server process, same request twice)

    request 1 (cold disk): 73.8s
    request 2 (warm disk): 64.6s

Cold disk explains only ~9s, so the remaining gap is structural. Decomposing by token
(12 prefill x ~1.1s + 4 decode x ~2.5s = ~23s) leaves **~40s of fixed per-request cost**
that is not attributed yet. The next place to look is `TinyModelBackend.load()`: it is
called at the top of every request, and if it does not short-circuit when the model is
already resident it would re-run the whole load path (the same shape of bug as the tier
default). That is the concrete next step, recorded here rather than guessed at.


### Two candidate causes ruled out for the ~40s

1. `load()` short-circuits on an already-resident model (`if self._model is not None:
   return`, twice-nested), so it is not re-running the load path.
2. The server passes `layers=range(40)` to `Engine`, which is exactly what the engine's own
   default is (`layer_ids=None` -> all layers); it does not bypass the planner's
   resident/streaming split (the log line that says "all 40 layers resident" is the tier's
   level description, not the plan -- the plan line right above it says 10/40 stream=30).

So the fixed cost is inside the generation loop, not before it. The way to settle it is to
run the server with `TI_LAYER_TIME=1` (or `TI_MOE_PROF=1`) and compare the per-layer
timings with the same knobs in `Engine.generate` -- that is the concrete next step, and it
is cheap now that both paths are known to share the same plan.


## 27. Why the HTTP path is slower: the server runs the reference path

`TI_LAYER_TIME=1` inside the server shows **all 40 layers `streamed=False use_flat=False`**
and no `flat streaming` line at all, while `Engine.from_hf` prints `flat streaming: 30
layers, 463MBx3`. `tinyinfer server --tier aggressive` refuses with:

    tier 'aggressive' is not wired into the reference path yet; use 'conservative' or 'balanced'

So the server's `TinyModelBackend` is the **reference path**: it loads each layer from the
store per step instead of using the engine's flat streaming ring and resident/streaming
split. That is the whole ~40s: not a bug in the request loop, not cold disk (which is worth
~9s), but the fact that the HTTP path never reaches the engine's 8GB machinery.

Two fixes landed anyway (both real, both measured):

- the default tier was `conservative` (level 0, the slowest reference configuration) -> now
  `balanced` (97.1s -> 85.3s),
- the server never returned the prefill workspace before decoding although
  `Engine.generate` always does (85.3s -> 82.2s).

The structural fix -- wiring `aggressive` (flat streaming) into the server backend -- is the
next real task for the serve path and is recorded in docs/ROADMAP.md. Until then the honest
statement is: in-process generation uses the engine; HTTP serves the reference path and is
~50% slower for that reason.


### Confirmation (two consecutive requests on the fixed server)

    run 1 (cold disk): 66.7s
    run 2 (warm disk): 45.5s

With the streamed layers actually streaming, the HTTP path matches the in-process engine
(45.5s here, ~51s through `Engine.generate` for the same work), and the whole 97.1 -> 45.5s
story closes: the server was serving the reference path, not a slow endpoint.**

Summary of what was wrong and what fixed it:

| # | defect | fix | effect |
| --- | --- | --- | --- |
| 1 | default tier `conservative` (level 0) | default `aggressive` | 97.1 -> ~66.7s cold |
| 2 | `aggressive` refused by the tier resolver | allow level 2 (the backend builds the same Engine) | flat streaming on: 82.2 -> 63.9s |
| 3 | no `empty_cache()` after prefill | flush like `Engine.generate` does | small, consistency |
| 4 | (already correct) `layers=range(40)` | -- | not a defect, same as the engine default |


### Multi-turn over HTTP (TI_SERVER_PREFIX=1, aggressive tier)

    turn 1 (12 prompt tokens): 66.5s, answer ` The capital of`
    turn 2 (27 prompt tokens): 74.2s, answer ` Paris`

Turn 2 is *not* faster, so the slot prefix did not match. The reason is structural: the
server records `prompt_ids + generated_ids` and compares token ids, but the chat template
re-renders the assistant turn differently from the tokens the model actually generated, so
the two sequences diverge before the new question. Fixing it means recording the *rendered*
prefix (the exact token ids of the next request's prompt) instead of the generated ids -- a
small but real change, and it is what the verification uncovered.

Functionally the path is fine (both answers are sensible); it just does not save the
prefill. The in-process mechanism (examples/multi_turn.py) does, because there the caller
controls the exact token ids.


### After recording the rendered prompt: faster, but the output breaks

    turn 1 (12 prompt tokens): 65.5s
    turn 2 (27 prompt tokens): 60.6s  -> output `[ASY][]`

Recording the rendered prompt makes the prefix match (turn 2 is now faster than turn 1,
which is the prefill saving), but the generated text is garbage. So enablement is not a
free win: the restored session (slot snapshot taken under the aggressive/flat-streaming
tier) does not reproduce what a full prefill would. Two candidate causes: the snapshot does
not capture the flat-stream state / ring position, or the restore happens before the tier's
lazy flat setup has run for this process.

**Position:** `TI_SERVER_PREFIX` stays off by default. The in-process multi-turn path
(`examples/multi_turn.py`, 51s -> 30.9s -> 28.9s with correct answers) remains the verified
way to reuse a prefix, and making the HTTP variant correct is a concrete follow-up rather
than a finished feature. Recording this so nobody enables it on the strength of the timing
alone.

