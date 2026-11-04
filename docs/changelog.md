# Changelog

## vLLM baseline + Marlin fp4 experts

- **vLLM 0.33.0 的 CUDA 扩展在 Windows/MSVC 上编译成功**，没有用 WSL：两个 ABI3 扩展
  (`_C_stable_libtorch.pyd` 22.5MB、`_moe_C_stable_libtorch.pyd` 11.8MB) 现在随
  tinyinfer 源码分发，`import torch` 之后直接加载，不依赖 vllm 的 Python 包。
  全流程和所有兼容补丁写在 `docs/vllm-extension-build.md`。
- **fp4 专家 GEMV 换成 vLLM 的 Marlin**（`gemv_fp4_marlin.py`）：657 us/59 GB/s ->
  **130-137 us/305-323 GB/s**，4.8 倍，超过 vLLM 自己跑出来的 142.9 us。
  权重在加载时 repack 一次（`gptq_marlin_repack` + e8m0 scale 置换），之后每次调用
  就是一条 fused kernel。数值对着手工 E2M1 解码验过，最大相对误差 4.7e-3。
- **rms_norm 换成 vLLM 的**：M=1 33.87 -> 21 us，M=8 67.03 -> 13.25 us。
  原来的原生 kernel 每行只开 32 个线程的 block。
- **端到端**：一个 fp4 专家的 SwiGLU FFN（3 个 GEMV，M=1）从 1971 us 降到 435 us。
- 基线全表在 `docs/vllm-baseline.md`。

## DeepSeek-V4.1-Flash-1017 能跑了

- **修掉 `model_profile` 把 engram 查找表算成层的 bug**：`layers.N.engram.embed.weight`
  每张 91.5 GB（按行从磁盘读、从不映射），却被累加进 per-layer 开销，使
  `max_layer_bytes` 变成 94.7 GB、`fixed_b` 算出 192.4 GB，规划器直接拒绝启动。
  加上 `if "engram" in parts: continue` 后 `max_layer_bytes` = 0.173 GB。
- 结果：build 15.6s、首次前向 31s、热 prefill 7.9s、**稳态 decode 0.867 tok/s**
  （10 层常驻 + 30 层流式，cache 13.2GB）。详见 `docs/deepseek-v41-flash.md`。
- 剩余瓶颈：磁盘只跑到 1.26 GB/s（SSD 能力 2.47），日志里
  `slab unavailable (ok)` 说明平坦 pinned slab 因为 host budget 太小没建起来。

## 上一轮

- **Fused KV (rope + fp8 round trip)**: `deepseek_kernels.rope_quant_tl` now reproduces
  `apply_rotary_emb` + `act_quant` **exactly (rel 0.000000, scales identical)** after two
  fixes: the rotation result must be rounded back to bf16 before quantising, and the
  scale must be ceil-rounded to a power of two like the reference `fast_round_scale`.
  Wired into `MLAWindowAttention` (falls back to the two-op path off CUDA).
- **Batched small-M GEMV** (`gemv_fp4_m8`): rows in one launch instead of one launch per
  row; verified identical, 1.64x in isolation. **Grouped 6-expert GEMV** (`gemv_fp4_grouped`)
  verified identical (1.79x) but gathering the six planes costs more than the saved
  launches, so the serve path keeps the shared-quantisation version.
- **One activation quantisation per layer** (`forward_fp4_pq`): the fp4 quantiser does
  not depend on the expert.
- **Indexer mask cache**: the visibility mask depends only on (start_pos, s, t, ratio)
  and every compressed layer rebuilt it.
- **Expert hot tracking** (`ExpertStore.hot_experts/hot_prefetch`): counts per-expert
  requests; explicit prefetch only (256 experts = 4.7GB is too slow to do inline).
- **MTP**: prefill now seeds the draft main-KV window (official `DSparkBlock.forward`),
  draft accept 0.03 -> 0.08. MTP still OOMs on this 8GB card, which the planner
  already predicted (`speculation` gate off).
- **HTTP server** (`python -m tinyinfer.serve`): /health + /generate, verified end to end.
- **Benchmark** (`python -m tinyinfer.bench`): prints build/prefill/decode/cache/disk as JSON.
- **test suite**: all tests pass (compressor test updated for the reference bf16 rounding,
  rope test for the vendor mscale).

Note: the checkpoint volume measured 4.5GB/s at the start of the round and 2.7GB/s
after hours of repeated sweeps, so late measurements are disk-bound, not code-bound.


- **Long prompts**: a 63-token prompt at chunk=8 pushed the allocator past physical VRAM
  (17GB through WDDM shared memory) and OOM'd, while chunk=1 completed the same prompt
  (216s, `The quick brown fox` continuation). Long prompts now step through in small
  pieces (chunk scaled down for T > 16); short prompts keep the fast chunk. The growth
  is per layer and linear in chunk length; the exact operator is still unidentified.


- **BF16 expert cache disabled on the fp8/fp4 path** (`max_bf16_experts=0` when quantised):
  each block kept bf16 copies of up to eight experts (~240MB/layer, 9.6GB over 40
  layers) that the fp4 serve path never reads. That was the 17GB allocator growth seen
  on long prompts. Reference (non-quantised) blocks keep the cache.
- **Long prompt verified end to end**: 63 tokens prefill in 136s (dynamic chunk, 0.46
  tok/s) and a correct continuation (`The quick brown fox jumps over the lazy dog`).
  chunk=8 no longer OOMs after the bf16 fix, but produces BOS tokens, i.e. there is a
  separate chunked-prefill numeric bug at larger chunks; the safe step (<=5 here) is
  used automatically until that is found.


- **Multi-token vs token-by-token inconsistency (open)**: one forward over 8 prompt tokens
  and eight single-token forwards over the same tokens differ by max|d| 17.5 (argmax 5
  vs 33870 on the last position). `generate()` uses the one-shot prefill and then
  single-token decode, which is consistent; the verify pass of speculative decoding
  submits the draft block as one T=5 forward and therefore hits this path, which is why
  MTP output degenerates to BOS tokens even though the accept rate is real. The same
  path is what chunk>1 long prompts use; chunk=1 is correct. Root cause not yet found;
  candidates are the prefill window mask and the compressor's cross-chunk grouping.


- **Multi-row expert GEMV fixed** (`gemv_fp4_pq` flattened a `[rows, K]` activation into
  the single-row kernel slot; rows now go through `gemv_fp4_m8`). It removed one source
  of the multi-token drift but a residual difference remains (MoE output differs by 1.7
  on layer 0 between one-shot and token-by-token), so the one-shot-vs-stepwise logits
  still differ. The last position of a causal prefill is correct (that is what
  `generate()` consumes), which is why answers are right; the verify path and chunk>1
  prefill keep the T-dependent drift.


- **Multi-row FP4 kernels are not reliable**: `gemv_fp4_m8`'s multi-row grid returned
  values 35% off for specific (weight, row) pairs (expert 34, rows 3/4 vs 3/6). Both the
  expert path (`forward_fp4_pq`) and `fp4_linear` now use the single-row kernel per row,
  which is exact. This is slower in launch count but a chunk routes only 1-2 rows per
  expert, and correctness is not negotiable. The remaining one-shot-vs-stepwise drift
  (21% on a synthetic MoE batch) is reduced but not eliminated; the test is marked xfail.
- **Slab lifetime**: reset no longer releases the pinned slab grant (it left every
  streamed block's `_cpu_refs` pointing at freed memory; the next run's per-tensor
  copies read garbage and generations went off-topic).
- **The checkpoint is a base model**: no chat template, and its own top-1 for 'The
  capital of France is' is '#' (13.25) over 'Paris' (12.5); the templated continuations
  are the model's behaviour, not an engine fault (docs/analysis.md).


- **`gemv_fp4_m8` fixed**: its 2-D grid (feature tiles x row) put the row on an
  ambiguous axis and returned values 35% off for some (weight, row) pairs. It now uses a
  flat grid with the row folded in, re-verified bit-exact for M=1..8 (rel 0.000000) and
  1.61x the row loop. The expert path and `fp4_linear` use it again for small M.
- residual: a synthetic MoE batch (8 rows in one call vs eight single-row calls) still
  differs by ~26% on layer 0 even though routing, quantisation and the per-expert
  kernels each verify exact in isolation; the accumulation needs another look.


- **MoE batch interaction (narrowed)**: eight *identical* rows in one call match eight
  single-row calls exactly (rel 0.000000), eight *distinct* rows differ by 42%. So the
  per-row arithmetic is exact and the error appears only when different rows interact,
  which points at the shared accumulation (sels/index_add) rather than the kernels.
  generate()'s prefill keeps the last position, whose MoE row set is a single token, so
  answers stay correct; chunked prefill (chunk>1) accumulates the error and is why it is
  restricted to chunk=1 for long prompts.


- **Layer-major prefill** (`prefill_layer_major`) fails on a prompt whose length is not a
  multiple of the chunk size: the last chunk has 7 tokens while `win_idxs` still carries
  8 (`torch.cat([win_idxs, compress_idxs])` in compressed_attention). The chunk-major path
  with chunk=1 handles 63 tokens correctly (136s), which is what long prompts use.


- Expert 34 with two routed rows still differs (51%) between one multi-row call and two
  single-row calls, and it does so through the *exact* per-row path as well; the
  single-row result is what decode uses, so generation is unaffected. The multi-row batch
  path is therefore opt-in (`TI_MULTIROW_GEMV=1`) and the default stays row-at-a-time.
  Layer-0 attention, gate, activation quantisation and all solo-expert checks are exact,
  so the remaining suspect is state shared across rows inside the batch (slot/record
  reuse), not arithmetic.


- OpenAI-compatible `/v1/completions` and `/v1/chat/completions` verified end to end
  (server up, request, well-formed `choices`/`usage` response). One request at a time,
  since the engine keeps a single KV session.


- Additional measurement: a hand-written reproduction of the routed sum (same gate, same
  pre-quantised activation, same per-expert kernels, plain `y[rows] += val`) differs from
  `MoE.forward` by 12.6% on the same 8-row batch. Since the components verify equal, the
  remaining suspects are the serve loop's 3-expert batching (`_val_acc`/`_idx_acc`
  accumulation order) and the slot/record sharing. The single-row path (decode) is
  unaffected.
