# MTP / DSpark 投机解码集成设计（等 model-00044..46 分片）

## 权重命名（官方）
mtp.{0,1,2}.attn.{wq_a,wq_b,wkv,wo_a,wo_b,q_norm,kv_norm,attn_sink}
mtp.{stage}.ffn.{gate,shared_experts,experts.*}（n_routed_experts=128, top-6）
mtp.0.main_proj / main_norm（stage 0）
mtp.2.norm / markov_head.embed / markov_head.head / confidence_head.proj（最后 stage）
mtp.{stage}.attn_norm / ffn_norm / hc_*（与主 block 同名）

## 结构（reference/official/model.py）
- DSparkAttention(Attention)：window-only；forward(x, start_pos, main_x)：
  - main_kv = kv_norm(wkv(main_x)) + rope -> 写 window_kv_cache
  - start_pos==0：只 seed，返回 x（prefill）
  - else：q=wq_b(q_norm(wq_a(x)))·rope；kv=kv_norm(wkv(x))·rope；
    topk_idxs = [0..min(win,start_pos)] + [win, win+block)；
    sparse_attn(q, cat[cache, kv], sink, topk_idxs) -> o；inverse rope；wo_a einsum；wo_b。
- DSparkBlock(Block)：forward_embed(main_hidden, ids) -> (x, main_x)；
  forward_head(x, pre_mix, ids) -> (output_ids, logits, confidence)
  - markov_head: embed(ids) -> logits 修正；confidence_head: cat(hidden, markov_embed) -> 1。
- Transformer.forward_spec：draft 各 stage -> 主模型 verify（并行 attention + rejection sampling）。

## tinyinfer 实现方案
1. mtp.py：
   - _PrefixStore(store, "mtp.{stage}")：把 layers.0.attn.* 映射到 mtp.{stage}.attn.*
   - MTPAttention(MLAWindowAttention)：加 forward_with_main（官方语义）
   - MTPBlock：attn + MoE(128 experts) + HC（复用 hc_mixes/hc_pre/post）+ 可选 main_proj/markov/confidence
2. Engine.speculative_generate(prompt, n, draft_len)：
   - 主模型 forward 得 logits -> 采样 t0
   - MTP stage0: main_hidden -> forward_embed -> draft tokens
   - 主模型 verify：对 draft tokens 做一次并行 forward + rejection sampling
   - 接受前缀，回滚 KV（快照/恢复已有 snapshot/restore）
3. 风险：
   - 专家 IO 不减少（每 token 40 层 x 6 专家）——投机只省 launch/步数，预计 +10-30%
   - markov head 1.84ms/5-token（需融合/量化）
   - 验证需 chunk prefill（attention 已支持 start_pos/s）

## 分片到位后的步骤
1. 验证 mtp.* 加载（safe_open header）。
2. 单 token 前向数值 vs 参考。
3. 投机循环 + 接受率测量。
4. 与 40 层全模型联调。
