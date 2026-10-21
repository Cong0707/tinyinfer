# MTP / DSpark 投机解码集成设计（生产版）

> 状态：权重加载 ✅；stage0/1/2 前向链（attention+MoE+HC）已实现；forward_head（markov/confidence）待完成。
> 源码：`src/tinyinfer/mtp.py`（文件头有完整参考映射与 checkpoint 张量名）。

## 参考实现（修复时先读这些）

| tinyinfer | DeepInfer 参考 | 说明 |
|---|---|---|
| `MTPBlock.__init__` | `mtp/dflash.py:DFlash.__init__` | 3 stage 权重加载；stage0 带 main_proj/main_norm |
| `MTPBlock.main_hiddens` | `DFlash.update_main_cache`（projection+project_norm）| target 层 hidden 拼接 → main_x |
| `MTPBlock.forward_embed` | `DFlash._embed_draft_init_tokens` | noise(128799) 填充 + golden 首 token |
| `MTPBlock.forward` | `DFlashDecoderLayer.forward_drafts` | HC + attention + MoE |
| `MTPAttention.forward_with_main` | `v4_modules/attn.py:DFlashAttention` | main KV 滑窗 + draft KV |
| `MTPBlock.forward_head` | `DFlash.sample_drafts` | post_norm → head → markov → confidence（TODO）|
| 字段 `main_proj/main_norm` | `projection/project_norm` | checkpoint 名与官方模块名对照 |
| 字段 `markov_embed/markov_head` | `shared_head.markov_w1/markov_w2` | |
| 字段 `confidence_proj` | `confidence_head.proj` | |
| `normalize` | `ConfidenceHead.temperature` | 置信度温度归一（参考实现按 num_steps 配置）|

其它框架对照：vLLM 的 nextn/MTP 层（`num_nextn_predict_layers=3`）与 HF transformers 的 MTP 实现同构；draft block_size=5、markov_rank=256。

## 权重命名（checkpoint 精确名）

```
mtp.{S}.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight/.scale     fp8 注意力
mtp.{S}.attn.{q_norm,kv_norm}.weight, attn_sink
mtp.{S}.attn_norm.weight / ffn_norm.weight
mtp.{S}.hc_attn_fn / hc_attn_scale / hc_attn_base / hc_ffn_*
mtp.{S}.ffn.gate.{weight,bias,bias_vl} / shared_experts.* / experts.{i}.{w1,w2,w3}.{weight,scale}
mtp.0.main_proj.weight/.scale [dim, n_target*dim]   （官方 projection）
mtp.0.main_norm.weight [dim]                         （官方 project_norm）
mtp.2.norm.weight [dim]                              （官方 post_norm）
mtp.2.markov_head.embed.weight [vocab, 256]          （官方 markov_w1）
mtp.2.markov_head.head.weight  [256, vocab]          （官方 markov_w2）
mtp.2.confidence_head.proj.weight [1, dim+256]       （官方 confidence_head.proj）
```

## 关键契约

- **main_x** = `projection_norm(main_proj(cat(h_L37, h_L38, h_L39)))`，每层 hidden 是 **HC collapse 后**的 `[1,T,dim]`（`Engine._target_hidden`）。
- **main KV**：`main_kv = kv_norm(wkv(main_x))`（每 stage 自己的 wkv），rope 后进 **main 滑窗**（官方 swa cache），窗口 128。
- **draft 输入**：主 embedding 的 noise 填充块（`noise_token_id=128799`，位置 0 = golden token）。
- **draft KV**：每 stage 的 wkv(x)，rope 后**仅本次 block 有效**。
- **attention**：`[main 窗口, draft kv]` 拼接后 sparse attention（sink + window 索引）。
- **head**：`post_norm(hc_pre(x))` → 主 head logits → markov 修正（embed(ids) @ markov_head）→ confidence（`sigmoid(proj([hidden, markov_emb]))`）。
- **掩码/位置**：draft 位置 = start_pos..start_pos+4；main 位置 = start_pos-1（当前 token）。

## 投机解码流程（Engine.speculative_generate）

1. 主模型 forward 得到 logits，贪心采样 t0，并捕获 target 层 hidden（`_target_hidden`）。
2. `main_x = stage0.main_hiddens(target_hiddens)`；`x0 = stage0.forward_embed(t0)`。
3. stage0→1→2：`x, pre = st.forward(x, pos, pre, main_x)`；最终 stage 的 `forward_head` 得到 draft tokens + 置信度。
4. 主模型 verify（一次并行 forward），贪心匹配接受前缀；不匹配则回滚 KV（snapshot/restore）。
5. 接受率与加速比记录到 benchmarks。

## 实测状态（2026-10-22 10:20，cache/weights 40 层）

- **全链可跑**：prefill→target hidden→main_x→draft embeds→3 stage→forward_head（logits+confidence）无崩溃。
- **投机解码（speculative_generate, draft_len=2, 4 token）**：0.056 tok/s，accept_rate=0%。
  - 结论：MTP 输出分布与主模型不一致，**draft 无效**——投机当前是**负收益**（比常规 decode 0.2 tok/s 慢）。
  - 待查（参考官方 dflash 数值）：markov 修正的 draft_ids 语义（官方每步用上一步采样 token，当前用 noise 填充首轮）、main_x 的 target hidden 语义（collapse 时机）、draft attention 的 main KV 窗口累积。
- **VRAM**：投机时 3 stage 权重上卡 → 峰值 10.7GB > 8GB（WDDM 分页）。投机路径需 stage 权重换出（CPU）或按需加载。

## 已知限制

- 专家 IO 不减少（每 token 40 层×6 专家）——投机省 launch/步数，预计 +10-30%，实测为准。
- markov head 与 confidence 需在 GPU 上融合（1.84ms/5-token 量级）。
- forward_head 的 markov 修正首轮用 noise 填充（官方为逐步 draft token）——正确性待参考数值验证。
- 投机路径 VRAM 需要单独预算（stage 权重不要常驻）。
