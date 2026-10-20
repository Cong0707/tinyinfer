# DeepInfer 性能机制 → tinyinfer 映射

DeepInfer 是当前最快的推理引擎（SM100 集群，TP/EP 并行）。以下按"机制 → 我们是否采用/计划"整理。

## 已借鉴 / 已实现

| DeepInfer 机制 | tinyinfer 对应 |
|---|---|
| HF safetensors 直接加载（专家按层聚合、engram 分块行读、ue8m0 转换） | weights.py 直读；engram.py 按行 mmap 读 |
| 压缩 KV + indexer（top-k 稀疏注意力） | compressor.py / compressed_attention.py（支持分块 prefill/decode） |
| tilelang 算子（fp8_gemm 32x32、fp4_gemm、act_quant、sparse_attn） | gemm.py 直接复用官方 reference/official/kernel.py；kernels.py 的 sm120 sparse_attn 补丁 |
| shared_attn 跨层复用（source 层产出、consumer 层读取） | compressor.SharedAttentionRuntime |
| Hyper-Connections（Sinkhorn 双随机） | hc.py（PyTorch 移植，数值与 kernel 一致） |
| DSpark/MTP 结构 | 官方 DSparkBlock 已读；等 model-00044..46 分片实现 |
| 引擎侧 batch 与 slot 管理 | server 的 slot/session + Engine.snapshot/restore |

## 待采用（按收益排序）

1. **DSpark/MTP 投机**：模型自带 3 个 nextn 层 + markov/confidence head。预计 3-5 token/步（接受率 80%），对 8GB 单机是最大加速项。
2. **异步流水**：CPU 侧（采样/预取调度）与 GPU 计算重叠；Python 串行是当前 decode 主要开销（5 层 86ms/层 vs 稳态 profile 10ms/层）。
3. **专家热度统计**：loading.ExpertCache 的 hit/miss 已可用于槽位 pin 与预取调参。
4. **batch-invariant 采样**：多 agent 时 per-slot 隔离（server 已有 slot）。
5. **结构化输出**（xgrammar）：后续接入 server 的 response_format。

## 关键差异（不要照搬）

- DeepInfer 假设模型全驻 GPU（SM100 141-192GB/卡）且 TP/EP 跨卡；tinyinfer 必须把 296GB 专家放 NVMe 流式、非专家 17.7GB 放 RAM、只有 8GB VRAM。
- 每 token 4.5GB 冷读是系统上限：加速应减少 token 数（投机）或提高命中率（预取/缓存），而不是盲目追 kernel 峰值。
- DeepInfer 的 kernel 面向 SM100 的 tcgen05/FP4 MMA；sm_120 实测：fp8 32x32 38-47 TFLOPS、fp4 23-29 TFLOPS、sparse_attn 62-70µs/层（decode）。
