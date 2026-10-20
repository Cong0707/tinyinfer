# tinyinfer

> 用各种技术实现看似不可能的推理。
>
> Run an impossible model on an impossible machine.

## 目标

在一台消费级笔记本上**直接加载并运行** DeepSeek-V4.1-Flash-1017：

| 约束 | 数值 |
|---|---|
| GPU | RTX 5060 Laptop, 8 GB VRAM, Blackwell sm_120 |
| RAM | 47 GB（WSL 可用 34 GB） |
| 模型 | 475 GB（HF safetensors，fp8 权重 + fp4 专家 + 189 GB Engram 表） |

这不是要训练，也不是要生产级服务 —— 是要证明：**通过分层内存、按需流式、kernel 移植和 DeepInfer 的前沿算法，8 GB 显存的笔记本也能跑 475 GB 的旗舰模型**。

## 设计原则

1. **直接加载 HF 权重**：不转 GGUF、不预先 requantize。直接读 `model.safetensors.index.json` + 分片，沿用 DeepInfer 的 `hugging_face/weights.py` 覆盖校验（专家聚合、Engram 表按行读取）。
2. **复用 DeepInfer**：以深寻官方推理框架为上游，复用其模型定义（MLA / YWCO / SWA / indexer / Engram / MoE / Hyper-Connections / DSpark / MTP）与引擎组件；只替换"装不下"的部分。
3. **分层内存**：VRAM（8 GB）→ RAM（34 GB）→ NVMe（2.4 TB）三级。专家权重与 Engram 表按 token/行按需从 NVMe 拉取，参考 llama.cpp PR #25294（MoE 磁盘流式）与 flash-moe（"Trust the OS"）的经验。
4. **消费级 kernel 适配**：把 DeepInfer 的 SM100 kernel 栈移植到 sm_120 —— 参考 vLLM PR #43477（SM120 DeepSeek V4）、DeepGEMM PR #324（SM120 MXFP4）、FlashMLA sm120 fork、FlashInfer sparse MLA PR #3395。
5. **可验证**：每个阶段有明确验收，不追求速度，追求正确性与可运行；所有实验可复现（脚本 + 日志 + 记忆沉淀）。

## 架构

```
             ┌──────────────── tinyinfer ────────────────┐
             │  DeepInfer 算法层（直接复用/裁剪）          │
             │  MLA · YWCO/SWA · Engram · MoE · HC ·     │
             │  DSpark/MTP · 调度 · 结构化输出            │
             ├───────────────────────────────────────────┤
             │  分层内核适配层                            │
             │  tilelang(mma/convert) · DeepGEMM SM120 · │
             │  FlashInfer sparse MLA · PyTorch fallback │
             ├───────────────────────────────────────────┤
             │  内存/存储编排层                           │
             │  VRAM 活跃集 · RAM 非专家权重 · NVMe 流式  │
             │  （专家 fp4 按需 GEMM、Engram 行读取）     │
             ├───────────────────────────────────────────┤
             │  HF 直接加载层（safetensors mmap + index） │
             └───────────────────────────────────────────┘
```

## 目录

```
tinyinfer/
├── README.md          # 本文件：理念、约束、架构
├── PLAN.md            # 阶段计划与验收标准
├── src/tinyinfer/     # 核心代码
├── scripts/           # 侦察、基准、工具脚本
├── environment/       # WSL/环境配置
└── cache/             # 权重与中间产物（大文件，可清理）
```

## 参考项目

- [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) —— 官方大规模推理框架（算法与 kernel 来源）
- [antirez/ds4](https://github.com/antirez/ds4) —— V4.1 单机引擎与 SSD streaming 经验
- [danveloper/flash-moe](https://github.com/danveloper/flash-moe) —— 397B 模型在 48GB MacBook 上的方法
- [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) —— MoE 专家磁盘流式
- [vLLM PR #43477](https://github.com/vllm-project/vllm/pull/43477) —— SM120 上的 DeepSeek V4

## 状态

- [x] P0.1 环境：WSL2 Ubuntu 26.04（GPU 透传确认）
- [ ] P0.2 工具链：CUDA / Python 3.12 / torch cu132 / DeepInfer vendor
- [ ] P1 权重侦察与资源预算
- [ ] P2 单组件参考前向
- [ ] P3 HF 直接加载器
- [ ] P4 分层执行
- [ ] P5 端到端
