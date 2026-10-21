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

## 状态（2026-10-22）

- [x] P0.1 环境：Windows 11 + Python 3.13 + torch 2.9.0+cu130（无 WSL，直接 Windows CUDA）
- [x] P0.2 工具链：tilelang 0.1.15（vcvars64 + CUDA_HOME）、uv、官方 kernel.py 复用
- [x] P1 权重侦察与资源预算：510GB 权重结构、专家 4.5GB/token、显存预算 0.44GB/层
- [x] P2 单组件参考前向：attention/MoE/HC/compressor/indexer 全部单测
- [x] P3 HF 直接加载器：pread 直读（修复 mmap 多分片崩溃）、fp8/fp4/engram
- [x] P4 分层执行：7 层常驻 + 33 层流式、RAM 专家缓存（弹性）、VRAM 槽状态机
- [x] P5 端到端：**全 40 层真实权重**（42 分片 SHA256 校验）prefill/decode/生成通过
  - decode **0.20 tok/s**（5.0s/token，8GB 卡 + 33 流式层）；生成语义正确（"Hello, I am DeepSeek"）
  - OpenAI API（多 slot/SSE/function calling 回显）端到端通过
- [x] 测试 68 passed / 文档（benchmarks/kernels/loading/api/mtp_design/deepinfer_ref）/ bootstrap / git

- [x] P6 HF 仓库直载：`Engine.from_hf(repo_id | hub snapshot | 平铺目录)`——模型统一放 `E:\Models\<repo-name>`（HF 平铺布局），适配器自动解析；也接受 HF hub 缓存快照。
- [x] P7 MTP：3 stage 权重加载 + 官方 DFlash 契约前向链（main_hiddens/forward_embed/滑窗 attention/MoE/HC/forward_head=主head+markov+confidence）全部跑通（`scripts/e2e_mtp.py`）。
  - 投机解码实测负结果：accept_rate 0%、比常规 decode 慢（draft 分布待参考数值验证），已记录 docs/mtp_design.md，暂不启用。
- [x] P8 回归：pytest **68 passed / 1 skipped**（含 attention/compressor/server/quant）。

**进行中**：engram 表下载（00048 分片进行中，完成后 `E:\Models\DeepSeek-V4.1-Flash-1017` 全量终验）；流式层 H2D 与专家 IO 进一步重叠（显存预算约束）；MTP 投机的参考数值对齐。
## 优化分级（O1 → O4）

像编译器的 -O 级别一样，tinyinfer 的加速按可验证的阶梯推进；每一级都保留上一级的正确性基准：

| 级别 | 内容 | 状态 |
|---|---|---|
| **O1 正确性** | 纯 PyTorch 参考实现：HF 直接加载、MLA + sink、压缩 KV/Indexer、Hyper-Connections、MoE（fp8/fp4 反量化） | ✅ 已实现 + 数值单测 |
| **O2 融合** | tilelang kernel：RMSNorm+RoPE、SwiGLU、MoE combine、sparse-MLA gather | 研究中（kernel-researcher） |
| **O3 量化加速** | FP8 E4M3 / NVFP4（e2m1）tensor core 直接计算、KV 量化（8bit / 4bit+Hadamard / K8V4） | 研究中 |
| **O4 系统级** | 专家分层与 tier 交换、路由预测/预取、SSD streaming、KV streaming、投机解码（DSpark/MTP）、多 agent batching | 设计中（loading-researcher） |

## 团队分工

- **Lead**：架构与正确性——模型/注意力/MoE/HC 集成、端到端验证、记忆维护
- **kernel-researcher**：sm_120 算子矩阵（torch / tilelang / DeepGEMM / FlashInfer / SDPA）与原型
- **loading-researcher**：三层内存（VRAM/RAM/NVMe）预算、专家缓存与路由预测、权重选择加载
- **api-engineer**：OpenAI 兼容 API（多 agent 会话、SSE、slot）
- **data-manager**：完整仓库（510GB）下载、续传、校验与状态维护

## 参考思想来源

- DeepInfer：HF 直接加载器、EPLB、压缩 KV/Engram/MTP 的官方实现
- Strata（docs/DETAILS.md）：融合 int8 tensor-core、KV streaming、专家 tier 交换、PCIe 分担、O1-O4 式工程优化
- ds4 / flash-moe / llama.cpp #25294：SSD 专家流式与缓存策略

## 使用（生产路径）

### 1. 环境
```powershell
# 复用本机已装 torch 的环境（推荐，省下载）
.\scripts\bootstrap.ps1 -ReuseEnv D:\Develop\Other\python\.venv
# 或全新安装（torch 走 cu130）
.\scripts\bootstrap.ps1
```
tilelang kernel 需要 vcvars64 + CUDA_HOME（prototypes/kernels/run_py.bat 已封装）。

### 2. 权重
模型放在 `cache/weights`（HF safetensors 原格式，无需转换）。
```powershell
uv run python scripts\download_repo.py      # 完整仓库（断点续传）
uv run python scripts\download_progress.py  # 进度
```

### 3. 起服务（OpenAI 兼容）
```powershell
.venv\Scripts\python.exe -m tinyinfer.server --model-dir cache\weights --tier balanced --device cuda --port 8000
curl http://127.0.0.1:8000/v1/models
```
- 端点：`/v1/models`、`/v1/completions`、`/v1/chat/completions`（SSE）、`/v1/slots`、`/health`
- 多 slot：`X-Session-Id` 绑定 KV 快照；`reset_slot` 清空
- 多模型：`--extra-model NAME=DIR`，`/v1/models` 动态列出

### 4. 基准
```powershell
.venv\Scripts\python.exe scripts\bench.py --max-layers 6 --device cuda --tier balanced
```

## 当前实现状态

| 模块 | 状态 |
|---|---|
| HF safetensors 直接加载（fp8/fp4/engram 行读） | ✅ |
| MLA window + 压缩 KV/Indexer（分块 prefill/decode） | ✅ 真实权重（L0-L5） |
| MoE（fp4 官方 kernel / GPU dequant） | ✅ |
| Hyper-Connections | ✅ |
| Engram（L1/L14，按行读表） | 代码就绪，表分片（196GB）待下载 |
| 层驻留 + 流式（GPU/RAM 预算） | ✅ |
| 多 slot KV 快照 | ✅ |
| OpenAI API + SSE + function calling 回显 | ✅ |
| MTP/DSpark 投机 | 设计完成（等 MTP 分片） |
| tier 分档（conservative→extreme） | ✅ 开关定义 |

## 限制
- 全 40 层需等完整下载（当前 L0-L5）。
- Engram 表（196GB）为最后下载项；缺表时 L1/L14 跳过 engram（数值偏差，文档标注）。
- Windows 上 tilelang 依赖 vcvars64；无 CUDA 环境自动回退 PyTorch 参考。
- per-slot KV 是快照隔离，尚不支持跨请求 prefix 复用。
## 实测性能（RTX 5060 Laptop 8GB + Ryzen 9 9955HX，2026-10）

| 配置 | prefill | decode | 说明 |
|---|---|---|---|
| 5 层 balanced（GPU，fp8 常驻 + fp4 专家 kernel） | 16 tok / ~0.7s | **~180ms/token（5.4-5.7 tok/s）** | 预测预取 + 状态机槽 + 融合 dequant |
| 40 层（外推） | — | ~1.4-1.6s/token（0.6-0.7 tok/s） | 25 层常驻 + 15 层流式 |

优化点：
- fp8→bf16 融合反量化 kernel（wo_a 2.0→0.48ms，4.2x）
- CPU 上下文预测器（token/bigram/trigram 表，命中 94.9%）+ 异步 RAM 预取
- VRAM 槽状态机（FREE/RESERVED/COPYING/STAGED/IN_USE + 事件完成通知，并发安全）
- 双 staging（pinned）+ 专用 copy stream（H2D 26.6GB/s，与计算重叠）

理论分析：瓶颈是专家冷读（4.5GB/token @ SSD）；命中 90% 时 IO 下限 ~100ms/token（10 tok/s），GPU 算力远未饱和。
