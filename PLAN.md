# tinyinfer 路线图

## 总目标
在 8GB VRAM + 34GB RAM + NVMe 的笔记本上，直接从 HF safetensors 加载 DeepSeek-V4.1-Flash-1017 并生成 token。

## P0 环境地基
- P0.1 WSL2 Ubuntu 26.04（完成：GPU 透传 OK，内存 34GB，清华源）
- P0.2 基础工具链：build-essential/git/curl/cmake/ninja；uv；Python 3.12
- P0.3 CUDA Toolkit 13.x（WSL 内，nvcc）
- P0.4 torch 2.13+cu132 + DeepInfer Python 依赖
- P0.5 DeepInfer vendor 构建（DeepGEMM/FlashMLA/DeepEP/DeepSelect），尝试 SM120 补丁
- **验收**：`python model.py`（官方最小实现自测）或 DeepInfer 单测在 sm_120 上跑通（随机权重，不下载 475GB）

## P1 权重侦察与资源预算
- 解析 `model.safetensors.index.json` + 分片 safetensors header（Range 读，不下载全量）
- 张量分类：非专家 / 专家 fp4 / Engram 表 / MTP / vision
- 用 DeepInfer `hugging_face/names.py` 做覆盖映射校验（哪些张量可被 DeepInfer 识别）
- 输出资源预算表：VRAM 驻留集、RAM 驻留集、NVMe 流式集、单 token 激活专家数/字节数
- **验收**：`scripts/probe_checkpoint.py` 输出完整统计 + 兼容性报告；给出"哪些部分可以全驻留、哪些必须流式"的结论

## P2 单组件参考前向（正确性优先）
- CPU/PyTorch 参考实现：MLA 注意力（含 SWA 压缩）、Engram 查找、MoE 路由与专家计算、Hyper-Connections
- 与官方 `inference/model.py` 的行为对齐（小张量、随机权重、逐层比对）
- **验收**：单层前向数值一致（容差内）；Engram 行读取路径正确

## P3 HF 直接加载器
- 按 index 分片流式加载，支持"层窗口"（只加载部分层）与按需张量
- 专家聚合（HF per-expert → 运行时 3D packed）、fp8/ue8m0 scale 处理、Engram 行范围读取
- 内存放置策略：VRAM 活跃集 / RAM / 文件 mmap
- **验收**：不下载全量（只取需要的分片）即可构建可运行的前缀模型，并与 P2 参考实现对齐

## P4 分层执行
- 专家流式 GEMM：NVMe → RAM/VRAM 的按需加载 + 缓存策略（参考 PR #25294 的 O_DIRECT 与槽位缓存、flash-moe 的 OS page cache 结论）
- Engram 表按行读取（DeepInfer `_load_engram_tables` 思路）
- CPU offload 管理、KV 压缩、小上下文贪心生成
- **验收**：一段短 prompt 能生成 token（速度不论），内存不 OOM

## P5 端到端
- 全量下载 475GB（视网络/流量策略）
- 跑通完整 prompt，记录 prefill/decode 速度、内存曲线
- 与官方精度做小样本对比（GSM8K 子集等）
- **验收**：可复现的推理记录 + 基准报告

## P6 结题
- 技术报告：架构、移植清单、基准、失败经验、剩余优化空间
- 结论沉淀到 `E:\Develop\Deepseek\记忆.md`

## 风险与对策
| 风险 | 对策 |
|---|---|
| DeepGEMM/FlashMLA 在 sm_120 无 kernel | 先用 PyTorch/FlashInfer 参考实现跑通正确性；再按 vLLM PR #43477 思路补 kernel |
| torch 2.13+cu132 / CUDA 13.2 下载慢 | 走国内镜像/分时段；先装最小集 |
| 475GB 下载成本 | 阶段化：P1-P4 只取所需分片，P5 再全量 |
| 47GB RAM 装不下非专家权重 | 按层/按张量流式；非专家也可分层 |
| 8GB VRAM 连单层都紧张 | 激活/KV 压缩；分块计算 |

## 优化分级 O1–O9（扩展版：为 500B 模型压进 8GB 笔记本）

> 目标不是"跑得动一般模型"，而是让 **DeepSeek-V4.1-Flash（~510GB, 384 专家/6 激活）** 在
> 8GB VRAM + 47GB RAM + NVMe 上产生可用 token。O1–O4 是常规优化；**O5–O9 是让 500B 可行的关键**。

| 级别 | 名称 | 核心手段 | 验证方式 |
|---|---|---|---|
| O1 | 正确性 | PyTorch 参考实现、HF 直接加载、数值对齐 | 已完成的单测（RoPE/sink/fp8/fp4） |
| O2 | 算子融合 | tilelang：RMSNorm+RoPE、SwiGLU、MoE combine、sparse MLA、gather-scatter | ms/op、与 O1 数值一致 |
| O3 | 量化加速 | FP8 E4M3 / NVFP4 tensor core 直接计算；KV 8bit/4bit(Hadamard)/K8V4；分组 scale | tokens/s、PPL 变化 |
| O4 | 内存分层 | 非专家 17.7GB 驻 RAM、专家 296GB 流式、engram 196.6GB 行读、KV 分层 | 峰值内存、命中率 |
| **O5** | **投机与预判** | DSpark/MTP 内置 draft（模型自带 3 个 MTP 层 + DSpark 头）、n-gram draft、树形投机、**speculative prefill**（长 prompt 稀疏早停）、CPU 侧 draft 采样/验证 | 接受率、tokens/成本、TTFT |
| **O6** | **CPU 辅助决策** | CPU 计算 router/gate（5120×384 很小）提前 1–2 层预测专家集合 → 预取；CPU 做采样与验证（避免 GPU→CPU 同步）；CPU 决定 PCIe/磁盘分工；CPU 侧 n-gram/engram 查询 | 预测命中率、同步开销、CPU 利用率 |
| **O7** | **IO/加载协同** | 双缓冲预取（GPU 算 attention 时 NVMe 读下一层专家）、O_DIRECT/overlapped IO、NVMe 多队列、fp4 原样搬运、batch 内专家去重、热专家 pinning + 冷专家 eviction（LRU/ARC/S3-FIFO） | 有效磁盘带宽、miss 延迟、tokens/s |
| **O8** | **跨 agent 协同** | 多会话共享专家热度、batched MoE（多 token 汇聚一次 I/O）、prompt cache 复用 KV/专家状态、slot 级缓存归属 | 并发吞吐、缓存命中、内存预算 |
| **O9** | **异构流水线** | GPU(L2)/RAM(L3)/NVMe(容量) 三级流水：attention 在 GPU、专家在 CPU+GPU、SSD 供数；PCIe 分担与计算-IO 重叠的调度器；"GPU 只做 attention + 部分专家，CPU 主力"的混合执行 | 端到端 tok/s、各资源利用率、稳定性 |

### 500B 模型的关键算式（决定一切）
- 每 token 激活：6/384 路由专家 + 1 共享 ≈ **~10–12B 参数**；fp4/fp8 混合下每 token 需读约 **3–5GB** 专家权重。
- 8GB VRAM：放不下整层专家（每层 ~7.2GB Q2），只能放 **专家缓存 + attention 层权重 + KV**。
- 47GB RAM：非专家 17.7GB 全驻 + 专家页缓存 ~20GB + KV/激活。
- 结论：**瓶颈是"每 token 4.25GB 的专家读取 + 非专家驻留"**，因此 O5–O8（投机减少 token 数、CPU 预测提前 I/O、跨 agent 合并 I/O、流水重叠）才是可行性的核心；O2/O3 是单步成本，O5/O6/O8 是总量削减。

### 每级都要保留的可测指标
每 token 磁盘字节、专家缓存命中率、预取提前量（层数）、CPU/GPU 空闲率、TTFT、tokens/s、PPL 回归。

## 实现进度（2026-10-21 01:15）
- ✅ HF safetensors 直接加载（fp8/fp4/engram 行读表）；L0-L5 分片完整可跑。
- ✅ MLA window + 压缩 KV/Indexer（统一支持任意 start_pos/s：分块 prefill + decode）。
- ✅ MoE：官方 fp4 kernel（M>=32）/GPU dequant 回退；ExpertStore（ARC + 24 VRAM 槽）。
- ✅ Hyper-Connections（Sinkhorn）；Engram 代码就绪（表分片 196GB 最后下载）。
- ✅ Engine：chunked prefill、层驻留预算（bf16 3 层 / fp8 33 层）、GPU 流式、slot snapshot/restore、stream_generate。
- ✅ OpenAI 兼容 server（多 slot/session、SSE、function calling 回显）驱动 Engine。
- ✅ bench.py（prefill/decode/TTFT/内存）+ tiers（conservative→extreme）+ adapters（HF/GGUF 预留）。
- ⏳ MTP/DSpark 投机（等 model-00044..46 分片；官方结构已读，初版因质量不足已删除）。
- ⏳ Engram 表（model-00047/48，196GB）。
- ⏳ 全 40 层端到端（下载中，7.75%，ETA ~16h）。
- 已知问题：CPU 参考路径 ~0.3-0.5 tok/s（无优化）；GPU balanced 5 层实测待确认（两个 bench 进程曾并发争用 GPU）。
