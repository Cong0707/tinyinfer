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