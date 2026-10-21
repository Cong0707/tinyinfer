# tinyinfer 基准（RTX 5060 Laptop 8GB / Ryzen 9 9955HX / 47GB RAM）

测量日期 2026-10-21/22，模型 DeepSeek-V4.1-Flash-1017（HF safetensors 原格式，逐层下载中）。
所有数据来自真实权重（非合成），balanced 档（fp8 常驻 + fp4 专家 kernel + 学习型预取）。

## 0. 全 40 层（完整模型，2026-10-22 实测）

| 项 | 实测 | 条件 |
|---|---|---|
| 全 40 层加载 | 1.6s（懒构造） | 40 层分片 SHA256 校验通过 |
| prefill 8 tok（冷） | 54.6s | 含 40 层 Block 构造 + JIT 首编 |
| decode | **5882 ms/token（0.17 tok/s）** | 7 常驻 + 33 流式，vram 峰值 8122MB |
| 生成 | " Deep..  I am Deep." | 修复 L20 ratio=1 kv_source 的 src_ratio 后语义正常 |
| 每层成本 | MoE 56ms/层（72%）、attn 3ms/层、HC 1.7ms/层 | 专家 IO + fp4 kernel |

## 1. 已验证的层数 vs 性能

| 层数 | 完整分片 | decode | prefill | GPU 常驻 | VRAM 峰值 | 备注 |
|---:|---|---:|---:|---:|---:|---|
| 5 | 00003-00007 | 5.4-5.8 tok/s | 16 tok / 0.7s | 5 全常驻 | 6.0 GB | 早期基线 |
| 7 | 00009 | 4.7 tok/s | 16 tok / 5.3s | 7 全常驻 | 6.2 GB | |
| 8 | 00010 | 2.4-3.1 tok/s | - | 8 全常驻 | 7.2 GB | pread 修复后 |
| 19 | 00016 | 0.96-1.15 tok/s | 8 tok / 14s | 7 常驻+12 流式 | 6.2 GB | 稳定里程碑 |
| 24 | 00021 | 0.55 tok/s | 8 tok / 18.7s | 7 常驻+17 流式 | 6.2 GB | |
| 25 | 00027 | 0.41 tok/s | 8 tok / 19.5s | 7 常驻+18 流式 | 6.2 GB | 下载中 |

趋势：每增加一个流式层 ~+80-100ms/token（权重 H2D ~27ms + 专家 IO ~40ms + 计算 ~30ms）。

## 2. 单层分解（19 层模型、真实权重、decode b=1,s=1,kv=129）

| 阶段 | ms/层 | 说明 |
|---|---:|---|
| attention（window 层） | 2.0（低显存）/ 8.4（超显存分页） | 显存超限使 attn 慢 4x |
| 其中：wq_a/wq_b/wkv GEMM | 0.05 / 0.28 / 0.05 | fp8 gemm（M=1，官方 kernel） |
| compressor + indexer（压缩层） | 0.16 + 0.51 | fp32/bf16 小算子 |
| sparse_attn（window 128） | 0.03-0.05 | tilelang kernel |
| wo_a dequant（融合 kernel） | 0.48（原 2.0，4.2x） | 每 token 反量化 |
| MoE（6 专家） | 3-45（波动） | 专家 fetch/dequant/fp4_gemm |
| HC（融合，2 次 hc_mixes） | ~0.12（原 3.3，27.5x） | tilelang fused |
| 学习型预取器 | 命中 94.9%（上下文表） | CPU 侧 |

## 3. 关键瓶颈（40 层预测）

1. 流式层：40 层中 ~33 层需流式（8GB 显存放不下 17.6GB 权重），每 token 单向 H2D 6-7GB @26GB/s ≈ 250ms + 分配器开销。
2. 专家冷读：每 token 4.5GB（40 层 x 6 专家 x 18.8MB；下载完成后 SSD 4.5GB/s → 1s；RAM 全命中时 ~150ms H2D）。
3. 显存分页：超过 8.1GB 显存时 WDDM 分页使 attention 慢 4 倍——引擎预算（0.44GB/层，7.0GB 上限）已规避。
4. 磁盘竞争：下载中 SSD 仅 2.5-2.8GB/s（正常 4.5-8.8GB/s）。

40 层外推：0.3-0.4 tok/s（下载中）；下载完成后预期 0.4-0.5 tok/s；理论极限（专家命中 90% + 全异步）~10 tok/s（见 README 理论分析）。

## 4. API 基准（19 层，真实权重）

- /health、/v1/models、/v1/chat/completions（非流式/SSE）全部通过；usage 正确；4 token 请求 ~5-20s（含冷启动）。
- 多 slot KV 快照、function calling 回显通过。

## 5. 复现方法

    .venv\Scripts\python.exe scripts\bench.py --layers 0-18 --device cuda --tier balanced --prompt-tokens 8 --decode 6
    .venv\Scripts\python.exe scripts\sweep.py
    .venv\Scripts\python.exe scripts\full_check.py --layers 0-18 --bench

## 6. 限制

- 当前下载中（38%），40 层/engram/MTP 未全部就位；engram 表（196GB）为最后下载项。
- 流式槽双缓冲未启用（显存预算边缘）；状态容器已就绪。
- CUDA graph 已验证可捕获（replay 无收益——显存分页时 GPU 执行本身慢）。
