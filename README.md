# tinyinfer

从零实现的本地推理引擎。目标不是"能跑"，而是**在显存远小于模型的机器上跑出可用输出**。

引擎负责与机器相关的一切：从磁盘流式读权重、层驻留与预取、会话 KV 归属、CUDA graph
捕获。模型只提供与架构相关的一切：权重布局、一层的前向、embedding 与 head。
两者之间的契约在 `src/tinyinfer/protocol.py`。

这就是为什么同一套代码既能跑 510GB 的 DeepSeek，也能跑 51.8GB 的 Qwen。

## 当前实测

| | |
| --- | --- |
| GPU | RTX 5060 Laptop, 8151 MiB, driver 621.01 |
| CPU / RAM | Ryzen 9 9955HX (16C) / 47.2 GB |
| Python | 3.13.7, torch 2.9.0+cu130, transformers 5.20.2 |

| 模型 | 权重 | 驻留 | 引擎路径 | 说明 |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | 1.4 GB bf16 | 24/24 全在卡上 | **180 tok/s** | CUDA graph，输出与参考逐 token 一致 |
| DeepSeek-V4.1-Flash-1017 | 510 GB | 10/40 层 | **1.22 tok/s** | AGGRESSIVE 档，流式 + 专家缓存 |
| Qwen3.8-27B | 51.8 GB | 2/64 层 | 逐层读入 | 64 层能逐层读入并跑完一遍，速度未测通 |

Qwen3.5-0.8B 走引擎：同一套代码，`engine.block(i)` 逐层驱动，整步捕获成一张 CUDA graph。
eager 驱动 49.9 tok/s，捕获后 180.4 tok/s（3.6 倍），十个 token 与 eager 逐位相同。

这台卡实测有效带宽约 343 GB/s，1.4GB ÷ 343GB/s = 4.1ms 是纯带宽下限，即 244 tok/s。

## 优化档位

四档，精度和能力的边界是硬性的：

| 档 | 语义 | 在这台 8GB 卡上 |
| --- | --- | --- |
| 0 `CONSERVATIVE` | PyTorch 参考路径，最慢但最容易核对 | 装不下 |
| 1 `BALANCED` | 开源栈的做法：整个 checkpoint 常驻显存，不流式 | **275 GiB 需求 > 6.9 GiB，明确拒绝** |
| 2 `AGGRESSIVE` | 本引擎的流式加载 + 专家缓存 + 预取 + MTP | **1.22 tok/s** |
| 3 `EXTREME` | 在 2 的基础上允许损失精度/能力（跳层、低秩、fast math、KV4） | 更快，但输出不再等价 |

1 和 2 精度相同、能力相同，区别只在**机制从哪来**：1 是"把模型全放进显存"这条路，
2 是"显存放不下时怎么跑"。所以 1 在小卡上会直接报错而不是硬撑——Windows 的 WDDM
遇到超配不会报 OOM，而是把溢出的显存换到内存里，结果是"能跑但慢一半"，这种假象比
失败更难查。3 是唯一允许牺牲精度的档，损失项都在 `tiers.py` 的 `approximations` 里。

## 用法

模型由**启动引擎的人**决定，客户端只连端口：

```powershell
.\start.ps1 -Model <path>                              # 前台
.\start.ps1 -Model <path> -Bg                          # 后台，日志进 serve.log
$env:TINYINFER_MODEL = <path>; .\start.ps1             # 或用环境变量
.\start.ps1 -Model "<pathA>;<pathB>"                   # 一个 checkpoint 跨盘分片

.\venv\Scripts\python.exe repl.py                     # 唯一客户端
```

引擎读 checkpoint 的 `model_type` 选择运行时路径（`adapters.open_adapter`）：

| 架构族 | 适配器 | 层 | 状态 |
| --- | --- | --- | --- |
| `deepseek_v41` | `HFDeepSeekV41Adapter` | MLA + 压缩器 + engram + 超连接 + MoE | 完整 |
| `qwen3_5` | `HFQwen35Adapter` | GatedDeltaNet 混合注意力 | 流式加载 + 层驻留就绪 |

两个模型实现同一个 `Backend` 协议，HTTP 层只看到 `GenerationRequest` 进、`TokenEvent` 出。

## 工程细节

**结构。** `engine.py` 不知道层里在算什么，它只做四件事：把层在 CPU/GPU 之间搬、
按需从磁盘读权重、维护预取环、持有会话状态。`qwen_model.py` / `model.py` 提供层本身。
新增一个架构族只需要一个 adapter，不需要第二个引擎。

**显存不够，所以按需流式。** DeepSeek 的 30 层权重和 384 个路由专家放在磁盘上，
每 token 只读当前需要的部分：环深 2 的预取缓冲、按 token 路由结果的专家缓存
（14GB 上限，约 913 个专家块）。IO 下限由实测常数算出（专家块 18.8MB @ 4.5GB/s）。

**更大的模型用同一套办法。** Qwen3.8-27B 是 51.8GB，既装不进 8GB 显存也装不进 47GB 内存。
引擎在 `meta` 设备上建出 64 层的模块树（RSS 0.57GB，零权重），然后按层从 shard 读入：
一层 731MB、1.5 秒，跑完就可以搬走腾地方。层 0 的 14 个张量与参考加载器逐位一致。

**小模型反过来，瓶颈变成 kernel 开销。** Qwen3.5-0.8B 全在卡上，1.4GB 一次读完约 4.1ms，
但一次解码有一千多个 kernel，每个在 CUDA graph 里仍有约 1.2μs 固定成本。所以：

- 整个 decode step 捕获成一张 CUDA graph，一次 replay 代替一千次 launch
- KV 用 `StaticCache`：transformers 的 `DynamicLayer.update` 每 token `torch.cat`，
  图捕获会把这个分配冻结，实测 KV 长度卡在 16 不再增长、模型永远输出同一个词
- GatedDeltaNet 的 delta rule / 4-tap 卷积 / gated RMSNorm 都是手写的 CUDA kernel
  （`native/gdn.cu`），因为 `causal_conv1d` 和 `flash-linear-attention` 在本环境装不上
- RMSNorm 从 7 个 torch 算子合成 1 个 kernel，一步省掉约 280 个 launch：7.07 → 6.41ms

**Kernel 通过 `nvcc -cubin` 编译成 cubin，运行时用 `cuModuleLoad` 直接加载**，
不走 torch C++ 扩展（省去编译链依赖，改一个 kernel 是秒级而不是重编整棵扩展树）。
cubin 是按架构编译的产物，不入库：`scripts/build_native.ps1` 从 GPU 探测计算能力并
编译全部 `*.cu`。

有成熟实现就抄进本仓库，不从头写。DeepSeek 这条路径上的四个热点全部是这么来的：

| 本仓库 | 上游 | 数值对拍 |
| --- | --- | --- |
| `native/vllm_rms_norm.cu` | vLLM `layernorm_kernels.cu` | 与参考逐位相同 |
| `native/vllm_silu_and_mul.cu` | vLLM `activation_kernels.cu` | 1-2 bf16 ulp |
| `native/vllm_topk_softplus_sqrt.cu` | vLLM `moe/topk_softplus_sqrt_kernels.cu` | idx/权重逐位相同 |
| `native/marlin/` | vLLM `quantization/marlin/` | GEMV 与 repack 逐位相同 |

源码在仓库里，可以改、可以裁剪（Marlin 只实例化实际用到的那个模板，cubin 50 KB
而不是整套 kernel_selector 的几 MB），参数顺序按本引擎的习惯定。同一个模板实例在
cubin 里比原来的 C++ 绑定更快：Marlin 的 18 个参数用预分配参数槽打包，实测
19.7μs vs 20.2μs。

## 正确性

同一个稀疏机制在两种规模上的数字要能对上，所以仓库里每个"更快"都伴随一个"一致"的验证：

- Qwen 与 transformers 原生实现逐 token 对比：五个 prompt 各 48 token，四个完全一致，
  第五个在第 9 个 token 分叉（手写 RMSNorm 差 1 ulp，累积到贪心 argmax 翻转）
- Qwen3.8-27B 的层权重与参考加载器 `max abs diff = 0.0`
- DeepSeek 与厂商栈 DeepInfer 的分量对比：窗口层注意力 0.000000、fp4 专家 0.000000、
  门控 0.000000、超连接 0.000000、RoPE 9.7e-06，压缩层 0.87% / 1.42%

## 构建

需要 CUDA toolkit（`nvcc`）和 VS Build Tools：

```powershell
.\scripts\bootstrap.ps1             # 建 venv、装依赖
.\scripts\build_native.ps1          # 编译全部 kernel 到 cubin
.\scripts\check.ps1                 # 语法 + 单测（99 个用例，11 秒）
```

Linux 用 `scripts/bootstrap.sh`。

## 目录

```
src/tinyinfer/
  protocol.py      引擎与模型之间的契约（层、模型、KV）
  engine.py        流式、预取、驻留、会话状态、CUDA graph
  adapters.py      权重布局 + 按 model_type 分派
  config.py        从 config.json 解析
  kv.py            KV/状态容器（各架构自己的实现）
  server.py        OpenAI 兼容 API + Backend 协议
  model.py         DeepSeek 层（MLA / 压缩器 / engram / 超连接）
  qwen_model.py    Qwen 层（meta 骨架 + 按层读入）
  hf_graph.py      通用 CUDA graph 解码器
  native/*.cu      手写 kernel（含从上游抄入并裁剪的 vllm_*.cu）
  native/marlin/   Marlin fp4 GEMV + 权重 repack（抄入，仅实例化所用配置）
  gdn*.py          Qwen 线性注意力
repl.py            命令行客户端
scripts/           构建与工具
tests/             单元测试
docs/              设计文档与实测记录
```

## 许可

Apache-2.0
