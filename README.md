# tinyinfer

从零实现的本地推理引擎。目标不是"能跑"，而是**在显存远小于模型的机器上跑出可用输出**：
DeepSeek-V4.1-Flash-1017 是 510GB，这台机器只有 8GB 显存和 47GB 内存。

## 当前实测

| | |
| --- | --- |
| GPU | RTX 5060 Laptop, 8151 MiB, driver 621.01 |
| CPU / RAM | Ryzen 9 9955HX (16C) / 47.2 GB |
| Python | 3.13.7, torch 2.9.0+cu130, transformers 5.20.2 |

| 模型 | 权重 | 速度 | 说明 |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 1.4 GB bf16 | **195 tok/s** | 全部驻留显存，CUDA graph 解码 |
| DeepSeek-V4.1-Flash-1017 | 510 GB | **1.2 tok/s** | 40 层 / 384 路由专家 top-6，逐层流式 |

Qwen 的 195 tok/s 是复用图之后的稳态（首次请求 77 tok/s，含 137ms 图捕获）。
这台卡实测有效带宽约 343 GB/s，1.4GB / 343GB/s = 4.1ms 是纯带宽下限，即 244 tok/s。

## 两个模型走同一条路径

```powershell
.\start.ps1 -Model "E:\Models\Qwen3.5-0.8B"            # 引擎启动时决定模型并预热
.\start.ps1                                            # 默认 DeepSeek-V4.1-Flash

.\venv\Scripts\python.exe repl.py                     # 唯一客户端
```

模型由**启动引擎的人**决定，客户端只连端口、只说模型名。引擎读 checkpoint 的
`model_type` 选择运行时路径：

- `qwen3_5*` → `QwenBackend`（混合线性注意力，CUDA graph）
- 其它 → `TinyModelBackend`（MoE + 流式专家）

两者实现同一个 `Backend` 协议，HTTP 层只看到 `GenerationRequest` 进、`TokenEvent` 出。

## 工程细节

**显存不够，所以按需流式。** DeepSeek 的 30 层权重和 384 个路由专家放在磁盘上，
每 token 只读当前需要的部分：环深 2 的预取缓冲、按 token 路由结果的专家缓存
（14GB 上限，约 913 个专家块）。IO 下限由实测常数算出（专家块 18.8MB @ 4.5GB/s）。

**Qwen 反过来，卡装得下，瓶颈变成 kernel 开销。** 1.4GB 权重一次读完约 4.1ms，
但一次解码有一千多个 kernel，每个在 CUDA graph 里仍有约 1.2μs 固定成本。所以：

- 整个 decode step 捕获成一张 CUDA graph，一次 replay 代替一千次 launch
- KV 用 `StaticCache`：transformers 的 `DynamicLayer.update` 每 token `torch.cat`，
  图捕获会把这个分配冻结，实测 KV 长度卡在 16 不再增长、模型永远输出同一个词
- GatedDeltaNet 的 delta rule / 4-tap 卷积 / gated RMSNorm 都是手写的 CUDA kernel
  （`native/gdn.cu`），因为 `causal_conv1d` 和 `flash-linear-attention` 在本环境装不上
- RMSNorm 从 7 个 torch 算子合成 1 个 kernel，一步省掉约 280 个 launch：7.07 → 6.41ms

**Kernel 通过 `nvcc -cubin` 编译成 cubin，运行时用 `cuModuleLoad` 直接加载**，
不走 torch C++ 扩展（省去编译链依赖）。

## 正确性

Qwen 的输出与 transformers 原生实现逐 token 对齐：五个 prompt 各生成 48 token，
四个完全一致，第五个在第 9 个 token 处分叉——手写 RMSNorm 比参考实现差 1 ulp，
累积到贪心 argmax 翻转，两个续写都合法。

DeepSeek 与厂商栈 DeepInfer 的分量对比：窗口层注意力 0.000000、fp4 专家 0.000000、
门控 0.000000、超连接 0.000000、RoPE 9.7e-06，压缩层 0.87% / 1.42%。

## 构建

需要 CUDA toolkit（`nvcc`）和 VS Build Tools：

```powershell
.\scripts\build_native.ps1          # 编译全部 kernel 到 cubin
.\scripts\bootstrap.ps1             # 建 venv、装依赖
```

Linux 用 `scripts/bootstrap.sh`。

## 目录

```
src/tinyinfer/
  server.py        OpenAI 兼容 API + Backend 协议 + 两个后端
  engine.py        DeepSeek 引擎：流式、专家缓存、KV 复用
  model.py         DeepSeek 层实现（MLA / 压缩器 / engram / 超连接）
  hf_graph.py      通用 CUDA graph 解码器（任何 transformers 模型可用）
  config.py        从 config.json 解析，按 model_type 分派
  adapters.py      权重布局适配器
  native/*.cu      手写 kernel
  gdn*.py          Qwen 线性注意力
repl.py            命令行客户端
scripts/           构建与工具
tests/             单元测试
docs/              设计文档与实测记录
```

## 许可

Apache-2.0
