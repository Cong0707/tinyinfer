# vLLM 传统引擎基线（Windows / RTX 5060 Laptop）

所有数字来自 **vLLM 0.33.0 官方 C++/CUDA 源码在本机用 MSVC + nvcc 13.0 手工编译出的扩展**
（`_C_stable_libtorch.pyd` 22.5 MB、`_moe_C_stable_libtorch.pyd` 11.8 MB），
不是复刻、不是估算、不是 tilelang fallback。

- GPU: RTX 5060 Laptop, sm_120, 26 SM, 8 GB GDDR7
- 纯读带宽实测上限: **295 GB/s**
- 环境: Windows / MSVC 19.44 / CUDA 13.0 / torch 2.14.0+cu130 / Python 3.13
- 计时: CUDA event, warmup 5, iters 30

## 权重只有 41.9 MB 时的 fp4 专家 GEMV（N=16384, K=5120, group=16, nvfp4）

| M | vLLM Marlin 时间 | 有效带宽 |
|---:|---:|---:|
| 1 | 158.3 us | **264.9 GB/s** |
| 8 | 152.8 us | **274.5 GB/s** |
| 16 | 152.8 us | **274.5 GB/s** |
| 32 | 190.1 us | 220.6 GB/s |
| 64 | 347.1 us | 120.8 GB/s |
| 128 | 648.8 us | 64.7 GB/s |
| 256 | 1293.7 us | 32.4 GB/s |

**M=1 达 265 GB/s，是纯读上限 295 GB/s 的 90%。**

## 对照：我们自己的实现（同一场景）

| 实现 | 时间 | 带宽 | 相对 vLLM |
|---|---:|---:|---:|
| vLLM Marlin (M=1) | 158.3 us | 265 GB/s | 1.00x |
| tinyinfer `gemv_fp4_fast` | 657 us | 59 GB/s | **4.15x 慢** |
| 引擎 tilelang fallback | 1044 us | 40 GB/s | 6.6x 慢 |

差距来源（已用 ncu 定位）：我们的 kernel DRAM 利用率只有 19%、SM 利用率 52%、
`lg_throttle` 每秒 11.37 次——标量访存、没有 cp.async 四级流水、没有 ldmatrix 分片。

## 元素级 / 量化 kernel

| kernel | 配置 | 时间 | 带宽 |
|---|---|---:|---:|
| `rms_norm` | M=4096 K=4096 bf16 | 209.2 us | **321 GB/s** |
| `silu_and_mul` | M=4096 N=4096 bf16 | 320.2 us | **314 GB/s** |
| `scaled_fp4_quant` | 4096x4096 bf16 -> nvfp4 | 120.1 us | — |

这三个都跑在 315-320 GB/s，同样贴着带宽上限。

## mxfp4（e8m0 scale）路径 —— 这才是 checkpoint 的格式

上面那张表是 nvfp4（带 global scale）。DeepSeek 的专家权重存的是 **mxfp4**：
每 32 个权重一个 e8m0 指数，没有 global scale。同一条 Marlin kernel 走 mxfp4 更快：

| M | vLLM Marlin (mxfp4) | 有效带宽 |
|---:|---:|---:|
| 1 | 142.9 us | **293.6 GB/s** |
| 8 | 135.6 us | **309.3 GB/s** |
| 16 | 138.6 us | 302.7 GB/s |
| 32 | 201.2 us | 208.5 GB/s |
| 64 | 332.4 us | 126.2 GB/s |

309 GB/s 已经超过这张卡 295 GB/s 的纯读上限，因为部分权重命中 L2。

## tinyinfer 接入后的实测

`src/tinyinfer/gemv_fp4_marlin.py` 在加载时把权重 repack 成 Marlin 布局（一次性 152 ms），
之后每次调用就是一条 fused kernel：

| 路径 | 时间 | 带宽 | 相对手写 kernel |
|---|---:|---:|---:|
| tinyinfer `fp4_linear` -> Marlin | **136.2 us** | **308.0 GB/s** | **4.82x** |
| 手写 `gemv_fp4_fast` | 657 us | 59 GB/s | 1.00x |

数值已对拍：手工解码 E2M1（低 nibble 是偶数下标，高 nibble 是奇数下标，
e8m0 指数减 127 再 2^x）与 Marlin 输出的最大相对误差 **4.7e-3**，是 bf16 激活的精度量级。

## MoE 路由 / 辅助 kernel（受 launch 开销支配）

| kernel | 配置 | 时间 |
|---|---|---:|
| `grouped_topk` | T=1, E=256, topk=8 | 17.98 us |
| `grouped_topk` | T=128 | 14.72 us |
| `topk_softmax` | T=1, E=256, topk=8 | 10.94 us |
| `topk_softmax` | T=512 | 15.00 us |
| `moe_align_block_size` | T=1, E=256, block=128 | 19.94 us |
| `moe_align_block_size` | T=128 | 15.66 us |

这些不是带宽问题，是 **10-20 us 的 kernel launch 地板**。decode 一个 token 要跑几十条
kernel，把这类小 kernel 融合掉比优化它们的算术更有价值。

| `moe_sum` | T=1, topk=8, H=5120 | 11.47 us |
| `moe_sum` | T=128 | 10.38 us |

## 端到端：一个 fp4 专家的 SwiGLU FFN（3 个 GEMV，M=1）

| 路径 | 时间 | 每 GEMV |
|---|---:|---:|
| `Expert.forward_fp4` -> Marlin | **428.7 us** | 142.9 us |
| 同一 FFN，旧的手写 GEMV | 1971 us | 657 us |

4.6x。三个 GEMV 里 gate 和 up 的下投影都是 40 MB 权重，全部贴着带宽上限。

## 我们自己的元素级 kernel vs vLLM

| kernel | 我们的（改前） | 改后 | vLLM | 结论 |
|---|---:|---:|---:|---|
| rms_norm M=1 K=5120 | 33.87 us | **18.22 us** | 23.08 us | 反超 |
| rms_norm M=8 K=5120 | 67.03 us | **13.25 us** | 9.70 us | 1.37x（都是 launch 地板） |
| rms_norm M=4096 K=4096 | 215.9 us | 222.97 us | 220.12 us | 持平 |

改前的原生 kernel 每行只开 32 个线程的 block，M=8 时就暴露了。

## 复现方法

```python
import torch, vllm._C_stable_libtorch
out = torch.ops._C.marlin_gemm(
    a, None, qweight, None, scales, None, global_scale, None, workspace,
    562949953487106,  # scalar_types.float4_e2m1f.id
    M, 16384, 5120, False, False, False)
```

```
marlin_gemm fp4 M=  1:    158.3 us    264.9 GB/s
```