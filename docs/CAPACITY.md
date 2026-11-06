# TinyInfer 实际承载能力（实测数据）

> 整理时间：2026-11-06
> 目的：提供引擎的真实能力数字，用于盈利能力计算
> 数据来源：本机实测（RTX 5060 Laptop）+ 用户提供的官方规格
> **本文不含任何「理论可达」的推测值。所有性能数字都是完整 kernel 的端到端实测。**

---

## 0. 一句话

**当前 prefill GEMM 在生产形状下是 19.5 TFLOPS = 官方 fp8 峰值的 14.7%。**

---

## 1. 什么是「真实成绩」

这一节先划清界限，因为很容易拿一个好看但不相关的数字当性能上限。

### 不算数的测量

```
76.1 TFLOPS —— 纯 mma.sync 循环
  连续 20000 条 mma，无访存、无 scale 乘法、无依赖链
  → 这测的是「指令能发多快」，不是「这个模型能算多快」
```

这个数字只说明硬件的指令吞吐，**不能作为 GEMM 的上限**。真正的 GEMM 必须：

* 从 global 读 A 和 B（真实访存）
* staging 到 shared memory 并双缓冲
* 每 32 个 k 应用一次 scale（每条 mma 后都要）
* 处理边界和 tail

### 算数的测量

完整的 `fp8_gemm_mma` kernel，输入是真实的 `[M, K]` fp8 权重 + block-32 e8m0 scale：

| 形状 | TFLOPS | 占官方 132.8 | 备注 |
| --- | ---: | ---: | --- |
| M=1024 N=2304 K=5120 | **19.5** | **14.7%** | 生产形状（专家 gate/up）|
| M=1024 N=5120 K=8192 | **19.5** | 14.7% | 生产形状（wo_b）|
| M=512 N=4096 K=1280 | **19.6** | 14.8% | 生产形状（wq_b）|
| M=4096 N=4096 K=4096 | 23.8 | 17.9% | 大形状（occupancy 更充分）|

**生产形状是 19.5 TFLOPS。这是引擎当前能交付的真实数字。**

---

## 2. 硬件官方规格（用户提供，权威）

### RTX 5090（sm_120）

| 项 | 值 |
| --- | --- |
| FP8 dense | **838.2 TFLOPS** |
| FP4 dense | **1676.0 TOPS** |
| 显存 | 32 GB |
| 带宽 | 1792 GB/s |
| block-scale MMA 指令 | 不支持（需 tcgen05/sm_100）|

### RTX PRO 6000（sm_120）

| 项 | 值 |
| --- | --- |
| FP8 dense | **838.2 TFLOPS**（与 5090 相同）|
| FP4 dense | 1676.0 TOPS |
| 显存 | **96 GB** |
| 带宽 | 1792 GB/s |
| block-scale MMA 指令 | 不支持 |

### H200（sm_90）

| 项 | 值 |
| --- | --- |
| FP8 dense | **1979.0 TFLOPS** |
| FP4 | **无**（Hopper 无 FP4 Tensor Core）|
| 显存 | 141 GB |
| 带宽 | 4800 GB/s |
| 卡间互联 | NVLink 4，900 GB/s |
| block-scale MMA 指令 | 不支持 |

### RTX 5060 Laptop（本机，sm_120）

| 项 | 值 |
| --- | --- |
| FP8 dense | **132.8 TFLOPS** |
| FP4 dense | 265.6 TOPS |
| 显存 | 8 GB |
| 带宽 | 384 GB/s |
| 实测 SM 数 | 26 |

**关键：5090 / PRO 6000 / 5060L 都是 sm_120，H200 是 sm_90。三张候选卡全部没有
block-scale MMA 指令，都必须自己用标量指令应用 scale。**

---

## 3. 为什么只有 14.7%（ncu 实测）

在 M=1024 N=2304 K=5120 上的 ncu 数据：

```
Warp Cycles Per Issued Instruction   17.87 cycles
  其中 5.6 cycles (31.5%) 等 shared memory scoreboard
No Eligible                          57.36%   <= 一半周期没有可发射的 warp
Eligible Warps Per Scheduler          1.02
ALU 流水线利用率                      21.1%   <= 最忙的流水线
Compute (SM) Throughput              43.20%
DRAM Throughput                       4.61%   <= 内存完全不是瓶颈
```

**根本原因：算术密度低。** 一个 k-block 只有 4 条 mma，配 36 条标量指令（scale 应用、
索引计算、边界判断）。mma 只占指令数的 **9%**。

### 为什么密度上不去（结构性原因）

```
模型的 weight_block_size = [32, 32]    <- 每 32 个 k 一个 scale
mma.sync 指令的 k 也是 32
-> 一条 mma 恰好对应一个 scale block
-> 每条 mma 算完必须立刻乘 scale 再累加，无法攒多条一起发
```

这是模型量化格式和指令粒度的**恰好相等**造成的，不是实现细节。

---

## 4. 模型格式（config.json 实测）

```json
{
  "quant_method": "fp8",
  "activation_scheme": "dynamic",
  "weight_block_size": [32, 32],
  "scale_fmt": "ue8m0",
  "expert_dtype": "fp4"
}
```

| 部分 | 格式 | 实测大小 |
| --- | --- | --- |
| 路由专家（w1/w2/w3） | **fp4**（int8 packed） | 5.62 MB/专家 |
| 共享专家 | fp8 e4m3 | 11.25 MB/层 |
| 注意力投影 | fp8 e4m3 | 6.25 MB/层 |

架构：40 层，每层 384 专家，top-6，dim=5120，moe_inter_dim=2304。

---

## 5. 我们的负载（生产 logs 实测，10 天日均）

```
请求       194,444 /天
平均输入   129,378 token
平均输出   833 token
缓存命中   90.23%
prompt:completion = 155:1
```

计费（官方谷价 x 0.1）：

```
单请求官方谷价成本  0.018307 元
我们收费            0.001831 元
日收入              356 元
月收入              10,679 元
```

---

## 6. 承载能力（8 卡，P:D = 1:1，用真实效率 14.7%）

prefill 每 token 需 16 GFLOP（激活 8B），一半卡做 prefill。

| 配置 | 单卡峰值 | prefill 供给 | 能接请求/天 | 对比当前 | 月收入 | 月成本 | 净 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8×5090 | 838.2 TF | **493 TF** | **20,600** | **0.11x** | 1,131 | 14,000 | **-12,869** |
| 8×PRO 6000 | 838.2 TF | **493 TF** | **20,600** | **0.11x** | 1,131 | 未知 | 未知 |
| 8×H200 | 1979.0 TF | **1,164 TF** | **48,600** | **0.25x** | 2,670 | 未知 | 未知 |

**当前引擎即使跑在 H200 上，也只能接住 25% 的现有流量。**

---

## 7. 要接住现有流量需要什么效率

| 卡 | 需要的效率 | 相对当前 14.7% |
| --- | ---: | ---: |
| 8×5090 | **139%** | 超过硬件峰值，不可能 |
| 8×PRO 6000 | **139%** | 超过硬件峰值，不可能 |
| 8×H200 | **59%** | 需提升 4.0 倍 |

**8×5090 和 8×PRO 6000 无论怎么优化都接不住现有流量。** 它们的 fp8 峰值只有 H200 的
42%，即使效率打满也差得远。只有 8×H200 有理论可能。

---

## 8. 已验证可用的技术事实

| 项 | 状态 |
| --- | --- |
| mma.sync m16n8k32 e4m3 x e4m3（fp8×fp8） | 引擎在用 |
| mma.sync kind::f8f6f4 m16n8k32 e2m1 x e4m3（fp4×fp8） | 本机实测跑通，数值可解释 |
| fp4 fragment 布局 | 从 cutlass 头文件推出并在设备上验证 |
| 端到端输出正确性 | 89/89 kernel 用例，输出 [223,19,17,20] 稳定 |

---

## 9. 已验证不可用的方案

| 方案 | 状态 | 原因 |
| --- | --- | --- |
| cutlass 87a/87b/87c（sm120 blockwise）| 不可用 | 编译通过，任何尺寸运行都崩 |
| cutlass 80a/80b（Blackwell sparse）| 不可用 | 编译失败：需要 tcgen05 |
| cutlass 54（Hopper FP8）| 不可用 | 运行时拒绝：requires compute capability 90 |
| 硬件 block-scale MMA | 不可用 | 三张候选卡全部不支持（sm_120/sm_90）|
| 压缩 block-32 到 block-128 | 不允许 | 会损失精度 |

---

## 10. 提高效率的可行方向（未验证，不承诺数字）

| 方向 | 机制 | 状态 |
| --- | --- | --- |
| 削减标量指令 | 36 条/轮 -> 更少 | 已做一部分（bs 提升到循环外，+16%）|
| 提高 occupancy | 当前 4 blocks/SM（寄存器+smem 限制）| 未做 |
| fp4 kernel | fp4 峰值是 fp8 的 2 倍 | 指令已验证，kernel 未写 |
| 多 stage 预取 | 掩盖 smem 延迟 | 试过，无改善 |

**这些能提到多少，必须做出来才算。本文不写推测值。**

---

## 11. 待确认

| # | 问题 | 影响 |
| --- | --- | --- |
| 1 | RTX 5090/PRO 6000/H200 的真机效率 | 本机只有 5060L。三张卡是否也是 14.7% 未验证 |
| 2 | fp4 kernel 的实际吞吐 | 指令可执行，但 kernel 未写 |
| 3 | 单专家 fp4 理论 16.88 MB vs 实测 5.62 MB | 可能是参数算错或存储有额外压缩 |
| 4 | 显存约束（5090 256 GB vs 模型 291 GB）| 必须流式 13%，PCIe 开销未计入 |
| 5 | 多卡通信（5090 无 NVLink）| EP 的 all-to-all 开销未计入 |

---

## 12. 代码位置

```
src/tinyinfer/kernels/cuda/fp8_gemm_mma.cu      prefill GEMM（实测 19.5 TFLOPS）
src/tinyinfer/kernels/cuda/fp8_gemv_bs.cu       decode fp8 GEMV
src/tinyinfer/kernels/cuda/gemv_fp4_w4a8.cu     专家 GEMV（fp4 × fp8）
src/tinyinfer/kernels/cuda/moe_gemv6.cu         MoE 融合 kernel
docs/PERF_NOTES.md                              性能测量笔记（含 5 个失败尝试）
```

---

*本文所有性能数字均可在本机复现。测量脚本在 local-scripts/。*
