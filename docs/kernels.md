# tinyinfer 算子调研报告 — sm_120 消费级 Blackwell（RTX 5060 Laptop / 8 GB）

> 2026-10-21 · kernel-researcher · 范围：FP8 / FP4 / attention / 投机与 CPU 辅助算子
> 结论基于**本机实测**（随机张量，未用模型权重）；所有脚本在 `tinyinfer/prototypes/kernels/`。
> 目标机器：RTX 5060 Laptop（26 SM，8151 MiB，sm_120）+ Ryzen 9 9955HX + 47 GB RAM + NVMe。

---

## 0. 结论速览（TL;DR）

1. **tilelang 0.1.15 在 Windows + sm_120 可以 JIT 编译**：Python 3.13 venv + nvcc 13.0 + VS2022 BuildTools 的 cl.exe（`vcvars64.bat` 环境）即可，**不需要 clang/MSYS2**。首个 kernel 编译 2–5 s，之后走磁盘缓存。
2. **官方 `reference/official/kernel.py` 的 4 个核心 kernel 在本机全部编译、数值通过**：`act_quant`（FP8 32-block e8m0）、`fp8_gemm`（32×32 blockwise）、`fp4_gemm`（FP8 act × FP4 weight，32-block e8m0）、`hc_split_sinkhorn`。只有 `sparse_attn` 因请求 141 KB 动态 shared memory 超过本卡 99 KB 上限而失败；**已写 sm120 补丁**（head 分块）并实测正确。
3. **FP8 GEMM**：`torch._scaled_mm` 的 **tensorwise** 路径在 sm_120 可用（M=512 时约 67 TFLOPS）；**rowwise 被设备拒绝**；1×128 / 128×128 blockwise 因 stride 校验失败；MXFP8 e8m0 blockwise 能启动但 scale layout 无公开文档、实测结果错误，不建议依赖。**tilelang 32×32 blockwise 与 HF 权重格式完全一致，实测 38–47 TFLOPS（M≥64），是推荐路径。**
4. **FP4**：
   - 官方 `fp4_gemm`（FP8 act × FP4 weight，HF 32-block e8m0）可用，实测 23–29 TFLOPS；它是 **FP4→FP8 转换后走 FP8 MMA**，并不使用原生 FP4 tensor core。
   - `torch._scaled_mm` 的 **NVFP4（fp4×fp4，e4m3 per-16）在 sm_120 可跑**，大 M 下 **160–190 TFLOPS**（bf16 的 4–5 倍）；但需要 W4A4（激活也 FP4，精度风险），且 scale 张量布局无文档——本报告只验证了数值（单位 scale 正确），**不推荐直接用于 V4.1 expert**。
   - 原生 FP4 MMA 需要 `sm_120a` 目标（Strata 实测证据：只有 `12xa` 编译单元能用 FP4 MMA；本卡 26 SM < 80，Strata 默认也不启用其 w4a4）。tilelang 0.1.15 是否支持 `sm_120a` 未验证。
5. **Attention**：Windows 版 torch 2.9 wheel **没有编译 flash attention**；head_dim=512 时 SDPA 的 flash/efficient/cuDNN 后端全部不可用，只剩 math（decode 272–1559 µs，慢）。**必须走自研/tilelang**。sm120 补丁版 sparse_attn 实测：decode 1 query × 640 KV = **70 µs/layer**（纯窗口 128 = **17 µs**）；prefill 128 query × 640 = **1.2 ms/layer**；投机 tree 4 token = **62 µs/layer**（与 1 token 同价）。
6. **投机解码原语**：tree verify 4 token 的 attention 成本 ≈ 1 个 decode step；greedy 验证（4×129k argmax）18.6 µs；rejection sampling 4 token = 125 µs kernel / 0.15 ms wall。**O5 的算子基础已经具备**。
7. **CPU 辅助**：CPU router 0.44 ms/layer（fp32 384×5120），CPU 采样（D2H 0.33 ms + softmax/argmax 0.59 ms），一个 expert 的 CPU BF16 gate 矩阵 0.184 ms，PCIe H2D 实测 **28 GB/s**。在 NVMe 4.6 GB/s、每 token 4.25 GB 专家权重的现实下，**CPU 只适合做“预测/调度/采样”，不适合承担大量专家计算**（O6/O9 的边界）。
8. **KV 量化**：V4.1 自身设计已给出答案——窗口 KV 用 FP8 per-32（相对误差 2.25%，cos 0.99965），压缩 KV 用 FP4 per-16（10.3%，cos 0.9957）；int4+Hadamard 与 FP4 同档（10.9%），int8+Hadamard 最优（0.6%）。

### 可用性矩阵（本机，✅=实测通过，⚠️=可运行但有条件，❌=不可用/未通过，—=未测）

| 能力 | torch | tilelang 0.1.15 | DeepGEMM | FlashInfer | FlashMLA | SDPA(torch) |
|---|---|---|---|---|---|---|
| FP8 e4m3 GEMM（tensorwise） | ✅ 67 TFLOPS（M=512） | — | — | — | — | — |
| FP8 blockwise 32×32 e8m0 | ❌ API 无此布局 | ✅ 38–47 TFLOPS，误差≈bf16舍入 | ⚠️ SM120 支持已合并（PR #324），未在 Windows 构建 | — | — | — |
| FP4 权重 × FP8 act | ⚠️ 仅 NVFP4 fp4×fp4 | ✅ 23–29 TFLOPS（FP4→FP8 再 MMA） | ⚠️ 同上（DeepGEMM 有 fp8_fp4_gemm_nt） | — | — | — |
| 原生 FP4 tensor core | ⚠️ NVFP4 160–190 TFLOPS，W4A4、scale 布局无文档 | ❌ 未验证 sm_120a | ⚠️ 需 sm120 分支 + Windows 构建 | — | — | — |
| MLA sparse attention（D=512） | ⚠️ 手写 mm 路径 80 µs/layer | ✅ 17/70 µs/layer（sm120 补丁） | — | ⚠️ SM120 sparse MLA 已合并（PR #3395），未构建 | ⚠️ 上游有 SM120 PR，#119 Windows 构建 issue | ❌ fused 后端不可用；math 慢 |
| 融合 FP8 量化 / 反量化 | ⚠️ `.to(float8)` 可但慢 | ✅ act_quant 13.6 µs | — | — | — | — |
| 融合 HC Sinkhorn | ❌ Python 20 iter | ✅ 24.7 µs | — | — | — | — |
| 投机 tree verify | ⚠️ 逐 token mm 331 µs/layer（4 tok） | ✅ 62 µs/layer（4 tok） | — | — | — | — |
| KV FP4/FP8 量化 | ⚠️ 手写 | ✅（官方 fp4_quant/act_quant 同源 kernel） | — | — | — | — |

> DeepGEMM / FlashInfer / FlashMLA 的“已合并”指上游 PR 状态（GitHub API 查证：DeepGEMM #324 merged 2026-06-24；FlashInfer #3395 merged 2026-06-15；vLLM #43477 merged 2026-06-22），但**本机 Windows 未安装/未编译/未实测**，不计入“今天可用”。

---

## 1. 测试环境与复现

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop，sm_120，26 SM，8151 MiB，驱动 617.42 |
| shared memory | 101,376 B/block（99 KB），102,400 B/SM；**这是官方 sparse_attn 失败的直接原因** |
| CUDA | Toolkit 13.0（nvcc V13.0.88）；torch 2.9.0+cu130（CUDA runtime 13.0） |
| Python | 3.13.7（项目 venv：`tinyinfer/.venv`） |
| 编译器 | VS2022 BuildTools 17.14 / MSVC 14.44.35207（cl.exe）；MSYS2 gcc 存在但未用到；**无 clang** |
| tilelang | 0.1.15（PyPI win_amd64 wheel），依赖 apache-tvm-ffi 0.1.12、z3-solver 4.15 |
| 计时 | CUDA events，预热 3–10 次，取 3 轮最小值；桌面占用 ~1.6 GB VRAM，未做功耗/温度控制 |

> 说明：同一脚本多次运行会有 ±10–20% 波动（桌面 GPU 争用/频率）。表中数字是代表性测量；**跨实现的相对结论比绝对值可靠**。重跑命令见第 6 节。

复现（PowerShell）：

```powershell
# 1) 安装 tilelang（清华镜像，缓存放 E 盘）
$env:UV_CACHE_DIR='E:\Develop\Deepseek\deepseek\uvcache'
uv pip install --python 'E:\Develop\Deepseek\tinyinfer\.venv\Scripts\python.exe' 'tilelang==0.1.15' --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 带 nvcc + cl.exe 环境运行（bat 内调用 vcvars64 并设置 CUDA_HOME）
cd E:\Develop\Deepseek\tinyinfer\prototypes\kernels
.\run_py.bat bench_gemm.py
.\run_py.bat bench_fp4.py
.\run_py.bat bench_attention.py
.\run_py.bat bench_decode_pipeline.py
.\run_py.bat test_sparse_attn_sm120.py
.\run_py.bat bench_spec_tree_attn.py
```

> `run_py.bat` 里写死了本机 VS/CUDA 路径，换机时改这两行即可。**tilelang JIT 必须在 vcvars64 环境里跑**（nvcc 需要 cl.exe 做 host 编译），否则报 host compiler 错误；已编译 kernel 有磁盘缓存，第二次起不需要重新编译。

---

## 2. 实测明细

### 2.1 tilelang 在 Windows 上的可行性

| 测试 | 结果 |
|---|---|
| import tilelang / 建 target | ✅（`tilelang.env.CUDA_HOME` 自动识别） |
| 最小 elementwise JIT（1 block，1024 元素） | ✅ 首次 5.3 s（含 nvcc + dll 加载），运行正确 |
| 小 FP16 GEMM（256×256×256，T.gemm） | ✅ 3.7 s，max err 0.0156（fp16 舍入） |
| 官方 `act_quant`（[256,5120] bf16→fp8，32-block e8m0） | ✅ 编译 2.3 s，13.6 µs/次，max abs err 0.25（=量化步长） |
| 官方 `fp8_gemm`（32×32 e8m0，[128,2304,5120]） | ✅ 编译 ~4 s，77 µs，39.2 TFLOPS，max err 0.98（对 2k 量级输出） |
| 官方 `fp4_gemm`（FP8 act × FP4 weight，[128,2304,5120]） | ✅ 编译 ~5 s，141 µs，21.3 TFLOPS，max err 0.98 |
| 官方 `hc_split_sinkhorn`（[128, 24] mixes，20 iter） | ✅ 24.7 µs，max err 1.2e-7（对 tinyinfer Python 参考） |
| 官方 `sparse_attn`（h=64,d=512） | ❌ `Failed to set the allowed dynamic shared memory size to 141312`（>101376） |
| sm120 补丁 `sparse_attn_sm120`（H_CHUNK=16、KV block=32） | ✅ 65 KB shared mem，数值通过（见 2.4） |

**结论**：tilelang 是当前 Windows + sm_120 上唯一“今天就能用”的 GPU kernel 编译路径；官方 `kernel.py` 95% 可直接复用。

### 2.2 GEMM 实测（单位 µs；误差为对反量化参考的最大绝对误差）

**expert w1/w3（N=2304, K=5120）**

| M | bf16 | torch FP8 tensorwise | tilelang FP8 32×32 | tilelang FP8 TFLOPS | tilelang FP4 (fp8×fp4) | tilelang FP4 TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71.2 | 24.3 | 39.7 | 0.6 | 67.6 | 0.3 |
| 64 | 49.5 | 36.2 | 47.4 | 31.8 | 93.7 | 16.1 |
| 128 | 93.3 | 50.4 | 77.1 | 39.2 | 141.5 | 21.3 |
| 512 | 369.0 | 180.0 | 285.5 | 42.3 | 517.7 | 23.3 |

**attention wq_b（N=32768, K=1280）**

| M | bf16 | torch FP8 tensorwise | tilelang FP8 32×32 | tilelang FP4 |
|---:|---:|---:|---:|---:|
| 1 | 232 | 124 | 251 | 86 |
| 128 | 319 | 176 | 267 (40.1 TF) | 359 (29.9 TF) |
| 512 | 1196 | 650 | 909 (47.3 TF) | 1515 (28.4 TF) |

要点：
- decode M=1 时所有 GEMM 都是 20–250 µs 级；单 token 的 6 个 expert × 3 个矩阵 ≈ 0.6–1 ms，**与 NVMe 的 0.9 s/token 相比可忽略**，O3 的收益主要在 prefill / 热缓存场景。
- M≥64 时 tilelang FP8 相对 bf16 **1.3–1.5×**；torch tensorwise 更快（~1.7× bf16）但需要 per-tensor scale，**与 HF 的 32×32 blockwise 不兼容**（需重新量化，精度损失）。
- tilelang FP4 是“FP4 权重省一半 IO + 转 FP8 计算”，FLOPs 吞吐低于 FP8，但**每字节权重吞吐更高**；decode 场景（权重带宽受限）用它更合适。
- 官方 `fp8_gemm`/`fp4_gemm` 的 `C` 必须用 bf16（`torch.get_default_dtype()`），调用前 `torch.set_default_dtype(torch.bfloat16)`，否则报 C dtype mismatch（官方在 `set_dtype()` 里全局设置）。

### 2.3 FP4 路径细节

**(a) 官方 tilelang `fp4_gemm`（推荐，与 HF 格式一一对应）**

- A：FP8 e4m3，per-row per-32 e8m0 scale；B：FP4 E2M1 packed [N, K/2]，per-row per-32 e8m0 scale。
- 与 HF 的 routed_expert 格式（E2M1 + 1×32 e8m0）**完全一致**，不需要重排 scale。
- 实现：FP4→FP8（经 FP32）后做 FP8 MMA，再乘回 block scale；不是原生 FP4 MMA。
- 实测 23–29 TFLOPS；比 FP8 慢约 35–45%，但权重字节数减半。decode 用它，prefill 也可以用（或反量化后 FP8）。

**(b) `torch._scaled_mm` NVFP4（能跑，但不建议生产用）**

- 调用形态：`torch._scaled_mm(a_packed[M,K/2], b_packed[N,K/2].t(), sa, sb, out_dtype=bf16)`，a/b 均为 `torch.float4_e2m1fn_x2`，scale 为 `float8_e4m3fn`，per-16。
- 约束（从 C++ 报错反推）：`scale_a.numel() == 128*ceil(M/128)*(K/16)`、`scale_b.numel() == 128*ceil(N/128)*(K/16)`，均需 contiguous。
- 实测速度：M=512 时 expert 形状 **72.9 µs（165.7 TFLOPS）**、wq_b **267 µs（160.9 TFLOPS）**；M=1 时 32–58 µs。**速度非常诱人（bf16 的 4–5×）。**
- 数值：单位 scale 时与手工 dequant 完全一致（code→E2M1 映射、packed 顺序已用常量矩阵验证）；但一旦用真实 per-16 e4m3 scale，结果与参考误差 ~50–100%。**scale 张量的 swizzle/索引布局没有公开文档，我们未在本次时间内反推出来**；torch 也没有公开 helper。
- 另外它是 W4A4（激活也要 FP4）。Strata 的实测给出警告：w4a4 单次乘积误差 8.6%、KL 0.008，远差于 w4a8（0.53%、KL 0.0018）。对 1017 这种深度 MoE 模型，精度风险高。
- **结论：列为 O4 研究项（需要先拿到 scale layout 或换 cuBLASLt/CUTLASS 直接调），不作为主路径。**

**(c) 原生 FP4 tensor core 的 arch 要求**

Strata 的 `docs/NVFP4.md` 明确指出：其 FP4×FP4 kernel（`mmq_nvfp4_w4a4.cu`）必须用 `12xa`（sm_120a）编译，其余引擎按普通 arch 编译；没有该编译单元的卡回退 w4a8。即 **FP4 MMA 需要 `sm_120a` 目标**。tilelang 当前默认按 `torch.cuda.get_device_capability()` 生成 `sm_120`；要启用原生 FP4 MMA 需 patch target 为 `sm_120a`（未验证 tilelang 0.1.15 是否接受）。本卡 26 SM，Strata 也仅在 80+ SM 的 sm120 卡上默认开 w4a4。

### 2.4 Attention（D=512，V4.1 MLA 形状）

| 实现 | decode 1×640 KV | decode 1×128（窗口层） | prefill 128 q × 640 KV | tree 4 tok × 640 |
|---|---:|---:|---:|---:|
| SDPA flash / efficient / cuDNN | ❌ 全部 No available kernel | ❌ | ❌ | ❌ |
| SDPA math（GQA） | 1558.8 µs | — | 864 µs（kv=128） | 1524.9 µs |
| SDPA math（KV 展开成 64 头） | 271.9 µs | — | — | — |
| torch 手写 mm（scores→softmax→out） | 80.9 µs | 83.6 µs | — | 331 µs（4 次逐 token） |
| **tilelang sm120 补丁 sparse_attn** | **62–70 µs** | **17.2 µs** | **1202 µs** | **62 µs（m=4）/ 121 µs（m=8）** |
| 40 层合计（decode） | 2.5–2.8 ms → 360–400 tok/s（attention-only） | 0.69 ms → 1450 tok/s | — | 2.5 ms（m=4） |
| tinyinfer 现有 `sparse_attn_ref`（einsum+gather） | 438.8 µs | — | 4492 µs | — |

- **SDPA 失败原因**：Windows torch wheel 未编译 flash attention（运行时警告原文：“Torch was not compiled with flash attention”）；efficient 后端对 head_dim 有上限（本机 D=512 直接被拒）；cuDNN 对 D=512 也无 kernel。D=128 的 indexer 形状可以用 cuDNN SDPA（150 µs @640，1208 µs @4096），但也不如手写 mm。
- **sm120 补丁**：官方 kernel 的 `q_shared`、`o_shared`、`kv_shared` 合计 141 KB（按 64 头全量），超过 99 KB。补丁把 head 维切成 16 一组（grid 增加一维），KV tile 32；shared 降到 ~65 KB。数值与 naive gather 参考一致（max err 0.002–0.004，bf16 舍入级）。
- 关键洞察：**tree verify 的 attention 成本与单 token 几乎相同（62 µs vs 62 µs）**——投机解码在 attention 侧几乎“免费”，代价主要在 MoE 权重读取（见 O5）。

### 2.5 投机 / 验证原语

| 原语 | 实测 | 说明 |
|---|---|---|
| tree verify attention（4 draft tokens，640 KV） | 62 µs/layer（tilelang 单 kernel）；40 层 2.5 ms | 与 1 token decode 同价；m=8 翻倍（121 µs） |
| greedy 验证（4×129280 argmax + 比较） | 18.6 µs | 可忽略 |
| rejection sampling（4 token，vocab 129k，含 gather/ratio/cumprod） | 125 µs kernel / 0.15 ms wall | 可 fuse 成 1–2 个 kernel 降到 <50 µs |
| DSpark confidence / MTP 调度 | 未测（config：3 个 MTP 层、block 5、markov rank 256） | 需要真实权重与训练后 head；O5 第二阶段 |

### 2.6 CPU 辅助与 PCIe（O6/O9 边界）

| 项 | 实测 | 含义 |
|---|---|---|
| CPU router（384×5120 gate，fp32，top-6） | 0.44 ms/layer | 40 层 17.6 ms/token；但可提前一层预测并与 GPU/NVMe 重叠 |
| CPU 采样：D2H logits 129280 bf16 + softmax/argmax | 0.33 + 0.59 ms | 采样可 CPU 做，但 GPU 只需 ~20 µs；仅在“CPU 先行、GPU 不等待”时有价值 |
| GPU→CPU `.item()` 同步 | 27 µs/次 | 每 token 若同步 5–10 次就是 0.15–0.3 ms |
| CPU 单 expert gate（2304×5120 bf16，M=1） | 0.184 ms | 6 expert × 40 层 ≈ 44 ms/token（仅 gate/up 之一）→ CPU 不能承担主计算 |
| PCIe H2D 23.6 MB pinned | 0.84 ms → **28.1 GB/s** | 4.25 GB/token 全走 PCIe = 151 ms → 6.6 tok/s 上限 |
| Strata 参考（Ryzen 7 255，AVX-512，NVFP4 专家） | gate+up 0.058 ms/expert（1 tok）、全核 58.6 GB/s | 说明“AVX-512 + 手写 dequant-dot”可把 CPU 专家做到可用；Windows torch 默认路径达不到 |

### 2.7 KV 量化误差（512 维 MLA latent，4096 token 随机 + 8× outlier 通道）

| 方案 | bytes/val | 平均相对误差 | cosine | attention score 相对误差 |
|---|---:|---:|---:|---:|
| FP8 e4m3 per-32 e8m0（V4.1 窗口 KV） | 1.0 | 2.25% | 0.99965 | 2.6% |
| FP4 E2M1 per-16 e4m3（V4.1 压缩 KV） | 0.5 | 10.3% | 0.99569 | 9.5% |
| int8 per-32 | 1.0 | 0.97% | 0.99996 | — |
| int8 + Hadamard per-32 | 1.0 | **0.60%** | **0.99999** | 0.49% |
| int4 per-32 | 0.5 | 17.5% | 0.98598 | — |
| int4 + Hadamard per-32 | 0.5 | 10.9% | 0.99629 | 8.8% |

结论：V4.1 自己的 FP8/FP4 KV 方案已在“压缩率-精度”前沿；int4+Hadamard 与 FP4 同档（可作为 O3 的替代/热缓存压缩），int8+Hadamard 精度最好但无压缩收益。Hadamard 对 int4 提升明显（17.5%→10.9%），值得在 O3 的 KV cache 里保留。

### 2.8 融合小算子

| kernel | 形状 | 实测 | 价值 |
|---|---|---:|---|
| `act_quant`（FP8 32-block e8m0） | [256, 5120] | 13.6 µs | 替代 `.to(float8)+手工 scale`，可 fuse 进 linear 前 |
| `hc_split_sinkhorn` | 128 行 mixes，20 iter | 24.7 µs | 替代 Python 20 次迭代（估计 0.2–0.5 ms），每层 2 次 → 重要 |
| 官方 `sparse_attn` | 见 2.4 | 141 KB smem 失败 | 用 sm120 补丁替代 |

## 3. O1–O9 算子路线（tinyinfer 落地视角）

> 约定：收益均为**本机实测**或明确标注“预期”；接入点写到 tinyinfer 的文件/函数；原型脚本在 `prototypes/kernels/`。
> 原则：**O1 正确性已由 Lead 完成；O2/O3 是本周可落地；O5/O6 是投机与 CPU 辅助；O7–O9 是系统级流水线，先定算子接口。**

### O1 — 正确性优先（已有，PyTorch eager）

- 现状：`src/tinyinfer/{attention,moe,quant,nn,hc,model}.py`，tests 全过；dequant + bf16 matmul 语义正确。
- 算子清单：`sparse_attn_ref`、`dequant_fp8_block`、`dequant_fp4_block`、`hc_split_sinkhorn`、`Gate/Expert/MoE`。
- 实测：正确性基线。
- 风险：eager 路径的 `sparse_attn_ref` 是 gather+einsum，decode 438 µs/layer、prefill 4.5 ms/layer，长上下文显存压力大。

### O2 — 融合 kernel（tilelang，本周可落地）

**目标**：把 attention / 量化 / HC / MoE 的小算子换成 tilelang 融合 kernel，减少 launch 与中间张量。

| 算子 | 实现方式 | 实测/预期收益 | 风险 | 接入点 |
|---|---|---|---|---|
| sparse MLA（窗口+topk gather+sink） | `sparse_attn_sm120.py`（官方 kernel 的 head 分块版） | decode 438→62–70 µs（6–7×）；prefill 4492→1202 µs（3.7×）；tree m=4 62 µs | H_CHUNK/block 需按卡调；99 KB smem 是硬约束；bf16 输入 | `attention.py::MLAWindowAttention.forward` 的 `sparse_attn_ref` 替换；保留参考实现做 fallback |
| FP8 activation 量化 | 官方 `act_quant`（32-block e8m0） | 13.6 µs/次；省显存/带宽 | 需 bf16 连续输入 | `nn.py` / `moe.py` 的 linear 前；shared expert 的输入 |
| HC Sinkhorn | 官方 `hc_split_sinkhorn` | 24.7 µs/层，替代 Python 20 iter（估 0.2–0.5 ms） | 需要 hc_scale/hc_base fp32 | `hc.py::hc_split_sinkhorn` |
| RMSNorm+RoPE 融合 | 参考 DeepInfer `tilelang_ops/cuda/fused_transform.py`，自写 kernel | 预期每层省 5–15 µs（未实测） | 开发成本；数值要与 `apply_rope` 对齐 | `nn.py` + `attention.py` 的 q/kv 预处理 |
| Gate/Up + SwiGLU + 量化融合 | 参考 Strata `STRATA_GU_SWIGLU`（routed experts 的 gate/up/SwiGLU/q8 一次 launch，sm120 默认开） | Strata 在 sm120 RTX PRO 5000 整体 +13.8% decode（6/6 pairs），该项未单独拆出 | 需要 expert 权重按 tile 流式载入，kernel 较复杂 | `moe.py::Expert.forward`；建议 O3 之后做 |
| MoE combine / scatter | 自写 tilelang（index_add 融合 routing weight） | 预期 <20 µs/层（当前 Python 循环按 expert） | 专家数 384，稀疏路由；要处理空 expert | `moe.py::MoE.forward` 的 `y[idx] += ...` 循环 |

**O2 风险**：tilelang JIT 首次编译 2–6 s/kernel，进程启动需预热；kernel 缓存目录要固定；Windows 必须 vcvars 环境（部署脚本要封装）。

### O3 — 量化加速（FP8 / FP4 / KV）

**目标**：让权重“直接以量化态参与计算”，省 IO 和显存，而不是 dequant 成 bf16。

| 路径 | 现状 | 实测 | 建议 |
|---|---|---|---|
| 非专家 FP8 权重（HF 32×32 e8m0） | `dequant_fp8_block` → bf16 | tilelang `fp8_gemm` 32×32：38–47 TFLOPS（M≥64），误差 bf16 舍入级 | **O3 主线**：`ShardStore` 保留 fp8 + scale，`Linear` 走 `fp8_gemm`；activation 用 `act_quant(32, e8m0)` |
| routed expert FP4 权重（HF 1×32 e8m0） | `dequant_fp4_block` → bf16 | tilelang `fp4_gemm`：23–29 TFLOPS，权重字节减半 | **O3 主线**：expert 权重保持 fp4 packed，activation 先 `act_quant` 到 FP8，直接 `fp4_gemm` |
| torch tensorwise FP8 | 可用 | 67 TFLOPS（M=512），M=1 24–124 µs | 只在“愿意把权重重新 per-tensor 量化”时用；与 HF 32×32 不兼容 |
| torch NVFP4（fp4×fp4） | 可运行 | 160–190 TFLOPS（M=512） | **不推荐**：scale layout 无文档（本次未反推出）、W4A4 精度风险（Strata: KL 0.008 vs w4a8 0.0018） |
| KV cache 压缩 | bf16 现状 | FP8 per-32：2.25% / cos 0.99965；FP4 per-16：10.3% / 0.9957；int4+Hadamard 10.9% | 窗口 KV FP8、压缩 KV FP4（与模型训练一致）；用 `act_quant`/`fp4_quant` kernel 在写入 cache 时量化 |
| 直接量化权重计算 | 目标 | — | 关键收益：expert 保持 fp4；GPU 侧显存/带宽减半；prefill 也省 dequant |

**O3 收益估算**：prefill M=128–512 时 GEMM 1.3–1.5× bf16；专家权重 IO 减半（4.25 GB→2.13 GB/token，若采用 fp4 直接读+算，理论 decode 从 ~1.1 s/token 降到 ~0.55 s/token，**但受 NVMe 随机读与 page cache 影响，需实测**）。这是本项目最确定、最值得先做的优化。

**风险**：FP8 `fp8_gemm` 当前 kernel 的 block_M=32、无 TMA/warp-specialized（官方 pass_configs 关掉了），大 M 只到 47 TFLOPS，离 cuBLAS 的 67 TFLOPS 有差距；可后续用 tilelang autotune 或 DeepGEMM（O4）。

### O4 — 激进优化（需要移植/编译/新 kernel）

| 方向 | 参考 | 本机可行性 | 预期收益 | 风险/成本 |
|---|---|---|---|---|
| 原生 FP4 MMA（W4A4） | Strata `mmq_nvfp4_w4a4.cu`（sm_120a）；torch NVFP4 | 需 `-arch=sm_120a`；tilelang 未验证；本卡 26 SM，Strata 在 <80 SM 卡默认不启用 | 理论 4–5× bf16（torch NVFP4 已测 160–190 TFLOPS） | 精度差（KL 0.008）、layout/编译器支持不确定；**建议先做精度评估再投入** |
| 融合 int8 prompt experts | Strata `STRATA_PF_FUSED`（sm86 +16–22% 端到端；sm120 默认开，80+ SM） | int8 tensor core 在 sm120 可用；tilelang 可写 mma int8 | prefill +10–20%（Strata 数据；本机需实测） | kernel 复杂；本卡 26 SM，Strata 在 RTX 5070（48 SM）上该路径反而 -4.7%，**本机可能不划算** |
| thread-block cluster decode | Strata sm_90+ cluster kernel（greedy argmax/块选择） | Blackwell 支持 cluster；tilelang 0.1.15 的 cluster 支持未测 | decode +3–10%（Strata sm86 ATTN_DAILY +3.2%；cluster 是另一路径） | launch/同步复杂；收益依赖 kernel 占比 |
| 大 tile / TMA / warp-specialized GEMM | DeepGEMM PR #324（sm120）、cuBLASLt | 需源码编译 DeepGEMM；Windows 未验证 | 可能把 FP8 从 47 提到 67+ TFLOPS | 构建成本高；O4 再做 |
| FlashInfer SM120 sparse MLA | PR #3395（已合并） | 需要 Linux/源码构建；Windows 无 wheel | 与我们的 patch 同类，可能更快（paged/批量化） | 移植成本高；O4 备选 |
| FlashMLA SM120 | 上游 #124/#146/#189；Windows 构建 issue #119 | 未编译/未验证 | sparse decode 专用优化 | 同上 |
| KV streaming（KV 在 RAM，VRAM 只放当前 slice） | Strata `KV_PREFETCH`；tinyinfer 8 GB 约束 | 纯 Python 调度即可（cudaMemcpyAsync + events） | 长上下文 KV 显存从 GB 级降到百 MB | 需要 attention kernel 支持分段/分页（O4） |
| CPU 分担 missed experts | Strata `STRATA_PCIE_BALANCE`、`--pcie-frac` | 本机 CPU 单 expert 0.184 ms、PCIe 28 GB/s | 命中率低的长尾专家可 overlap；整体 +5–15%（Strata 数据） | 需要 GPU/CPU 双实现同一 expert 量化格式（O9） |

### O5 — 投机解码（500B 模型进笔记本的关键杠杆之一）

**目标**：用 MTP/DSpark 草稿 + tree verify 把每 token 的“权重读取次数”摊薄。

| 算子 | 实现 | 实测 | 接入点 |
|---|---|---|---|
| tree verify attention（4/8 draft tokens） | `sparse_attn_sm120` 一次调用传 `idxs [b, m, topk]`（每个 draft token 的祖先+窗口索引） | m=4: 62 µs/layer（与 m=1 同价）；m=8: 121 µs | `attention.py` 增加 `forward_verify(x, idxs)`；decode loop 在 `model.py` |
| greedy 验证 | `argmax` + 逐位比较（GPU），一次 kernel | 18.6 µs（4×129k） | `nn.py::softmax_sample` 旁加 `greedy_verify` |
| rejection sampling | 现原型 125 µs kernel；可 fuse gather/ratio/cumprod 到 <50 µs | 0.15 ms wall（含 sync） | `nn.py` 采样路径；temperature>0 时用 |
| DSpark confidence 调度 | config：3 个 MTP 层（37–39）、block 5、markov rank 256、3 experts/token | 未测（需真实权重） | 后续 `dspark` 模块；先做 MTP draft + tree verify |
| speculative 调度 | 用 `--spec 4` 类策略：草稿数 4、min_p 0.5 | Strata 在 sm120 上 spec 4 比 spec 3/5/6/8 更稳（1.12–1.14× before） | tinyinfer decode loop |

**预期**：若 draft 接受率 60–80%，MoE 每步可验证 4 token，权重读取摊薄 2–3×；结合 O3 的 fp4 直接计算，理论上 decode 可从 ~1 tok/s 提到 2–3 tok/s（NVMe 受限下），热缓存下更高。**风险**：MTP/DSpark head 需要真实权重与训练分布；草稿质量差时 rejection 回退会浪费读取。**建议 O5 在 O3 之后做，先用 tree attention 原语 + 简单 n-gram/MTP draft 验证接受率。**

### O6 — CPU 辅助决策（router / gate / 采样，提前一层）

**目标**：GPU 等 NVMe 的时候，CPU 先算下一层 router、预测 expert、准备采样；隐藏 CPU 时间。

| 项 | 实测 | 设计 | 风险 |
|---|---|---|---|
| Router on CPU | 0.44 ms/layer（fp32） | 每层 x 只有 5120 维；CPU 用 bf16/int8 + 多线程可降到 ~0.1 ms；**提前 1 层**跑，与 GPU 计算/IO 重叠 | 路由与 GPU 端结果必须逐位一致（否则专家选择不同）；或接受微小误差（需评估） |
| 采样 on CPU | 0.33 ms D2H + 0.59 ms softmax/argmax | 只在 GPU 需要继续算下一 token 时并行做；或改为 GPU 采样（20 µs） | CPU 采样会延迟下一 token 的启动；需 pipeline |
| 同步开销 | `.item()` 27 µs | 每 token 最多 1–2 次同步；用 pinned memory + event | 频繁同步会吃掉投机收益 |
| 预测专家 | router 已知 top-6 + bias，可用上一层 hidden 预测下一层（DSpark 思路） | O7 的预取输入 | 预测错误导致预取浪费 NVMe 带宽 |

**结论**：O6 的收益不是“算力”，而是**把 CPU 工作藏进 NVMe stall**；本机 4.25 GB/token 的 IO 时间 ~0.9 s，足以隐藏 17.6 ms 的 CPU router。**优先做“router/采样异步化 + 下一层专家预取”，不需要 CPU 算专家。**

### O7 — 加载协同（IO/计算重叠）

| 事实 | 数值 | 设计 |
|---|---|---|
| NVMe 顺序读 | 4.6 GB/s（项目基线） | 6 expert × 17.7 MB × 40 层 = 4.25 GB/token → 冷缓存 ~0.92 s/token |
| PCIe H2D | 28.1 GB/s（pinned，23.6 MB） | 专家权重从 RAM 到 VRAM 0.63 ms/17.7 MB；若已在页缓存可用 |
| GPU 计算/token | 6 expert × ~108 µs（tilelang fp4）= 0.65 ms + attention 2.5 ms | 与 IO 完全可重叠（0.9 s >> 3 ms） |
| VRAM 容量 | 8 GB，桌面占 1.6 GB | 可缓存 ~200 个 fp4 expert（3.5 GB）+ KV/激活；命中率决定实际速度 |
| 接口需求 | 双缓冲 expert slot + CUDA events；`cudaMemcpyAsync` 在独立 stream；engram 行读取（256 B/行）用 O_DIRECT/异步 | 当前 tinyinfer 的 `MoE.forward` 同步加载 expert，**O7 的第一刀** |

**建议**：O7 先做“专家预取队列”：在 attention/前一层 FFN 计算时，按 router 结果异步读下一层 6 个专家到 pinned buffer → H2D → 计算；用 2–3 个 slot 轮转，避免阻塞。预期把“读 0.92 s + 算 3 ms”变成 max(0.92 s, ...)，在热缓存下收益更大。**风险**：NVMe 随机读、Windows 文件缓存行为、8 GB 显存 slot 数与命中率的平衡。

### O8 — 跨 agent / 多请求协同

- 场景：多 agent（或投机树多分支）共享同一模型；可共享 **expert cache**（同一层专家被多个请求命中）与 **prefix KV**（prompt cache）。
- 算子接口：
  - 批量 sparse attention：`idxs` 增加 batch 维（[B, m, topk]），kernel 已支持 b 维；decode 时把多个请求拼成 batch（KV 长度不同 → 需要 paged KV / per-request offset，O4 的 paged kernel）。
  - 批量 MoE：把 N 个请求的 token 合并成一个大 M 的 `fp4_gemm`（grouped GEMM）；官方 `fp4_gemm` 已支持任意 M（block_M=32），批到 M=512 时 23–29 TFLOPS 比 M=1 的 0.3 TFLOPS 高 70 倍 → **batch 是专家计算的关键**。
  - 共享专家缓存：按 (layer, expert_id) 的 LRU；命中率统计。
- 预期：多 agent 时吞吐近线性提升（受 NVMe/PCIe 带宽限制）；单请求延迟不变。
- 风险：请求间 KV 隔离/前缀共享一致性；批处理的调度延迟。

### O9 — 异构流水线（GPU attention / 部分专家 + CPU 专家）

- 现状：CPU 单 expert BF16 0.184 ms（24 MB 权重），6×40 层 ≈ 44 ms/token（仅 gate/up 之一），全部 expert 约 130 ms/token；PCIe 4.25 GB/token = 151 ms；**CPU 路径与 PCIe 路径都不如 NVMe 直读 + GPU 计算**（0.92 s/token 是全 IO 的上限，但 GPU 计算几乎免费）。
- Strata 的 CPU 多 token kernel 经验：一个权重块只 dequant 一次，供同一 verify window 的 1–4 个 token 复用（1 tok 0.058 ms → 4 tok 0.125 ms，约 3.3×）；tinyinfer 若做 CPU 尾部专家，应沿用这个“按 token 复用 dequant”的接口，而不是每 token 重解。
- 真正有意义的分工：
  1. **CPU 做“缺失专家”尾部**（Strata `--pcie-frac` 思路）：GPU 处理命中专家，CPU 处理未命中的少量专家，PCIe 只传 hidden/结果（KB 级），不传 17.7 MB 权重 → 避免 PCIe 成为瓶颈。
  2. **CPU 做 router/采样/验证**（O6），GPU 做 attention + 专家 GEMM。
  3. **CPU 做 engram 行读取/预处理**（196.6 GB 表，按行读 256 B），GPU 只收 embedding 结果。
- 需要的算子接口：
  - CPU 端 fp4 dequant + dot（AVX-512）：参考 Strata `src/kernels/cpu/nvfp4_avx512.cpp`（0.058 ms gate+up/expert，58 GB/s）；本机 Ryzen 9 9955HX 支持 AVX-512，但 Windows torch CPU 没有该 kernel，需要 C++/numpy 实现。
  - GPU/CPU 结果合并：hidden 维 5120 bf16 = 10 KB，PCIe 往返 <10 µs。
- 预期：只有当“专家命中率低 + NVMe 饱和”时才有收益；本机 NVMe 4.6 GB/s、6 expert/token，命中率 50% 时 CPU 分担可省 ~0.45 s/token（理论），但 CPU 计算 130 ms/token 会变成新瓶颈 → 需要 AVX-512 优化（0.058 ms/expert 可到 ~14 ms/token）。
- 风险：CPU 实现与 GPU 数值不一致（fp4 解码顺序、scale）；Windows 线程/大页/亲和性；实现成本高。**建议 O9 放到最后，先做 O6/O7。**

---

## 4. 推荐方案（按优先级）

### 今天就能用（P0）

1. **tilelang 官方 kernel 复用**：`act_quant`、`fp8_gemm`(32×32)、`fp4_gemm`(fp8×fp4)、`hc_split_sinkhorn` 直接接入 tinyinfer；`sparse_attn` 用 `sparse_attn_sm120.py` 补丁。
2. **O2 先换 attention**：`attention.py` 的 `sparse_attn_ref` → `sparse_attn_sm120`，decode 6–7×、prefill 3.7×；先保证与参考实现数值一致（误差 <1e-2）。
3. **O3 再换 expert GEMM**：`moe.py` 保留 FP4 packed 权重 + FP8 activation，走 `fp4_gemm`；shared expert / attention 走 `fp8_gemm`。这是省 IO 的核心。
4. **O5 原语**：tree verify 用同一个 sparse kernel（`idxs` 传树祖先），先接 MTP draft 验证接受率。

### 需要移植/编译（P1–P2）

- DeepGEMM（sm120 已合并，Windows 源码构建）→ 更快的 grouped FP8/FP4 GEMM；建议在 WSL2 或 Docker 里先验证，再决定是否值得移植到 Windows。
- FlashInfer SM120 sparse MLA（PR #3395）→ paged/批量 attention；与我们的 patch 功能重叠，O4 再评估。
- FlashMLA SM120（上游 PR/issue）→ 专用 sparse decode；Windows 构建成本高。
- 原生 FP4 MMA（sm_120a）→ 需要 tilelang target patch + 精度评估；W4A4 精度风险大。

### 放弃 / 不推荐

- `torch._scaled_mm` 的 rowwise FP8 在 sm_120 不支持；blockwise 1×128/128×128 的 stride 校验不通过（除非改 layout，无收益）。
- torch MXFP8 e8m0 blockwise：scale 布局无文档，实测错误。
- torch NVFP4 作为主路径：scale layout 未解 + W4A4 精度风险（仅作 O4 研究）。
- Windows 上 SDPA 融合后端：torch wheel 未编译 flash，D=512 无 kernel；不要浪费时间试 SDPA。
- 不要用 CPU 承担主专家计算（会从 IO-bound 变成 CPU-bound）。

---

## 5. 风险清单

| 风险 | 影响 | 对策 |
|---|---|---|
| tilelang 依赖 vcvars64/cl.exe 环境 | 部署脚本在无 VS 的机器上失败 | 文档写死 `run_py.bat`；打包时检测/报错；或预编译 kernel 缓存 |
| 首次编译 2–6 s/kernel × 数十 kernel | 启动慢 1–2 分钟 | kernel 缓存目录随仓库分发；或 build 阶段预热 |
| sm_120 shared memory 99 KB | 官方 sparse_attn 不能直接用 | 用 `sparse_attn_sm120`（H_CHUNK 可调 8/16/32） |
| FP4/FP8 数值与 HF 块尺度不一致 | 精度下降 | 保持 HF 原 scale 布局（32-block e8m0），不做 per-tensor 重量化 |
| 8 GB VRAM + 桌面占用 | kernel/权重 OOM | 专家分块、KV 压缩、激活 checkpoint；O7 slot 管理 |
| NVMe 4.6 GB/s 与 4.25 GB/token | 冷缓存 ~1 s/token | 专家缓存/预取/O5 投机摊薄；这是系统级上限，不是 kernel 能解决的 |
| Windows 文件缓存/杀毒干扰 IO | 实测波动 | 基准时关闭实时扫描；O_DIRECT 需谨慎 |
| 测量受桌面 GPU 占用/温度影响 | 数字 ±10% | 报告中已注明；关键结论用相对值 |

---

## 6. 复现与脚本清单（`tinyinfer/prototypes/kernels/`）

| 脚本 | 作用 | 状态 |
|---|---|---|
| `probe_torch_api.py` | 检查 torch 2.9 的 fp8/fp4/scaled_mm API | ✅ |
| `probe_scaled_mm.py` | tensorwise/rowwise FP8 初探 | ✅ |
| `probe_blockwise2.py` | 1×128/128×128/e8m0/e4m3 blockwise 支持探测（错误信息很有价值） | ✅（部分按预期失败） |
| `probe_mxfp8_layout.py` | MXFP8 scale 布局变体测试 | ✅（结论：不可用） |
| `probe_nvfp4_layout.py` / `probe_nvfp4_codes.py` / `probe_nvfp4_scale_map.py` | NVFP4 数值/布局反推 | ✅（值正确、scale 布局未解） |
| `bench_gemm.py` | bf16 / torch FP8 / tilelang FP8 / FP4 GEMM 扫描 | ✅ |
| `bench_fp4.py` | tilelang FP4 GEMM 全形状 | ✅ |
| `bench_nvfp4_torch.py` | torch NVFP4 速度/误差 | ✅ |
| `bench_attention.py` | SDPA 后端 + 手写 sparse MLA + prefill | ✅（D=512 fused 全失败） |
| `test_official_kernels.py` | 官方 kernel.py 编译/数值/计时 | ✅（sparse_attn 预期失败） |
| `sparse_attn_sm120.py` | sm120 补丁 kernel（H 分块） | ✅ |
| `test_sparse_attn_sm120.py` | 补丁正确性 + decode/prefill 计时 | ✅ |
| `bench_spec_tree_attn.py` | tree verify m=1..16 + greedy | ✅ |
| `bench_decode_pipeline.py` | torch decode/prefill、投机、CPU/PCIe | ✅ |
| `probe_kv_quant.py` | KV FP8/FP4/int4+Hadamard 误差 | ✅ |
| `try_tilelang_jit.py` + `run_tilelang_jit.py`/`run_py.bat` | tilelang 最小 JIT 验证 | ✅ |
| `probe_sdpa_d128.py` | D=128 indexer 形状 SDPA 探测 | ✅ |

---

## 7. 参考

- 官方参考实现：`tinyinfer/reference/official/kernel.py`（tilelang kernels）、`model.py`（V4.1 attention/MoE/HC/indexer/compressor）
- DeepInfer：`repositories/DeepInfer`（`linear.py` 的 deep_gemm fp8_fp4、`tilelang_ops/`、sm100 硬编码）
- Strata：`repositories/Strata/docs/DETAILS.md`（sm120 默认开关、fused int8 prompt experts、cluster decode、KV 预取、CPU 专家）、`docs/NVFP4.md`（sm_120a FP4 MMA、w4a4 精度）、`src/kernels/cpu/nvfp4_avx512.cpp`
- vLLM PR #43477（SM120 DeepSeek V4，merged 2026-06-22）：https://github.com/vllm-project/vllm/pull/43477
- DeepGEMM PR #324（SM120，merged 2026-06-24）：https://github.com/deepseek-ai/DeepGEMM/pull/324
- FlashInfer PR #3395（SM120 sparse MLA，merged 2026-06-15）：https://github.com/flashinfer-ai/flashinfer/pull/3395
- FlashMLA SM120：#124/#146/#189，Windows 构建 issue #119：https://github.com/deepseek-ai/FlashMLA/issues/119
