# tinyinfer 算子原型（sm_120 / RTX 5060 Laptop）

本目录是 `docs/kernels.md` 的可运行原型：FP8 / FP4 / attention / 投机 / CPU 辅助。
全部使用**随机张量**，不需要模型权重；目标是把“能不能跑、跑多快、误差多大”在本机测出来。

## 环境与运行

- Python：`E:\Develop\Deepseek\tinyinfer\.venv\Scripts\python.exe`（3.13.7 + torch 2.9.0+cu130）
- tilelang 0.1.15（已安装在上述 venv；重装见 `docs/kernels.md` 第 1 节）
- CUDA Toolkit 13.0（`nvcc` 在 PATH），VS2022 BuildTools（`cl.exe`）
- **tilelang JIT 必须在 vcvars64 环境里跑**。本目录的 `run_py.bat` 已封装：

```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "PATH=%CUDA_HOME%\bin;%PATH%"
"E:\Develop\Deepseek\tinyinfer\.venv\Scripts\python.exe" %*
```

用法（PowerShell，在任意目录）：

```powershell
& 'E:\Develop\Deepseek\tinyinfer\prototypes\kernels\run_py.bat' 'E:\Develop\Deepseek\tinyinfer\prototypes\kernels\bench_gemm.py'
```

> 换机器时只需改 `run_py.bat` 里的 VS/CUDA 路径。tilelang kernel 有磁盘缓存，同一 kernel 第二次运行不再编译（首次 2–6 s/个）。

## 脚本清单

| 脚本 | 内容 | 预期 |
|---|---|---|
| `try_tilelang_jit.py` | 最小 elementwise + FP16 GEMM 的 JIT 可行性 | ✅ 编译并正确 |
| `probe_torch_api.py` | torch 2.9 的 fp8/fp4/_scaled_mm API 面 | ✅ 打印支持情况 |
| `probe_scaled_mm.py` | tensorwise/rowwise FP8 初探 | ✅ tensorwise 通过，rowwise 失败 |
| `probe_blockwise2.py` | blockwise 1×128 / 128×128 / MXFP8 / NVFP4 的支持边界 | ⚠️ 部分按预期失败（错误信息用于判定） |
| `probe_mxfp8_layout.py` | MXFP8 e8m0 scale 布局变体 | ⚠️ scale 布局不对，结论“不可用” |
| `probe_nvfp4_layout.py` / `probe_nvfp4_codes.py` / `probe_nvfp4_scale_map.py` | NVFP4 数值映射、packed 顺序、scale 索引反推 | ⚠️ 值/打包正确；真实 scale 布局未解 |
| `bench_gemm.py` | bf16 / torch FP8 tensorwise / tilelang FP8 32×32 / tilelang FP4(fp8×fp4) | ✅ 见 docs 表 2.2 |
| `bench_fp4.py` | 上表的 FP4 扩展（M=1..512） | ✅ 23–29 TFLOPS（M≥64） |
| `bench_nvfp4_torch.py` | torch NVFP4 速度/误差 | ✅ 160–190 TFLOPS，但真实 scale 下误差大 |
| `bench_attention.py` | SDPA 四后端 + 手写 sparse MLA + dense prefill | ✅ D=512 的 fused 后端全部失败；math/手写有数字 |
| `test_official_kernels.py` | 官方 `reference/official/kernel.py` 的编译/数值/计时 | ✅ act_quant/fp8_gemm/fp4_gemm/hc 通过；sparse_attn 因 141KB>99KB 预期失败 |
| `sparse_attn_sm120.py` | 官方 sparse_attn 的 head 分块版（H_CHUNK=16、KV block=32） | ✅ 可直接 import 使用 |
| `test_sparse_attn_sm120.py` | 补丁正确性 + decode/prefill 计时 | ✅ decode 17–70 µs/layer |
| `bench_spec_tree_attn.py` | tree verify m=1..16 + greedy | ✅ m=4 与 m=1 同价 |
| `bench_decode_pipeline.py` | torch decode/prefill、tree、rejection、CPU router/sampling/PCIe | ✅ 见 docs 2.5/2.6 |
| `probe_kv_quant.py` | KV FP8/FP4/int4+Hadamard 误差 | ✅ 见 docs 2.7 |
| `probe_sdpa_d128.py` | D=128 indexer 形状的 SDPA 后端 | ✅ cuDNN 可用但慢 |
| `probe_tilelang_math.py` / `probe_tilelang_math2.py` | tilelang 数学 intrinsic / 位运算 / floor / ldexp 支持 | ✅ |
| `approx_kernels.py` | 精确 vs fast：RMSNorm / SwiGLU / softmax / RoPE 表 kernel | ✅ |
| `test_approx_kernels.py` | 近似 kernel 数值校验 | ✅ |
| `bench_approx.py` | exact/hw/poly 的 µs + KL/相对误差 + attention exp_mode | ✅ |
| `bench_real_weights.py` | L0 真实权重：近似误差、RoPE 表、fp8/fp4 GEMM vs dequant | ✅ |
| `spec_ops.py` | tree/DSpark attention idx、rejection sampling、DSpark head 接口 | ✅ |
| `bench_spec_ops.py` | tree/DSpark attention、rejection、markov head 计时 | ✅ |
| `fused/fused_ops.py` | fused_rmsnorm_rope / fused_gemm_rope / rope_inplace | ✅ |
| `fused/test_fused_ops.py` | 三个融合 kernel 正确性 + 单次耗时 | ✅ |
| `fused/bench_fused_attn.py` | 真实 L0 权重：整层 PyTorch vs fused，kernel 数 | ✅ |
| `fused/hc_fused.py` | fused_hc_mixes / fused_sinkhorn / fused_hc_pre / fused_hc_post | ✅ |
| `fused/bench_hc_fused.py` | 真实 L0 HC：torch vs fused，正确性/耗时/kernel 数 | ✅ |
| `fused/gemv_fp8.py` | decode fp8 GEMV（bf16 入口 / 预量化 / pq4 / RoPE epilogue） | ✅ |
| `fused/bench_gemv_fp8.py` | 真实 L0 四个 GEMM：official vs GEMV，带宽/正确性 | ✅ |

## 2026-10-21 第二轮新增结论（详见 `docs/kernels.md` 2.9-2.11、第 8 节）

- **近似数学（负结果）**：把 `approx_math` 的 fast_rsqrt / fast_exp / fast_silu 写进 tilelang kernel，
  sm_120 上没有可测收益（softmax 96.7 vs 97.8 µs；attention 64.5 vs 65.3 µs；RMSNorm 的 fast_rsqrt 更慢）。
  真实权重精度损失：fast_rsqrt 0.45%、fast_exp softmax 0.19%（KL 3.2e-5、top1 100%）、fast_silu 0.16%。
  **建议 GPU 侧关闭近似，保留硬件 intrinsics**；近似留给 CPU/O9（llama.cpp `ggml_v_expf` 也是 CPU 专用）。
  RoPE 预计算表 kernel 相对 torch complex 路径 **~10x**（24-43 µs vs 293-447 µs）。
- **投机算子**：tree attention 18.6-35.0 µs/layer（m=4/5/8），DSpark 窗口+block 18.7-35.0 µs；
  rejection sampling 417 µs/步（eager torch，需 fuse）；**DSpark markov head 1.84 ms/5-token 是 O5 瓶颈**。
- **O3 真实权重**：wq_b fp8 M=1 306 vs bf16 249 µs（更慢），M=512 957 vs 1308 µs（1.37x）；
  expert w1 fp4 M=1 73 vs 73.5 µs，M=64 101 vs 52 µs（预 dequant 驻留时 bf16 更快）——
  prefill 用 fp8、decode 用 dequant/bf16，expert 的 fp4 收益在权重 IO 减半。
- **llama.cpp 借鉴清单**：`block_mxfp4`/`block_nvfp4` 与 HF/V4.1 KV 布局对应、`dsv4-hc.cu` 的 HC pre/post、
  `topk-moe.cu` 的 fused routing、`ggml_v_expf`（CPU SIMD exp）、`softmax.cu` 的 sink softmax。

## 关键结果（详见 `docs/kernels.md`）

- tilelang JIT：Windows + sm_120 可行（nvcc 13.0 + MSVC cl.exe；无 clang）。
- FP8 32×32 blockwise（与 HF 一致）：38–47 TFLOPS（M≥64），误差 = bf16 舍入级。
- FP4（fp8 act × fp4 weight）：23–29 TFLOPS；权重减半。
- Attention（D=512，SDPA fused 全部不可用）：
  - tilelang sm120 补丁：decode 1×640 = 62–70 µs/layer；窗口 128 = 17 µs；prefill 128×640 = 1.2 ms。
  - tree verify 4 token = 62 µs/layer（≈1 token 的成本）。
- 投机原语：greedy 18.6 µs；rejection 125 µs kernel / 0.15 ms wall。
- CPU：router 0.44 ms/layer；采样 0.33+0.59 ms；一个 expert gate 0.184 ms；PCIe 28 GB/s。
- KV：FP8 2.25% / FP4 10.3% / int4+Hadamard 10.9%（相对误差）。

## 注意

1. 本目录所有脚本默认 `torch.set_default_dtype(torch.bfloat16)`（与模型一致）；tilelang kernel 的 `C` 必须是 bf16，否则报 dtype mismatch。
2. 官方 `sparse_attn` 在本卡不可运行（shared mem 141 KB > 101376 B）；请用 `sparse_attn_sm120.py`。
3. `torch._scaled_mm` 的 NVFP4/MXFP8 路径在本机“能跑”，但 scale 布局没有公开文档；本目录的 probe 脚本只是研究记录，**不要直接用于模型权重**。
4. 计时受桌面 GPU 占用影响（约 1.6 GB VRAM 被系统占用），数字 ±10%；关键结论用相对值。

## 2026-10-21 第三轮：融合 attention kernel（fused/）

把每层 attention 的 Python 小算子合并进 tilelang：`fused_rmsnorm_rope`、`fused_gemm_rope`（官方 fp8 32×32 e8m0 GEMM + RoPE epilogue）、`rope_inplace`（inverse RoPE）。

真实 L0 权重、decode b=1,s=1,kv=129、median of 5：

| 步骤 | fused | PyTorch 分步 | 加速 |
|---|---:|---:|---:|
| q_norm (N=1280) | 25.8 µs | 116.5 µs | 4.5x |
| kv norm + RoPE (N=512) | 18.1 µs | 232.4 µs | 12.8x |
| wq_b GEMM + RoPE (N=32768,K=1280) | 223.6 µs | 244.9 µs | 1.10x |
| inverse RoPE on o | 15.8 µs | 81.6 µs | 5.2x |
| 整层 attention | 818.7 µs | 884.3 µs | 1.08x |

- 正确性：`fused_gemm_rope` vs 同数学 dequant+rope rel 1.9e-4；整层 fused vs 同 fp8 数学参考 0.0（bf16 舍入），vs bf16 权重路径 2.27%（fp8 量化本身，符合预期）。
- kernel 数：host aten launch-ish 56→23；tilelang kernel 1→6；估计 GPU 启动 57→29（未到 ~10；剩余 q_a/rmsnorm、act_quant、wo_a+wo_b 以及若干 clone/cat/contiguous 小 op）。
- 源码：rmsnorm_rope 为 float4 + rsqrtf；gemm_rope 为 mma + float4/uint4 + 7×__syncthreads（TMA 被官方 pass_configs 关闭）。
- 约束：`head_dim % 128 == 0`、`rope_dim <= 128`；decode M=1 时 block_M=32 有 31/32 浪费，下一步可做 block_M=16 / GEMV+rope。

## 2026-10-21 第四轮：融合 Hyper-Connections（hc_fused.py）

`hc_mixes`（F.linear 24×20480 + rsqrt + 20 次 Sinkhorn）从 torch 分步压到 2 个 kernel。

真实 L0 hc_attn 权重、decode B=1,S=1,hc=4,D=5120、median of 5：

| 步骤 | torch | fused | 加速 |
|---|---:|---:|---:|
| 1 次 hc_mixes（linear+rsqrt+sinkhorn） | 1655.7 µs | 60.2 µs（mixes 55.9 + sinkhorn 4.5） | 27.5x |
| hc_pre | 58.6 µs | 23.1 µs | 2.5x |
| hc_post | 69.9 µs | 20.5 µs | 3.4x |
| 整层 HC（2 hc_mixes） | ~3.3 ms / 286 launch-ish op | ~0.12 ms / 4 kernel | ~27x |

- 正确性：pre 0、post 0、comb 6.3e-8；hc_pre/hc_post 0（vs `tinyinfer.hc`）。
- sinkhorn iters=6 vs 20：comb 偏差 15.6%，pre/post 不变 → 20 次不能省。
- 坑：官方 hc_post 的 comb 是**转置索引**（out[h]=Σ_h2 comb[h2,h]*res[h2]），按 comb[h,h2] 写会 rel≈0.26。
- 目标“整层 HC ~80 → ~4 kernel”达成（mixes+sinkhorn 各 2）；含 pre/post 为 8 kernel。
- 限制：hc_mixes 每行读 1.97 MB hc_fn（decode M=1 即一次 2 MB），prefill M>1 会按行重复读（L2 缓解）；后续可按 block_M 聚行优化。

## 2026-10-21 第五轮：decode 专用 fp8 GEMV（gemv_fp8.py）

M=1 时官方 fp8_gemm 的 block_M=32 浪费 31/32 行。GEMV 改为每个线程负责一个输出行的 k-lane、K 串行累加，coalesced 32B 权重读。

真实 L0、M=1、median of 5（official = kernel.fp8_gemm，同一 aq/asc）：

| 张量 | 形状 | official | 最佳 fused | 加速 | 有效带宽 | rel err |
|---|---|---:|---:|---:|---:|---:|
| wq_a | 1280×5120 | 37.5 µs | 26.9 µs（bf16 入口） | 1.39x | 244 GB/s | 1.4e-3 |
| wq_b | 32768×1280 | 251.7 µs | 139.7 µs（pq4） | 1.80x | 300 GB/s | 1.4e-3 |
| wkv | 512×5120 | 37.3 µs | 20.1 µs（bf16 入口） | 1.86x | 130 GB/s | 1.35e-3 |
| wo_b | 5120×8192 | 270.9 µs | 147.1 µs（bf16 入口） | 1.84x | 285 GB/s | 1.41e-3 |

- RoPE epilogue（wq_b/wkv，最后 64 列/head）正确性 1.40e-3 / 1.32e-3。
- **<60 µs 对 wq_b/wo_b 物理不可达**：41.9 MB / 实测 338 GB/s ≈ 124 µs 下界；当前 139.7 µs 已达 89% 有效带宽。wq_a/wkv 已达标（26.9/20.1）。
- 推荐：wq_a/wkv/wo_b 用 bf16 入口（无额外 act_quant launch）；wq_b 用 pq4（act_quant 14 µs + GEMV 140 µs = 154 µs，仍比 252 快 1.6x）；wkv 后接 fused_rmsnorm_rope。
- 四 GEMM ~597 → ~350 µs/层，40 层约省 10 ms/token。
- 变体：wq_b pq4(32/64)=139.7 < pq8=170.6；wo_b bf16=147.1 < pq8=173.2 < pq4≈200+（K=8192 时 4-FMA/迭代不划算）。
- 限制：32 B/warp/迭代的 sector 读，尚未做 16B 向量 load；下一步 16B/thread + 2 warps/行、或 wq_b 与 act_quant 单 kernel 融合。wo_a 是分组 8×(1024×4096)，本表未按单 GEMV 计。
