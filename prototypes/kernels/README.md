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
