# 贡献

## 环境

见 [docs/ENV.md](docs/ENV.md)，Windows 用 [scripts/bootstrap.ps1](scripts/bootstrap.ps1)，
Linux 用 [scripts/bootstrap.sh](scripts/bootstrap.sh)。

自写 kernel 需要 CUDA toolkit 的 `nvcc`：

```powershell
.\scripts\build_native.ps1        # 编译到 src/tinyinfer/native/*.cubin
```

## 改代码前

```powershell
.\scripts\check.ps1              # 语法 + 导入 + 单测
```

## 约定

**正确性先于速度。** 改 kernel 或融合算子时，先在相同输入上与参考实现逐位比对，再测速。
仓库里每个"更快"的数字都伴随一个"输出与参考一致"的验证；只报速度不报一致性不算完成。

**数值差异要写清楚。** 用重排归约换速度时，在提交信息里写明差异量，例如
"state diff 1.5e-8，贪心输出前 20 token 一致"。

**一个模型一条路径。** 新模型接入走 `adapters.py` + `Backend` 协议，不加旁路脚本；
客户端只连端口，模型选择权在启动引擎的人手里。

## 提交

提交信息说明**为什么**这样改，而不是罗列改了什么。实测数字要写明测试条件。

## Kernels: copy a proven implementation before writing one

Every kernel starts from an upstream implementation whenever one exists. vLLM's csrc/ and
llama.cpp's ggml/ are checked out under repositories/ for exactly this; read them before
opening an editor.

This is not a style preference. Measured on this machine (RTX 5060 Laptop, sm_120, fp4
experts N=16384 K=5120, 40 MB of weights, three warm-up rounds):

| implementation | time | bandwidth |
| --- | --- | --- |
| tinyinfer fp4_linear -> vendored vLLM Marlin (mxfp4) | **136.2 us** | **308.0 GB/s** |
| vLLM Marlin run standalone (mxfp4, M=1) | 142.9 us | 293.6 GB/s |
| tinyinfer, hand-written scalar decode | 657 us | 59 GB/s |
| tinyinfer, tilelang fallback | 1044 us | 40 GB/s |

Writing it from scratch cost five iterations (table lookup, ALU bit-assembly, SWAR, tensor
core with manual fragments, 16-byte vector loads) and the best of them reached 4.8x SLOWER
than the kernel that was already sitting in repositories/vllm/csrc/quantization/marlin/.
The gap is not in the arithmetic; it is in the four-stage cp.async pipeline, the ldmatrix
fragment loads, and the k-major interleaved weight repack -- three things Marlin already
does.

Vendoring it took one file (src/tinyinfer/gemv_fp4_marlin.py, ~150 lines): repack the
weights once at load with gptq_marlin_repack, permute the e8m0 scales into the block order
the dequantiser reads, then call marlin_gemm. Correctness is checked against a hand decode
of E2M1 (max relative error 4.7e-3, i.e. bf16 activation precision). Full numbers in
docs/vllm-baseline.md.

So: vendor the upstream kernel, adapt its interface, keep its name in a comment with the
commit or version it came from, and only then consider a replacement. A from-scratch kernel
has to beat the vendored one on a measurement before it replaces it.
