# 在 Windows / MSVC 上编译 vLLM 的 CUDA 扩展

目标：不用 WSL，用原生 MSVC + nvcc 把 vLLM 0.33.0 的 `csrc` 编成两个 ABI3 扩展。
产物 `_C_stable_libtorch.pyd`（22.5 MB）和 `_moe_C_stable_libtorch.pyd`（11.8 MB）
已经放在 `src/tinyinfer/` 下，tinyinfer 直接加载它们，不依赖 vllm 的 Python 包。

## 环境

- Windows + MSVC 19.44（VS 2022 BuildTools）
- CUDA 13.0，通过 `C:\cuda` 这个无空格 junction 暴露给 nvcc（路径里有空格会破坏 ninja 的响应文件）
- torch 2.14.0+cu130（0.33.0 要求；旧版 torch 没有 `torch/headeronly/core/Dispatch.h`）
- ninja
- 源码：vLLM commit `1c879919786497217b3b5fb0258d1d72f2442cb0`

## 三个必须的构建参数

1. **`-DCMAKE_CUDA_FLAGS="--pre-include=<root>/csrc/vl_msvc_compat.h"`**。
   nvcc 不读 `NVCC_PREPEND_FLAGS` 环境变量，这个头文件不会自动生效——曾经用它
   悄悄失败了一整轮构建，所有修补都没进编译器。确认方式：`build.ninja` 里
   `pre-include` 出现几百次。
2. **只构建这两个目标**：`ninja _C_stable_libtorch _moe_C_stable_libtorch`。
   不带目标名会连 flash-attn 等子项目一起编，103 个失败。
3. 链接 cuBLAS：`q_gemm.cu` 直接调 `cublasHgemm`，CMakeLists 里要为
   `_C_stable_libtorch` 加 `target_link_libraries(... PRIVATE cublas)`，
   否则 LNK2019。

## csrc/vl_msvc_compat.h 里有什么

强制预包含，补 MSVC 缺的 GNU 扩展：

- `_USE_MATH_DEFINES` + `<cmath>`，以及手写的 `M_SQRT2`/`M_LOG2E`/`M_PI`/`M_LN2`
- `__builtin_clz`/`ctz`/`popcount`/`clzll` 用 `_BitScanReverse` 实现
- `__int128_t` / `__uint64_t` / `uint` 别名，`#define __attribute__(x)`

**不要**在这里 `#define NAN` 或 `INFINITY`：`(0.0f/0.0f)` 在 cl 下是硬错误
C2124（除零），而且 `constexpr float x = -INFINITY` 也会报「浮点值放不进」。
需要 NaN 的地方用 `std::numeric_limits<float>::quiet_NaN()`。

## 源码层面的改动

| 问题 | 处理 |
|---|---|
| `__asm__ __volatile__` | Windows nvcc **接受** `asm volatile`，不接受 `__asm__`。26 处改回 `asm volatile` |
| 宏参数里出现 `#if`/`#endif` | MSVC 报 `"#" not expected here`。把 `USE_ROCM` 条件块整体折平（82 处 `#ifndef`、38 处 `#ifdef`，共 23+19 个文件） |
| `__attribute((aligned(128)))` 单下划线 | 不在宏覆盖范围，改 `__align__(128)` |
| `variable template quant_type_max_v<T>` | 改成函数模板 `quant_type_max_v<T>()`，所有调用点补 `()` |
| `::isinf`/`::isnan` 从 device 调 host 版 | 改 `::isinf`/`::isnan` |
| `__CUDA_ARCH__` 在 host pass 未定义导致 bf16 特化缺失 | `type_convert.cuh` 的守卫改成 `!defined(__CUDA_ARCH__) \|\| __CUDA_ARCH__ >= 800` |
| `std::bit_width` 用于 `next_pow_2` | 补 `#include <bit>` |
| marlin 的 875 层 `else if` 链 | MSVC 硬限制 C1061（块嵌套太深）。整条链改写成 876 个扁平 `if (...) { kernel = ...; goto marlin_sel_done; }` + 结尾一个 `marlin_sel_done:;`。两个 `kernel_selector.h` 都要改 |
| cutlass `tensor_fill.h` 里的 `NAN` | 该头文件在注释块中间被注入了宏，等于没生效——要在文件最前面、注释之后加 `#include <limits>` + `#define NAN (std::numeric_limits<float>::quiet_NaN())` |
| `grouped_topk_kernels.cu` 的 `constexpr float InvalidScore = -INFINITY` | 改 `-std::numeric_limits<float>::infinity()` |

## 构建

```powershell
# 配置（约 24 秒，不会重新下载依赖）
cmake -S <vllm> -B <vllm>/build_cmake -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_FLAGS="--pre-include=<vllm>/csrc/vl_msvc_compat.h" ...

# 构建
ninja _C_stable_libtorch _moe_C_stable_libtorch
```

## 验证

```python
import torch
import tinyinfer._C_stable_libtorch   # torch 必须先导入，否则 DLL 目录没注册
import torch
assert hasattr(torch.ops._C, "marlin_gemm")
```

失败的典型症状：
- `DLL load failed: 找不到指定的模块` —— 忘了先 `import torch`
- `LNK2019 cublasHgemm` —— 没加 cublas
- `C1061 块嵌套太深` —— marlin 的选择器还是 `else if` 链
