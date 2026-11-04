# vLLM kernel 对齐清单

核对方式：`torch._C._dispatch_get_all_op_names()` 过滤 `_C::` 和 `_moe_C::`。
**不要用 `dir(torch.ops._C)`** —— 它对 `_OpNamespace` 只返回一个 `name` 属性，
看起来像空命名空间，实际有 107 + 18 个算子。

## 装备情况

我们随包分发的是 `_C_stable_libtorch.pyd`（22.5MB）和
`_moe_C_stable_libtorch.pyd`（11.8MB），编进 **125 个算子**：

| namespace | 数量 |
|---|---:|
| `_C` | 107 |
| `_moe_C` | 18 |

## 硬件能力（RTX 5060 Laptop, sm_120）

| 查询 | 结果 |
|---|---|
| `cutlass_scaled_mm_supports_fp4(120)` | **True** |
| `cutlass_group_gemm_supported(120)` | **False** |

**SM100 → SM120 的真实差距**：不是算力。CUTLASS 里 SM120 有
`KernelTmaWarpSpecializedMxf4Sm120`（单 GEMM 的 MXFP4 UMMA），也有
`KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120`（pointer-array 的
block-scaled 调度），**唯独没有 "Mxf4 + PtrArray" 的组合 tag**。所以
`cutlass_mxfp4_group_mm` 在 SM120 上不可用。

要移植有两条路：

1. 改 CUTLASS：给 SM120 加一个 Mxf4 的 PtrArray 调度 tag（继承
   `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120` + mxf4 MMA），
   然后重编 vLLM 扩展。
2. 绕开 cutlass grouped，用 `_moe_C.moe_wna16_marlin_gemm`（Marlin 自己的 MoE
   kernel）。它支持 `float4_e2m1f`（见 marlin_moe.py:305-311），权重格式正好是
   我们已经在用的 `MarlinFp4Weight.qweight/.scales`。**但它要求全部专家堆成
   `[E, N, K/2]`**，我们是按专家流式的，这条要单独设计。

## `_moe_C`：18 个，全部 MoE 相关

```
batched_moe_align_block_size      moe_permute_sort_workspace_size
fused_globalize_align_block_size   moe_permute_unpermute_supported
grouped_topk                       moe_permute_with_scratch
moe_align_block_size               moe_prepare_scatter
moe_lora_align_block_size          moe_sum
moe_permute                        moe_unpermute
moe_wna16_gemm                     moe_wna16_marlin_gemm
shuffle_rows                       topk_sigmoid
topk_softmax                       topk_softplus_sqrt
```

**已接**：`topk_softplus_sqrt`（router，8.23x）。其余 17 个未接。

## `_C`：107 个，按用途分组（只列对我们有意义的）

**attention / MLA**
```
sm100_cutlass_mla_decode / sm100_cutlass_mla_get_workspace_size
merge_attn_states
fused_deepseek_v4_kv_rope_insert
fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
fused_qk_norm_rope / fused_qknorm_rope_cache
```

**MoE / 路由 / 量化**
```
fp32_router_gemm                 dsv3_fused_a_gemm
cutlass_mxfp4_group_mm           cutlass_fp4_group_mm（SM120 不可用）
cutlass_moe_mm / cutlass_moe_mm_data
get_cutlass_moe_mm_problem_sizes_from_expert_offsets
mxfp4_experts_quant              scaled_fp4_experts_quant
masked_moe_activation            persistent_masked_m_silu_mul_quant
```

**激活 / 归一化**
```
silu_and_mul                     silu_and_mul_with_clamp
silu_and_mul_quant               silu_and_mul_nvfp4_quant
silu_and_mul_per_block_quant     silu_and_mul_mxfp4_experts_quant
silu_and_mul_scaled_fp4_experts_quant
gelu_and_mul / gelu_tanh_and_mul / gelu_fast / gelu_new / gelu_quick
situ_and_mul / situ_and_mul_quant
fatrelu_and_mul                  mul_and_silu
rms_norm                         fused_add_rms_norm
rms_norm_static_fp8_quant        rms_norm_dynamic_per_token_quant
rms_norm_per_block_quant         minimax_allreduce_rms_qk
```

**量化（fp4/fp8/int）**
```
scaled_fp4_quant                 cutlass_scaled_fp4_mm
dynamic_scaled_fp8_quant         static_scaled_fp8_quant
dynamic_per_token_scaled_fp8_quant
per_token_group_fp8_quant        per_token_group_quant_int8
cutlass_scaled_mm                cutlass_scaled_mm_azp
marlin_gemm                      gptq_marlin_repack
cutlass_w4a8_mm                  cutlass_w4a8_moe_mm
```

## 我们已经接上的

| 算子 | 位置 | 状态 |
|---|---|---|
| `rms_norm` | `vllm_ops.rms_norm` | 已用 |
| `silu_and_mul` | `vllm_ops.silu_and_mul` | 已用 |
| `marlin_gemm` | `gemv_fp4_marlin` | 已用（4.8x） |
| `gptq_marlin_repack` | `MarlinFp4Weight` | 已用 |
| `scaled_fp4_quant` | 量化路径 | 已用 |
| `topk_softplus_sqrt` | `vllm_ops.gate_topk` | 已用（8.23x） |

**已装备未接（按价值排序）**：

1. `moe_align_block_size` + `moe_wna16_marlin_gemm` —— 消掉每层 8ms 的 host 同步，
   但要求全专家常驻，需要先解决权重布局。
2. `fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert` —— 把 qnorm+rope+kv
   写 cache 合成一个 kernel，DeepSeek 每层都跑。
3. `fp32_router_gemm` / `dsv3_fused_a_gemm` —— router 和 MLA 的 a-projection。
4. `silu_and_mul_with_clamp` —— 我们的 SwiGLU 带 `swiglu_limit` clamp，现在手写。
5. `sm100_cutlass_mla_decode` —— SM100 的 MLA decode，SM120 可能要改调度。
6. `moe_permute` / `moe_unpermute` / `moe_sum` —— 配合 grouped 路径。

## silu_and_mul_with_clamp 语义核对（可接）

vLLM 里 `act_first=true` 那一支（activation_kernels.cu:64-75）与我们的实现逐字对应：

| | 我们（moe.py:122-125） | vLLM |
|---|---|---|
| gate | `clamp(gate, max=limit)` 只压上限 | `g = fminf(g, limit)` |
| up | `clamp(up, min=-limit, max=limit)` 双边 | `u = fmaxf(fminf(u, limit), -limit)` |
| 输出 | `silu(gate) * up` | `PACKED_ACT_FN(gate) * u` |

签名：`silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit,
float alpha=1.0, float beta=0.0)`，input 是 `[..., 2*d]`，gate 在前。

**接线前提**：现在是两次独立 GEMV（`w1`→gate、`w3`→up），要拼成 `[T, 2N]` 才能喂。
把 w1/w3 在加载时合并成 `[2N, K]` 的一个 fp4 矩阵后，一次 GEMV 直接产出
`[gate|up]`，省一次 launch 和一次 cat。

## 已接上的第二个：silu_and_mul_with_clamp（4.59x）

`vllm_ops.clamp_swiglu(x, limit)`，接在 `Expert.forward_fp4` 里。

**接线方式有个坑**：本来想把 W1/W3 在权重上合并成 `[2N, K]` 一次 GEMV，结果
`_expert_from_rec` 里 `torch.cat` 直接 crash：
`!handles_.at(i) INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp:487`。
原因不是 cat 本身，是 `_blob_stage` 返回的是**环形缓冲区**，`t["w1.weight"]` 是它的
view，下一次 `tensors_on` 就会覆盖；而且权重的 cat 每个专家多 36MB，每步 6.4GB，
在这个 8GB 的卡上是自杀。

所以改成**只合并激活**：T=1 时 `[gate|up]` 只有 64KB，cat 几乎免费。

| | 原路径 | 现在 |
|---|---|---|
| 算子 | clamp(up) + clamp(gate) + silu + mul = 4 个 dispatch | cat + clamp_swiglu = 2 个 |
| 耗时 | 52.41 μs | **11.42 μs（4.59x）** |
| 数值 | — | max rel 7.2e-3，bf16 容差内 |

每步 6 专家 x 30 层 = 省约 7.2ms。

## 下一个目标：DeepSeek V4 的 norm+rope 融合（签名已拿到，未接）

```
fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    Tensor! q, Tensor kv, Tensor! k_cache, Tensor slot_mapping,
    Tensor position_ids, Tensor cos_sin_cache, float eps,
    int cache_block_size, bool apply_q_norm=True) -> ()
```

语义：对 q 做 q_norm(RMSNorm, eps) + RoPE（in-place 改 q）；对 kv 做 RoPE 后按
slot_mapping 写进 k_cache。

**接它要解决的差异**：

| | 我们（attention.py:179-199） | vLLM |
|---|---|---|
| RoPE 表 | `freqs_cis[start:start+s]` | `cos_sin_cache` + `position_ids` |
| q norm | `rms_norm(_qkv[..., :q_lora], q_norm, eps)` | 同一支 kernel 里做 |
| cache | window + compressed 两套 | paged `k_cache` + `slot_mapping` |

所以这不是换个函数调用，得先确认我们的 cache 布局能不能对上 slot_mapping。

## SM120 移植：CUTLASS 层已打通（编译验证通过）

**问题**：SM120 的 F4 白名单（`sm1xx_common.inl`）里只有单 GEMM 的 tag，没有
pointer-array（grouped）变体；而 ptr-array 的 SM120 tag 继承的是
`KernelPtrArrayTmaWarpSpecializedCooperative`，不继承 `KernelScheduleMx*` 基类，
所以 CUTLASS 在编译期直接拒绝：

```
static assertion failed: "Incorrect Kernel Schedule Policy for F4 type inputs."
static assertion failed: "Kernel Schedule policy should be auto"
```

**补丁（三处，全部在 CUTLASS 源码里）**：

1. `include/cutlass/gemm/dispatch_policy.hpp`，在 `KernelScheduleSm120Blockwise` 前加：

```cpp
struct KernelSchedulePtrArrayBlockScaledGemmSm120 : KernelScheduleBlockScaledGemmSm120 {};
struct KernelSchedulePtrArrayMxf8f6f4Sm120        : KernelSchedulePtrArrayBlockScaledGemmSm120 {};
struct KernelSchedulePtrArrayMxNvf4Sm120          : KernelSchedulePtrArrayBlockScaledGemmSm120 {};
struct KernelPtrArrayTmaWarpSpecializedCooperativeMxf4Sm120
    final : KernelSchedulePtrArrayMxNvf4Sm120, KernelPtrArrayTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedPingpongMxf4Sm120
    final : KernelSchedulePtrArrayMxNvf4Sm120, KernelPtrArrayTmaWarpSpecializedPingpong { };
```

2. `include/cutlass/gemm/collective/builders/sm1xx_common.inl`，F4 白名单（`check_input_datatypes`）：

```cpp
is_base_of_v<KernelSchedulePtrArrayMxf8f6f4Sm120, BuilderScheduleTag> ||
is_base_of_v<KernelSchedulePtrArrayMxNvf4Sm120, BuilderScheduleTag>   ||
```

3. 同文件的两个分支（`Mxf8f6f4` 与 `MxNvf4`）各自加上对应的 ptr-array 判断 ——
   否则 policy 落到 `else` 分支会撞 `"Kernel Schedule policy should be auto"`。

**验证**：`_sm120probe.cu`（纯 CUTLASS，不依赖 torch）用
`KernelPtrArrayTmaWarpSpecializedCooperativeMxf4Sm120` 实例化
`CollectiveBuilder<Sm120, OpClassBlockScaledTensorOp, mx_float4_t<e2m1>, ...>`，
`nvcc -arch=sm_120a` **编译通过 (exit 0)**，`InternalLayoutSFA/SFB` 都能解析出来。

**还没做完的两步**：

1. `EpilogueSchedule`：SM100 用 `PtrArrayTmaWarpSpecialized1Sm`，SM120 要换成对应的
   ptr-array epilogue tag。
2. `PersistentTileSchedulerSm100GroupParams`：SM120 需要对应物，否则 grouped 的
   tile 调度编不过。
3. 然后才是把 `mxfp4_blockwise_moe_kernel.cu` 整体实例化、编成一个可加载的扩展，
   并重编 vLLM 的 `.pyd`（当前 pyd 里没有这个新 kernel）。

**注意**：动了 `_deps/cutlass-src`，rebuild vLLM 时会用这份改过的 CUTLASS；
重装/清 CMake 会把这些改动冲掉，需要重新打补丁。

### SM120 完整 grouped MXFP4 GEMM 已编译通过（exit 0）

`_sm120full.cu` 把 `mxfp4_blockwise_moe_kernel.cu` 的完整类型链在 SM120 上实例化：

```cpp
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t,int32_t,int32_t>>;
using ElementA    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementC    = cutlass::bfloat16_t;
using MmaTileShape = Shape<_128,_128,_128>;
using ArchTag     = cutlass::arch::Sm120;
using KernelSchedule =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeMxf4Sm120;  // 新加的
using EpilogueSchedule =
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;             // 关键：不是 1Sm
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>>;
```

**踩到的第二个坑**：epilogue 一开始照抄 SM100 的 `PtrArrayTmaWarpSpecialized1Sm`，
SM120 的 epilogue builder（`sm120_builder.inl:485-491`）白名单里**没有它**，直接
`"Could not build a collective epilogue for given parameters."`。SM120 接受的是
`PtrArrayTmaWarpSpecializedCooperative` / `Pingpong` / `TmaWarpSpecialized*` /
`SparseTmaWarpSpecializedCooperativeSm120`。

**scheduler 不用改**：`PersistentTileSchedulerSm100GroupParams` 只是包了
`PersistentTileSchedulerSm90GroupParams`（`tile_scheduler_params.h:2420`），
架构无关，编译时 `typename Gemm::GemmKernel::TileScheduler::Params` 解析正常。

**距离真正用起来还剩一步**：把这个 kernel 编进可加载的扩展（torch::stable 那层封装
照 `mxfp4_blockwise_moe_kernel.cu` 抄），然后重编 vLLM 的 `.pyd`。

### SM120 MXFP4 grouped MoE GEMM 已跑通（数值 + 性能）

**编译链路打通后实测**（RTX 5060 Laptop，sm_120）：

数值（E=4, N=512, K=512, M=32，cosine diff 是官方 `test_mxfp4_moe.py` 的度量）：

```
per-expert cosine diff: [0.01291, 0.01234, 0.01297, 0.01295]
max diff: 0.01297  (official threshold 0.15)
```

性能（DeepSeek MoE 形状，E=6, N=2048, K=7168，T=1 每专家 1 行）：

```
weights: 44.0 MB
GEMM one shot: 182.4 us  -> 241.5 GB/s   (峰值 295 GB/s 的 82%)
```

对比：Marlin 的 fp4 GEMV 是**每个专家** 135 μs，6 个专家循环 = 810 μs；
grouped 一次 kernel 182 μs，**4.45x**。

**为了让它在 SM120 上真的跑起来，一共改了四处（都在 vLLM 源码里）**：

1. `CMakeLists.txt` SM12x 的 `FP4_SM120_SRCS` 加 `mxfp4_blockwise_moe_kernel.cu` 和
   `mxfp4_experts_quant.cu` —— 原来这两个只在 `FP4_SM100_SRCS` 里。
2. `mxfp4_blockwise_moe_kernel.cu`：#if 门控加 `ENABLE_NVFP4_SM120`，ArchTag
   `Sm100 → Sm120`，KernelSchedule `...1SmMxf4Sm100 → ...CooperativeMxf4Sm120`，
   EpilogueSchedule `PtrArrayTmaWarpSpecialized1Sm → PtrArrayTmaWarpSpecializedCooperative`，
   运行时 `version_num >= 120` 走新分支。
3. `mxfp4_experts_quant.cu`：`mxfp4_experts_quant_sm_supported` 的
   `capability < 120` 改成 `< 130`（它其实只要求 CUDA >= 12.9，不依赖 SM100 指令）。
4. CUTLASS 三处补丁（见上一节）。

**两个调用约定上的坑**：
- `mxfp4_experts_quant` 是 out 参数式：`(output, output_scale, input, expert_offsets,
  blockscale_offsets, n_experts)`，`output_scale` 要按**行数**分配
  `max_tokens * padded_k`（`padded_k = ceil((K/32)/4)`），权重量化时行数等于 `E*N`。
- `cutlass_mxfp4_group_mm` 的 `expert_offsets`/`sf_offsets` 长度必须和
  `problem_sizes` 的专家数**相等**，即传 `[:-1]`（去尾部哨兵）。

**下一步**：把这条 grouped 路径接进 `moe.py`，替换掉「每专家一次 Marlin GEMV」的
Python 循环，同时用 `moe_align_block_size` 消掉每层 8ms 的 D2H 同步。

### SM120 decode 上的取舍：grouped MXFP4 vs 手写 moe_gemv（移植保留）

实测（RTX 5060，E=6 专家，gate_up 投影，N=2048 K=7168，M=1）：

```
moe_gemv.gate_up : 312.0 us   (native/moe_gemv6.cu，手写)
grouped gate_up  : 328.2 us   (cutlass_mxfp4_group_mm + 堆叠权重)
堆叠代价          : 22.3 ms   ← 每层每步，不可接受
```

**SM120 decode 上暂时保留手写 moe_gemv，但 grouped 路径继续维护（见下）。** 原因是 M=1 时 grouped 每专家仍开
128×128 tile（127/128 行空着），tile 调度 + TMA descriptor 重建的开销吃掉了收益；
而 `moe_gemv6.cu` 把专家索引用 grid.y 承载、fp4 权重/e8m0 scale/fp8 激活全在 kernel
内解码，一层 4 次 launch。

**多硬件策略（重要）**：引擎不止跑在这一张卡上。

| 硬件 | 架构 | MXFP4 grouped | 本模型使用的路径 |
|---|---|---|---|
| RTX 5060 Laptop | sm_120 | 用这里移植的 SM100 kernel | grouped / moe_gemv 按形状选 |
| H200 | sm_90 (Hopper) | 不可用（Blackwell 特性） | SM90 原生（FP8 / Marlin FP4） |

所以 grouped 的移植**不是替代关系，是 SM12x 独有的能力补齐**：
H200 那条路走 vLLM 的 SM90 实现，SM12x 这条路走移植版。两条都保留。
prefill（M >= 32）和未来"多专家权重常驻"的架构仍然需要 grouped。

### SM120 的 MLA：硬限制在 shared memory，不在调度

**能编，不能跑。** `Sm100FmhaMlaKernelTmaWarpspecialized` 在 `-arch=sm_120a`
下**实例化、编译、链接全部通过**（exit 0）。但运行时 `initialize()` 返回
`kErrorInternal`，位置是：

```cpp
int smem_size = Kernel::SharedStorageSize;
cudaFuncSetAttribute(device_kernel<Kernel>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
if (cudaSuccess != result) return Status::kErrorInternal;   // sm100_mla.hpp:264
```

**根因**（实测设备属性）：

```
RTX 5060 Laptop (sm_120, 消费级 Blackwell)
  shared_per_block_optin : 101376  (99 KB)
  shared_per_sm          : 102400  (100 KB)

B200 (sm_100, 数据中心 Blackwell)
  shared per SM         : 228 KB
```

MLA 的 `TileShape = Shape<_128, _128, Shape<_512, _64>>`，单个 bf16 KV stage 是
`128 * (512 + 64) * 2 = 147 KB`，**超过 SM120 的 99 KB 上限**，所以属性设置失败。

**缩小 tile 这条路被硬件堵死**（试过 `TileShapeH=64`）：

```
error: static_assert(TileShapeH{} == 128);                       // kernel 源码硬断言
error: "SM100_MMA_F16BF16_2x1SM_SS M-mode size should be 128 or 256 for 2 CTA cluster MMA."
error: "Invalid TileShape_M."                                    // M 必须是 128 或 256
```

**唯一的 SM120 出路**：用 **fp8 (e4m3) KV cache**，stage 从 147 KB 减到 ~73 KB。
`sm100_cutlass_mla_kernel.cu` 本来就有 `float8_e4m3fn` 的 dispatch 分支
（`runMla<float_e4m3_t, bfloat16_t, ...>`），代价是 KV cache 格式改动 + 精度损失。
在 bf16 cache 下，SM120 跑不了这个 MLA kernel。

**H200 不受影响**：sm_90 有 228 KB/SM，且本模型在 Hopper 上走的是 vLLM 的 SM90
注意力后端，不是这个 kernel。

## 双架构构建（SM90 + SM120）：一晚踩掉的全部坑

目标：一个 `_C_stable_libtorch.pyd` 同时服务 RTX 5060 (sm_120) 与 H200 (sm_90)。

```bat
set TORCH_CUDA_ARCH_LIST=9.0a;12.0a
ninja -j 2 _C_stable_libtorch _moe_C_stable_libtorch
```

**产物**：`_C` 33.01 MB（原 22.9）、`_moe_C` 23.7 MB（原 11.8）；
`cuobjdump --list-elf` 确认含 `sm_90` 与 `sm_120`。

### 必须在 Windows 上修的六个 bug

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 1 | `CMakeLists` machete | `ModuleNotFoundError: vllm_cutlass_library_extension` | `PYTHONPATH` 用冒号拼（Linux 写法）；Windows 只认分号。改用 `set(ENV{PYTHONPATH})` 让子进程继承（分号放进 `cmake -E env` 参数会被拆成 list） |
| 2 | `CMakeLists` marlin | `Marlin generation failed. Result: "1"`，但日志为空 | `RESULT_VARIABLE` 与 `OUTPUT_VARIABLE` **同名**，stdout 覆盖了返回码。改名 |
| 3 | `utils.cmake` / 生成脚本 | `kernel_selector.h(167): fatal error C1061: 块嵌套太深` | `generate_kernels.py` 生成 876 条平铺 `else if`；MSVC 按分支算嵌套。改成每 **20** 条开一个新的 `if (kernel == MarlinDefault)` 块（43 段），876 个 kernel 一个不少 |
| 4 | `tensor_fill.h` | `error: identifier "NAN" is undefined` | MSVC 的 `<cmath>` 会清掉 C 的 `NAN` 宏。在 compat 头里 `#define` 也无效（被 `<cmath>` 覆盖）。直接把该文件里两处 `Element(NAN)` 换成 `cutlass::platform::numeric_limits<Element>::quiet_NaN()` |
| 5 | `w4a8_mm_entry.cu` | `threadIdx/curand` 在主机函数里 | 该 TU 拉进了 CUTLASS 设备端参考实现，MSVC 主机 pass 编不过。w4a8 是 GPTQ/AWQ 的 int4 路径，用不上 → `VLLM_BUILD_W4A8` 默认关 |
| 6 | machete 本身 | 同上 | 同样是 int4 路径 → `VLLM_BUILD_MACHETE` 默认关 |

### 内存不足（WinError 1455 / catastrophic error）

C 盘页面文件只有 1 GB，双架构并行编译把它顶穿。
修法：`ninja -j 2` + `set TMP=E:\tmp\vllmbuild`。编译耗时约 6 小时（全量）/ 30 分钟（增量）。

### ⚠️ 测性能前必须确认 GPU 空闲

这一夜最大的假警报：改完后 marlin 从 305 GB/s 掉到 47 GB/s，排查了两小时。
真相是 GPU 被别的进程占满：

```
utilization.gpu : 97 %
temperature     : 85-86 C
clocks.sm       : 2460-2497 / 3090 MHz   (降频)
torch.sum 42MB  : 35 GB/s   ← 纯读也只有 35
dst.copy_ 42MB  : 50 GB/s
marlin          : 51 GB/s   ← 比 sum 还快
```

**Marlin 没退化，是整卡带宽塌了。** 基准测试前先跑：

```powershell
nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu --format=csv
```

利用率不为 0 或 clocks 明显低于 max 时，任何数字都不是 kernel 的。

### 双架构 pyd 回归验证（GPU 空闲，2026-11-04 04:30）

**结论：无回归。**

```
RTX 5060 (sm_120), 空闲状态
  torch.sum 42MB  : 149.9 us -> 274 GB/s   (纯读基线)
  dst.copy_ 42MB  : 296.8 us -> 276 GB/s
  marlin mxfp4    : 136.6 us -> 307 GB/s   (N=16384 K=5120, 41.9MB)
  grouped MXFP4   : 328.3 us -> 268 GB/s   (E=6, 2N=4096, K=7168, 88MB)
```

对比单架构基线（136.2 us / 308 GB/s），Marlin 完全一致。
算子总数 126（`_C` 108 + `_moe_C` 18），全部可调用。

**端到端**（DeepSeek-V4.1-Flash-1017，40 层全常驻，`OptLevel.BALANCED`）：

```
prompt : "Hello, I am DeepSeek."
output : " I am currently working on a new project. I am a"
```

输出连贯正确。注意：`layers=list(range(19))` 会被内存预算压到 10 层，
此时输出必然是乱码（模型不完整），不要把那个当成 kernel bug。

**未解决的构建告警**（不影响正确性）：
```
CUDA graph unavailable (Cannot copy between CPU and CUDA tensors during
CUDA graph capture unless the CPU tensor is pinned)
```
decode 回落到 eager 模式。要恢复 CUDA graph 需要把捕获路径上的 CPU tensor
改成 `pin_memory=True`，这是一项独立改动。

## 移植进度：cubin 路线（native/vllm_*.cu）

**决定**：不编译 vLLM 的 pyd，改为把 vLLM 的 device 代码抄进 `src/tinyinfer/native/`，
用现有的 `nvcc -cubin` + driver API 方式加载。理由（实测）：

| | cubin + driver API | vLLM pyd |
|---|---|---|
| 单文件重编 | 秒级 | 30 分钟（增量） |
| 源码归属 | 我们的仓库 | 别人的二进制 |
| 参数顺序 | 我们定 | 跟 vLLM |
| 调用开销 | 11.7 us（BoundLaunch） | 8.0 us |
| 可改 kernel | 能 | 不能 |

调用开销差 3.7 us 是 ctypes 打包的代价，不是 kernel 差。

### 已移植

**`native/vllm_rms_norm.cu`**（源：`csrc/layernorm_kernels.cu` 的
`rms_norm_kernel<T, VEC, 2, true>`）

- 保留：16 字节向量读（`VecBF16`，8 个 bf16）、每行一个 block、vec_size = 8
- 改动：去掉 torch 外壳，改为裸 `extern "C" __global__`；
  `cub::BlockReduce` 换成 warp shuffle + shared 广播（不引入 CUDA toolkit 头）
- **数值**：M=1/8 与 vLLM 逐位相同（diff = 0.0）；M=64 有 1e-4 差异（归约顺序）

**性能（纯 GPU 时间，CUDA event 实测）**：

```
block=256   ours 11.19 us   vLLM _C 8.41 us   (0.75x)
```

**尚未追平的原因**：vLLM 用 `cub::BlockReduce`（tree reduction），我们用
shuffle + 两次 `__syncthreads()`。O3/--use_fast_math 无效（反而更慢），
block size 128-1024 扫描也无效。要追平得把 cub 的 `BlockReduce` 头拉进来，
或手写 tree reduction。

**但优先级很低**：rms_norm 每层调 2 次，40 层共 80 次，
3.7us × 80 = 0.3ms/token，对一个 560ms 的 decode step 是 0.05%。

### `native.py` 新增

- `BoundLaunch` / `bind()`：预解析 kernel + 签名 + 网格，稳态调用
  16.4us -> 11.7us（generic `launch()` 保留给一次性调用）
- `_current_stream_ptr()`：按 raw handle 缓存 stream 指针，
  绕过每次 `torch.cuda.current_stream()` 的 1.5us

## 移植完成：DeepSeek 推理路径全部走我们自己的代码

**规则**（cong 2026-11-04）：禁止在 DeepSeek 路径上使用「直接来自 vLLM 的代码」。
抄进本仓库的源码是我们自己的，可以编译、可以改；被禁止的是编译 vLLM 仓库再调用，
以及依赖它的 `_C_stable_libtorch.pyd`。

### 已抄进 native/ 的四组

| 本仓库文件 | 上游 | 数值对比 |
|---|---|---|
| `native/vllm_rms_norm.cu` | `layernorm_kernels.cu` | 与 pyd **逐位相同** |
| `native/vllm_silu_and_mul.cu` | `activation_kernels.cu` | 1-2 bf16 ulp |
| `native/vllm_topk_softplus_sqrt.cu` | `moe/topk_softplus_sqrt_kernels.cu` | idx/w **逐位相同** |
| `native/marlin/`（6 个头 + 实例化） | `quantization/marlin/` | GEMV 与 repack **逐位相同** |

Marlin 一组包含 `marlin_template.h`、`marlin_mma.h`、`dequant.h`、`marlin.cuh`、
`marlin_dtypes.cuh`、`kernel.h`、`scalar_type.hpp`，加上我们自己的
`marlin_compat.h`（把 torch 异常换成 std）、`instances.cu`（只实例化实际用到的
`Marlin<bf16, fp4, bf16, e8m0, 256,1,8,8,true,4,2,false>`）、
`repack.cu`（`gptq_marlin_repack_kernel<256,4,false>`）。
两个 cubin：`marlin_fp4.cubin`（50 KB）、`marlin_repack.cubin`（17 KB）。

### `native.py` 的两处改动

- `BoundLaunch` 预解析 kernel + 签名 + 网格 + **预分配参数槽**。Marlin 的 18 个参数
  原来每个都新建 ctypes 对象，实测 26us/次；改成复用 8 字节槽 + 一次建好的指针数组后
  19.7us，**比原来的 C++ 绑定（20.2us）还快**。
- `set_dynamic_smem()`：>48KB 动态 smem 必须先 `cuFuncSetAttribute`，Marlin 用满 99KB。

### 实测

单算子（N=2048 K=7168，专家形状）：

```
cubin  19.7 us
pyd    20.2 us
```

端到端（40 层，同一进程配置与 prompt，输出逐字相同）：

```
cubin  1466 ms/token  -> 0.68 tok/s
pyd    1768 ms/token  -> 0.57 tok/s
TEXT: ' I am currently working on a new project. I am a software engineer'
```

## 同时修掉的四个真 bug

1. **`BALANCED.use_fp4_gemm = True`** —— tier3 的注释写着 fp4 kernel
   "must never be the default of any tier"，tier1 却一直开着（~20% 每层误差）。
2. **`AGGRESSIVE.kv_bits = 8`** —— 让 tier2 精度低于 tier1，而两档应同精度只差机制。
3. **路由器被 vision 开关挡住** —— `Gate.forward` 要求 `bias_vl is None` 才走融合路由，
   而本模型 `vision_enabled=True`，所以文本推理从来没走过融合路由（一直 softplus+argsort）。
4. **host pool 上限过高** —— 28GB 池在 33GB 空闲 RAM 下建完计划就死在
   `cuMemSetAccess out of memory`；16GB 正常。改成按物理内存 34% 设上限。

## 还没做的

无。DeepSeek 推理路径上的 kernel 已全部来自本仓库编译的 cubin；
`_C_stable_libtorch.pyd` / `_moe_C_stable_libtorch.pyd` 只剩兜底作用，
删掉它们引擎仍可运行（`vllm_ops` / `gemv_fp4_marlin` 的 pyd 分支会走不到）。
