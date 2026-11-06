# 性能测量笔记

这台机器（RTX 5060 Laptop，384 GB/s，8 GB）上的测量结果。写下来是因为好几个结论
和直觉相反，而且都不止一次被重新调查过。

## 一次 decode step 到底花在哪

```
795 ms/step  (端到端)
├─ 96 ms   GPU kernel
├─ 338 ms  GPU memcpy (H2D，PCIe 已跑满 34 GB/s)
└─ 361 ms  CPU + IO 等待（两者几乎不重叠：795 ≈ 434 + 361）
```

kernel 那 96 ms 已经贴着带宽：`moe_gemv6` 353 GB/s、`fp8_gemv_bs` 冷读约 192、
`dequant_fp8` 309、`gemv_fp4_w4a8_e8m0` 300（数字都是相对 384 的额定值）。

**所以在这台机器上继续抠 kernel 的边际收益很小**，剩下的大头是 memcpy（硬件）和
CPU（launch 次数）。CPU 侧每步约 3840 次 kernel launch、4619 次 H2D、3030 次
`torch.empty`、18773 次 `aten::view`。

## slab 拿不到内存不是 bug，是当前最优

`_flat_setup` 会为流式层分配每层一块的 pinned slab（一层的所有权重一次 DMA，而不是
115 次小拷贝）。本机经常打印：

```
[flatdbg] pool total=18.76GB used=18.76GB free=0.00GB slab_total=6.51GB
[tinyinfer] slab unavailable (ok); falling back to per-tensor pinned
```

原因是专家缓存（12.6 GB）把 pinned 池吃光了，slab 需要 6.51 GB。把 `TI_CACHE_GB`
降到 6 GB 能让 22/30 层用上 slab，但代价是缓存命中率崩掉：

| 配置 | slab 覆盖 | 端到端 |
|---|---|---|
| cache 13 GB（默认） | 0/30 | **795 ms** |
| cache 8 GB | 12/30 | — |
| cache 6 GB | 22/30 | 1471 ms |

**专家缓存远比 slab 重要。** 不要再试图"修好" slab。

## fused MoE 和逐专家 GEMV 是分工，不是重复

profiler 里 `moe_gemv6`（30/步）和 `gemv_fp4_w4a8_e8m0`（约 720/步）同时出现，看着
像重复计算。实际是：

```
bf16 缓存命中的专家 -> Expert.forward_fp4     逐个算（167.8/步）
缓存未命中的专家   -> moe_gemv.gate_up/down  一次 6 个（30/步）
```

命中的专家已经在显存里，等其余专家 fetch 完再一起 fused 会丢掉那层 overlap，所以
这个分工是对的。两条路径的吞吐也相当（6 专家逐专家 ≈ 376 us，fused ≈ 372 us）。

## 数值是脆弱的：只有逐位等价才安全

这个模型 40 层累积下来，任何数值扰动都会改掉解码出的 token。实测失败并回滚的：

| 改动 | 破坏方式 |
|---|---|
| compressor 权重保留 bf16（省 950 MB 显存） | bf16 vs fp32 GEMM 累加不同 |
| `wo_a` 保持 fp8 做 grouped GEMV（省 12 ms） | fp8 GEMV vs bf16 einsum 累加顺序不同 |
| `fp8_gemv_bs` 改成一 lane 一组 scale | 归约顺序变，输出 `[223,19,271,20]` |
| 融合 kernel 让编译器把 `m*s+b` 收缩成 FMA | 一次舍入 vs 两次，1.2e-7 就够改 token |
| 改用 `__expf` | 末位不同 |

注意上面第二条：那个改动**数学上是完全等价的**。`wo_a` 存的是 e4m3，把它的值展开成
bf16 是无损的（4 位尾数 ⊆ 7 位），所以 fp8×fp8 和 bf16×bf16 乘的是同一组数。变的只有
累加顺序，而 40 层累积下来足够改掉 token。

**所以判断标准不是"数学上等价"，而是"逐位等价"。** 一个优化只有在能证明输出字节
完全相同时才安全，否则就是一次赌博 —— 这个模型上到目前为止赌输的比例是 100%。

**写融合 kernel 必须用 `__fmul_rn` / `__fadd_rn` 之类的显式舍入 intrinsic**，否则
nvcc 默认的 FMA 收缩会悄悄改掉结果。`hc_split.cu` 就是这么做到逐位一致的
（三路 maxdiff 全 0.0）。

## 已验证正确、本机却跑不快的东西

* **MTP/DSpark**：`TI_MTP=1` 能强制加载，输出与不投机时完全一致
  (`[223,19,17,20]`)，但本机 1665 ms/token —— 草稿层挤掉了流式层的显存。专业卡上
  显存够时才会赚。planner 会自己关掉它。
* **prefill 波长**：`TI_CHUNK=8` 和 `16` 输出完全相同（`[1,223,6895]`），所以
  planner 里 64 的上限放开到 4096 是安全的（4090→90，H100→930，H200→1845）。
* **tensor-core prefill**：`fp8_gemm_mma` 在 M=64 以上才划算（M=64 时 128 us
  对逐行 2.6 ms，M=128 时 154 us 对 5.1 ms）。M=1 时反而慢 3 倍（119 us 对 40 us），
  所以 decode 留在 GEMV。默认仍由 `TI_MMA_GEMM=1` 显式打开，因为 prefill 走 mma
  会改变解码出的 token。

## 多卡布局（照 vLLM 的 TP + EP）

attention 是 TP（每卡持有部分 head），MoE 是 EP（每卡持有部分专家，输出相加）：

```
wq_a   column-parallel  [dim -> q_lora_rank]          每卡一段输出
wq_b   column-parallel  [q_lora -> n_local_heads]     n_local_heads = n_heads / world
wo_a   column-parallel  [o_groups, o_lora_rank]       每卡一组内的 o_lora 切片
wo_b   row-parallel     [o_groups*o_lora_rank -> dim] + all_reduce
wkv    复制（只有一个 KV head，切了就要换 attention 布局）
MoE    专家按 id % world 分给各卡，shared expert 只在 rank 0，
       输出进 _ep_partials 由驱动层求和（ep.reduce_partials）
```

**这里的坑：`wo_a` 存的是 2 维 `[o_groups*o_lora_rank, dim]`，`o_groups` 是慢变维、
`o_lora_rank` 是组内快变维。** 所以切 o_lora 不能写成 `w[:, lo:hi]`（那是取每组的
前 k 个），必须 reshape 成 `[groups, per, rest]` 再切。`tp.shard_grouped` 就是干这个
的，两个 attention 类都用它。之前 `_shard_for_tp` 用一个 `w.dim() == 3` 的守卫来决定
是否切 `wo_a`，而存下来的权重永远是 2 维 —— 于是 `wo_a` 从来没被切过、`wo_b` 却被
切了，单卡测不出来，`tp>1` 直接形状不匹配。

## 多卡：一个已修、一个未完成

**已修**：`wo_a`/`wo_b` 的切分（见上）。

**未完成**：EP 的驱动层。`moe.py` 的归约逻辑是完整的 ——

```python
_bucket = getattr(self, "_ep_partials", None)
if _world > 1 and _bucket is not None:
    _bucket.append(y)
if _bucket:
    y = reduce_partials(list(_bucket), ...)
```

但 `_ep_partials` 在整个仓库里**没有任何地方被创建**。注释说它「is filled by the caller
that drives the per-rank passes」—— 那个 caller 不存在。

后果：`ep_world > 1` 时 `_bucket` 恒为 None，归约被跳过，每层只返回**本 rank 拥有的
那部分专家**的结果（`active` 已经被 `i % world == rank` 过滤），其余 rank 的专家被
静默丢弃。单卡 `ep_world == 1` 不受影响，所以本机永远测不出来。

**现在默认关闭，`TI_EP=1` 才打开，打开时大声警告。** 多卡时每层仍然算全部六个专家
（权重在每张卡上重复），结果正确，只是多占显存 —— 这比静默丢掉一半专家好。

为什么不是一个驱动循环就能补上：`ExpertStore.fetch_async(layer, experts)` **没有
device 参数**，专家的落点永远是层所在的那张卡。所以「让每张卡各算一部分」在当前架构
里没有落脚点 —— 要么给 fetch 加设备参数、把层的执行拆到多卡（层内并行），要么换一种
切分。这是架构改动，不是补丁。

原本打算的补法是：在一个进程内对每个 rank 各跑一遍该层的 `ffn.forward`（每次把
`ffn.ep_rank` 设成对应值、把激活送到那张卡、把该 rank 的专家结果收集回来），最后
`reduce_partials` 求和。涉及的跨卡激活复制和专家存储寻址目前也没有现成路径，所以这
是一个待实现的功能，不是一行修补。

**在那之前 `ep_world > 1` 不可用。** 多卡部署要么先用 TP-only（`tp.py` 那条路，已修
切分），要么先把这段补完。
## cutlass 的 sm120 blockwise 示例在本机跑不通（记下来，别再试）

`examples/87a_blackwell_geforce_fp8_bf16_gemm_blockwise` 用 `-arch=sm_120a` 能编过，
但**任何尺寸**跑起来都是 `exit=0xC0000409`。compute-sanitizer 报的是
`Misaligned shared or local address`。

排除项：

* 不是 smem 超限。默认 tile 128x128x128 x3 stage 的 A+B 是 96 KB，optin 上限 99 KB，
  看起来溢出，但把 tile 的 K 改成 64（A+B 降到 48 KB）重编后**一样崩**。
* 不是尺寸边界。M/N/K 从 128 到 1024、任意组合都崩；M=64 时只是 `can_implement`
  拒绝（程序正常退出）。

它用的是 `SM120_16x8x32_TN` + `SM90_TMA_LOAD` + `SM90_U32x2_STSM_N`，也就是
**TMA 路径**。而我们自己手写的 `fp8_gemm_mma`（纯 `mma.sync`，无 TMA）在这张卡上
kcheck 89/89 正常。所以更像是 TMA/stmatrix 在这张 Laptop sm120 上的问题，不是
cutlass 本身在 sm120 上不能用 —— H100/B300 的 TMA 是正常工作的。

**本机无法验证 UMMA/TMA 路径。** 手写 `mma.sync` 的 17.8 TFLOPS 大概就是这条指令
在这个形状下的上限（更快的 tcgen05/UMMA 只能走 cutlass 那条路）。所以 prefill 的
算力提升必须在专业卡上验。

另外：这个示例**不要在唯一一张卡上反复跑**，更不要叠 compute-sanitizer —— 它会重放
出错的 kernel，我这么干把整机搞到必须重启。

## 跨硬件的两处分歧（未修复，需要知道）

1. `expert_slots == 0`（8 GB 这类卡）decode 走 fused w4a8；有槽位的卡走 Marlin w4a16。
   把 native w4a8 提到 Marlin 前面之后 GEMV 本身统一了，但**累加方式仍不同**
   （`moe_gemv.acc` 的 kernel 内累加 vs `y[rows] +=` 的 scatter），所以同一份权重在两种
   卡上仍可能给出不同 token。
2. 同理，`TI_SLOTS=6` 强制启用槽位后输出会从 `[223,19,17,20]` 变成
   `[14,342,4571,1017]`。这是既存行为（改动前后一致复现过）。
