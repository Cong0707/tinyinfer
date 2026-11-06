# DeepSeek-V4.1-Flash-1017 在一张 8GB 卡上的实测

`E:/models/DeepSeek-V4.1-Flash-1017`，475.3 GB / 48 个 shard / 96085 个 tensor。
硬件：RTX 5060 Laptop（6.9 GB 可用显存）+ 47.2 GB 内存，SSD 顺序读能力 2.47 GB/s。

## 结果

> 这一组是某一轮的实测记录，**不是当前状态**。当前是 790 ms/token = 1.26 tok/s，
> 磁盘 3.0 GB/s，缓存命中率 68-93%（取决于负载），见 `docs/decode_bounds.md`。
> 下面这组数字保留是因为它记录了当时磁盘只有 1.45 GB/s 条件下的表现。

| 指标 | 值 |
|---|---:|
| build | 15.59 s |
| 首次前向 | 30.97 s |
| 热 prefill | 7.92 s |
| 稳态 decode | 0.935 - 1.190 s/token |
| **decode 速度** | **0.968 tok/s**（当时） |
| 磁盘吞吐 | 1.45 GB/s |
| expert cache 命中率 | 14.8% |

规划：10 层常驻 + 30 层流式，`cache=13.2GB`，`chunk=8`，`embed=cpu head=gpu`，
`layer=164MB expert=17.9MB`。MTP 被关掉（draft 阶段放不下，见 `resources.py` 的说明）。

## 之前为什么跑不起来：engram 表被算成了层

`model_profile()` 从 safetensors 头里统计每层的字节数，用来决定几层能常驻、
流式缓冲区要多大。它按 `layers.<int>` 解析层号，把不是 expert 的 tensor 都算进该层。

DeepSeek-V4.1-Flash 有两个 engram 层（layer 1 和 layer 14），每层带一张
**91.5 GB 的 `engram.embed.weight`** 加 2.9 GB 的 `embed.scale`。
这两张表是**按行从磁盘读的，从来不映射进显存**（`resources.py` 里原本就有注释
说明这一点，engram_bytes 也只统计了常驻的 wkv 投影）。但 per-layer 循环没有排除它们：

```
layer  14:   94.738 GB
layer   1:   94.720 GB
其余 38 层:  0.16-0.17 GB
```

`max_layer_bytes` 因此变成 94.7 GB，而它决定流式环形缓冲区的大小：

```python
fixed_b = embed + head + buffers * max_layer_b + ws0 + runtime_b
fits = fixed_b <= V          # 2 x 94.7GB >> 6.9GB
```

于是规划器判定「流式骨架需要 192.4 GB」并拒绝启动——一个 8GB 的卡当然装不下
200 GB 的骨架，但这个数字本身是假的。

修复是在 per-layer 循环里跳过 `engram` 段：

```python
if "engram" in parts:
    continue   # row-read from disk, never mapped into a layer buffer
```

修复后 `layer_bytes` 4.888 → **0.160 GB**，`max_layer_bytes` 94.738 → **0.173 GB**，
规划器随即给出上面的 10+30 方案。

## 当前瓶颈

- **磁盘只有 1.26 GB/s，SSD 能跑 2.47 GB/s**。日志里那条
  `slab unavailable (ok); falling back to per-tensor pinned` 说明平坦的 pinned
  预取 slab 没建起来，退回成了每 tensor 一次 pinned 拷贝。这是下一个该动的地方。
- **expert cache 命中率 10.1%**。256 个专家每 token 只用 8 个，13.2 GB 的 cache
  能装约 735 个 expert blob（17.9 MB 一个），理论上接近全覆盖；实际命中率这么低
  说明缓存替换策略或 key 有问题。


## 调参：ring 深度

把专家换成 Marlin 之后（专家 4.8 倍快），层的 H2D 在一步里占比变大，流水线的深度
重新变得重要。同机同模型实测：

| TI_RING | decode | 速度 |
|---:|---|---:|
| 2（旧默认） | 1.038 - 1.268 s | 0.867 tok/s |
| **3（新默认）** | **0.935 - 1.190 s** | **0.968 tok/s** |
| 4 | 1.029 - 1.400 s | 0.864 tok/s |

每多一个 ring 槽要占这张卡 463MB，所以 3 是测出来的最优点、不是能开多大开多大。

## 为什么 slab 只覆盖 5/30 层、expert cache 只命中 16%

两者都不是 bug，是同一笔 RAM 账：

```
[slabdbg] n=30 slab_total=6.51GB cap=13.21GB hb_total=47.2GB room=5.8GB first=166MB
[flatdbg] pool total=26.29GB used=25.17GB free=1.12GB
[tinyinfer] partial slab 5/30 layers (ok)
```

streamed 的 30 层权重本身就在 host pool 里（25.17 GB / 26.29 GB），pinned slab 只能
从剩下的 1.12 GB 里切，所以 30 层里只有 5 层拿到平坦块走单次 H2D，其余 25 层仍是
每个 tensor 一次 pinned 拷贝。**加大 `TI_RAM_FRACTION` 没用**（0.85 试过：pool 缩到
26.14GB、free 0.97GB、slab 反而只剩 4 层，速度 1.001 vs 0.986，噪声范围内）。

expert cache 那边，RAM 缓存按 `(layer, expert)` 存 key：DeepSeek 每层每专家都是独立
blob，40 层 × 256 = **10240 个**，每个 17.9 MB。实测 RAM 缓存装到 913 个条目（约 8.9%），
所以命中率停在 16.5% 左右是算术结果，不是替换策略的问题（`bench` 之前报
`entries=0` 是统计读错了属性名，已修）。

## 剩下的优化空间

1. **磁盘 1.47-1.88 GB/s，SSD 能力 2.47 GB/s**。差额在 25 层走 per-tensor pinned
   拷贝上。要拿回来，得让 streamed 权重和 slab 不再各占一份：现在 `pool` 存源、
   `slab` 存副本，同一个 tensor 在 RAM 里有两份。
2. **expert cache 想提高命中率，只能靠访问局部性**（同一层的专家在相邻 token 上
   重复的概率），而不是更大的 LRU。

## 零拷贝 slab：试过，卡在地址空间上

诊断：slab 只覆盖 5/30 层，是因为 30 层里 25 层走 per-tensor pinned 小拷贝，而
per-tensor 路径打不满 DMA（磁盘 1.4GB/s，SSD 能力 2.47）。追下去发现同一个 tensor
在 RAM 里存了两份：`weights.py.get()` 已经把权重 `readinto` 进 pool 的 slice，
`_flat_setup` 又 `copy_` 一份进 slab 块，5GB 的副本把 slab 的空间吃光了。

想法是让 slab 直接引用 layer 在 pool 里的那段字节。加了 `HostPool.slice_at` +
`weights.get()` 记录 `_tiny_pool_mv` + 按 `data_ptr()` 反查实际偏移，结果：

```
[zc] L10 base=8669626368 n=30 span=5305431031808 data=173987288
```

`span` 是 5.3 **TB**，不是 174MB。原因是 pool 来自 `cudaHostAlloc`，而
`torch.frombuffer` 对同一段内存报告的 `data_ptr` 和 `ctypes.addressof` 不在同一个
地址空间（实测：普通 ctypes array 两者相等，pinned 的不等）。偏移算不出来，连续性
判断全部失败。

**正确解法**（还没做）：不要事后反查地址，加载时就按 flat 布局落位 —— 每层的 tensor
用 256 对齐顺序切进同一个连续槽位，对齐规则和 `lay[]` 完全一致。这需要把
`Block.__init__` 的 `store.get()` 顺序和 `_weight_paths()` 的顺序统一，或者让
`get()` 接受目标偏移。代码留在 `TI_ZC=1` 开关后面，默认关闭。

## MoE 分项计时（TI_MOE_PROF=1，现在 bench 会打印）

跨一次 build + first forward + 3 个 decode token 的累计：

| 项 | 累计 | 含义 |
|---|---:|---|
| fetch | 12152 ms | 专家从磁盘/RAM 取回 |
| serve | 6484 ms | 专家的 GEMV 计算 |
| **bincount** | **1275 ms** | **`indices.reshape(-1).tolist()` 的 host 同步** |
| shared | 627 ms | shared expert（fp8） |
| gate | 406 ms | router（已换成 vLLM 的 topk_softplus_sqrt） |

**`bincount` 是纯同步开销**：逐专家 GEMV 必须先在 host 上拿到路由到的专家 id 才能
调度，而 `tolist()` 会阻塞到 GPU 算完。每层次约 8ms，40 层就是每步 ~320ms。

这条开销在 grouped GEMM 路径上完全消失：`moe_align_block_size` 在卡上算出每个专家的
token 区间，`cutlass_mxfp4_moe_mm` 按区间一次算完，host 全程不参与。按 0.96s/step 减
0.32s 估算，decode 会到 **~0.64s/step = 1.56 tok/s**。
