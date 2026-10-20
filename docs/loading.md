# tinyinfer 分层加载、权重选择与专家预取设计

> 状态：设计定稿 v1（2026-10-21），配套可运行骨架 src/tinyinfer/loading.py。
> 目标硬件：RTX 5060 Laptop 8 GB VRAM（sm_120）+ 47.2 GB RAM + NVMe。
> 目标模型：DeepSeek-V4.1-Flash-1017，HF safetensors 510.286 GB，直接加载、不转换。
> 覆盖优化层级：O4 基础分层 → O5 投机/预判 → O6 CPU 辅助决策 → O7 IO/加载协同 → O8 跨 agent 协同 → O9 异构流水线。
> 本文所有"实测"都标注测量条件；未标注者为解析计算或文献先验。

---

## 0. 结论摘要（先看这一页）

1. **每 token 冷读不是 4.25 GB，而是 4.512 GB（4.20 GiB）**。偏差来自专家 scale：每专家 = 3 份 fp4 权重 17,694,720 B + 3 份 ue8m0 scale 1,105,920 B = **18,800,640 B**；6 专家 × 40 层 = 240 块 = 4,512,153,600 B。旧估算 17.7 MB/专家漏掉了 scale（5.9%）。
2. **单个专家在文件里是两段**（不是 6 段）：3 个 weight 相邻（17.7 MB），3 个 scale 相邻（1.1 MB）。因此每 token 是 **480 次 pread**，不是 1440 次；NVMe 队列压力比想象小。
3. **本机实测 NVMe**：8 GB 新写文件、6 线程随机 18.8 MB 读 **8.80 GB/s**（p50 12.3 ms）；顺序 4 线程 7.53 GB/s；真实分片（下载中、部分缓存）直接读 3.5–5.0 GB/s。规划值取 **4.5 GB/s 保守 / 8.8 GB/s 乐观**。**PCIe H2D 实测：pinned 26.1 GB/s，pageable 8.96 GB/s**，因此专家上卡必须走 pinned staging。
4. **冷解码时间预算**（no cache）：NVMe 513 ms + PCIe 与固定开销 ≈ 110 ms → **~1.55 tok/s**（8.8 GB/s）；4.5 GB/s 时 ~0.88 tok/s。
5. **缓存是主杠杆，不是预取**。20 GB ARC 专家缓存在合成分布（Zipf 0.9 + topic + 25% temporal）下命中 **24.7%**，Zipf 1.2 下 **32.1%**；对应解码 **2.0–2.3 tok/s**。预取只能再拿 **+3%～+16%**，且必须过 gate，否则负收益（flash-moe 的 -18% 结论仍然成立）。
6. **预填充是最大惊喜**：512-token chunk 会触及几乎全部 384 专家/层 → 288.7 GB，33 s，**15.6 tok/s**；但若按"逐层预填充（layer-major）"把整个 prompt 一次喂完（激活放 RAM，专家每层只读一遍），8K prompt 也只要 ~33 s + 计算 → **~240 tok/s 预填充**。这是 O9 的核心设计。
7. **推荐默认配置（Lead 已拍板）**：MTP 关、head 放 CPU（实测 bf16 matvec 39–41 ms）、注意力 fp8 常驻 auto（29–34 层，可配全驻/部分驻）、24 个专家槽（451 MB）、RAM 专家缓存 20 GB 起步（可配）、ARC + 热度 pin、O_DIRECT 专家流、预取 lead=1 层 / k=2 / gate 开、活跃会话 2（上限 4）。fp8/int8 head 备选评估见 5.4.1：8 GB 卡上不划算，保持 CPU。
8. **真实 router 分布：layer 0 已完成，全 40 层待权重**。layer 0 真实预检（159 token）给出 232/384 专家、top-24 41.9%、单层 LRU(27) 命中 34.4%，不低于合成 24.7%，可把 24.7% 当保守下限。完整 E-L3/E-L4 需要全部 40 层分片 + tokenizer（当前下载 baseline 11.13/510.3 GB，约 11 h）；只下载 40 层 gate 权重（~160 MB）不够，因为路由输入是完整前向 hidden。脚本已就绪（scripts/prototypes/loading_research/）。

---

## 1. 事实与实测基线

### 1.1 模型布局（解析 48 个 safetensors header，非估算）

| 类别 | 字节 | 说明 |
|---|---:|---|
| routed experts 权重（fp4 packed） | 17,694,720 B/专家 | w1 2304×2560 I8，w2 5120×1152 I8，w3 2304×2560 I8 |
| routed experts scale（F8_E8M0） | 1,105,920 B/专家 | w1/w3 2304×160，w2 5120×72 |
| **单专家合计** | **18,800,640 B** | 17.93 MiB |
| 单层专家 | 7,219,445,760 B | 384 × 18.80 MB |
| 单层非专家 | 170,055,128 B | attention 162 MB + shared expert 35.4 MB + HC 3.9 MB + gate 3.9 MB |
| 40 层专家 | 288,777,830,400 B | 288.78 GB |
| 40 层非专家 | 6,802,205,120 B | 6.80 GB |
| embedding + head | 1,323,827,200 B ×2 | [129280, 5120] bf16 |
| Engram 表 1 张 | 101,377,628,352 B | 98.306 GB fp8 权重 + 3.072 GB ue8m0 scale（[rows, 8]） |
| Engram 表 2 张 | 202,755,256,704 B | 2 × 101.378 GB |
| Engram 投影（wkv+q/k） | 314,880,000 B | 2 层，wkv 25600×6144 fp8 |
| MTP/DSpark 3 层 | ~8.0 GB | 由总量 510.286 − 其余反推，~2.67 GB/层 |
| vision | 0.971 GB | 可禁用 |
| **总计** | **510,286,023,000 B** | 510.286 GB |

关键结论：
- **专家不是"每层一个 7.22 GB 大 blob"可顺序读**：文件内张量按字典序（0,1,10,100,…）排列，scale 全表连续、weight 全表连续。单专家的 3 个 scale 连续、3 个 weight 连续，但两段相隔很远。读取器必须按 (shard, offset) 聚合，不能假设专家整体连续。
- **fp4 原样搬运**：权重在 NVMe/RAM/PCIe 上永远是 packed int8（0.5 B/参数），反量化只发生在消费端（GPU）。任何"先反量化成 bf16 再缓存"的方案（当前 tinyinfer.moe.Expert 是 70.8 MB/专家 bf16）都会让 RAM 需求 ×3.8，必须替换。

### 1.2 本机实测

| 指标 | 实测 | 条件 |
|---|---:|---|
| GPU | 8151 MiB，sm_120，PCIe 5.0 ×8 | nvidia-smi |
| CUDA 分配前已用 VRAM | 1.7 GB | 桌面 + 浏览器（本次测量时） |
| RAM 可用 | 19.8–29.7 GB 波动 | 下载器/浏览器占 page cache 时偏低，非稳态；预算必须启动时实测 |
| PCIe H2D pinned | 26.11 GB/s | torch pinned 256 MB ×4 |
| PCIe H2D pageable | 8.96 GB/s | 同上 |
| NVMe 顺序读 4T | 7.53 GB/s | 8 GB 新写文件、NO_BUFFERING |
| NVMe 随机 18.8 MB ×40，6T | 8.80 GB/s，p50 12.34 ms，p95 18.35 ms | 同上（SLC 热态，属上界） |
| 真实分片读取（部分文件、page cache 热）buffered 6 专家 | 5.0 GB/s | model-00003 早期部分 |
| 真实分片读取 direct 6 专家 | 3.47 GB/s | 同上，O_DIRECT 等价路径 |
| 真实分片（刚合并、下载器同时写盘）buffered / direct | 1.25 / 1.79 GB/s | 6 专家 112.8 MB，4 次取最好 |
| bf16 GEMV 2304×5120（11.8 MB） | 1.46 ms/次 | GPU 非瓶颈；720 次/token = ~1.05 s 是 kernel launch 开销，不是算力 |

**规划值**：NVMe 4.5 GB/s（规划基准），8.8 GB/s（burst 上界），2.5–3 GB/s（最坏：下载/合并/杀毒同时写盘）。PCIe 25 GB/s；RAM→CPU 50 GB/s 级（按 DDR5-5600 估算，未实测）。所有 tok/s 预算按 4.5 和 8.8 两档给出。

### 1.3 每 token I/O 预算

| 项 | 冷缓存 | 说明 |
|---|---:|---|
| 专家数据 | 4.512 GB | 240 × 18.80 MB |
| 专家读取次数 | 480 pread | 每专家 2 段；按 offset 排序后 4–8 线程并行 |
| Engram 行 | 12,288 B（2 层 × 24 行 × 256 B） | 随机行；4 KB 页粒度下 ≤196,608 B/token |
| 非专家流（未常驻的注意力层） | 0–1.36 GB | 常驻 29 层时剩 11 层 × 170 MB = 1.87 GB；34 层时 6 层 = 1.02 GB |
| KV 窗口 | 5.24 MB/会话 | 40 层 × 128 × 512 × 2 B（仅在 VRAM） |
| head | 1.32 GB | CPU bf16 matvec 实测 39–41 ms；fp8/int8 备选见 5.4.1 |

时间模型（解码，单会话）：

    t_token ≈ bytes_expert_miss / B_nvme + 0.35 × bytes_pcie / B_pcie + fixed
    fixed ≈ 60 ms（采样/head/框架/同步）
    bytes_expert_miss = 4.512 GB × (1 − hit_rate)

预算表（B_nvme = 8.8 GB/s，PCIe 25 GB/s，注意力常驻 34 层 → 每 token 只流 1.02 GB）：

| 命中率 | 读盘 | NVMe | PCIe | 预估 tok/s |
|---:|---:|---:|---:|---:|
| 0% | 4.51 GB | 513 ms | 213 ms | 1.55 |
| 20% | 3.61 GB | 410 ms | 178 ms | 1.88 |
| 27% | 3.29 GB | 374 ms | 166 ms | 2.03 |
| 35% | 2.93 GB | 333 ms | 152 ms | 2.24 |
| 50% | 2.26 GB | 256 ms | 126 ms | 2.77 |

B_nvme = 4.5 GB/s 时：0.88 / 1.18 / 1.31 tok/s（0 / 27% / 35%）；B_nvme = 3 GB/s 时再打 0.67 折。这就是 8 GB 机器的现实区间：**0.6–2.8 tok/s 解码**（稳态 1–2 tok/s 最可能）。

---

## 2. 三级内存架构

~~~
                       ┌──────────────────────────────────────────────┐
                       │ NVMe (E:, 2.9 TB free)                       │
                       │  510.3 GB HF safetensors，只读，不可整层 mmap │
                       └───────────────┬──────────────────────────────┘
                          O_DIRECT/NO_BUFFERING 读专家 blob（2 段/专家）
                                       │ 4–8 线程，QD≈8
                       ┌───────────────▼──────────────────────────────┐
                       │ RAM (47.2 GB)                                │
                       │  ├─ 非专家常驻 9.76 GB（40 层 fp8 + embed/head）│
                       │  ├─ 专家 RAM cache 20–22 GB（ARC/S3-FIFO+pin） │
                       │  ├─ KV 长上下文 0.25–1 GB（压缩 KV，按需上卡）  │
                       │  ├─ pinned staging 0.8 GB（H2D 双缓冲）        │
                       │  └─ Engram 行缓存 0–0.25 GB（可选）            │
                       └───────────────┬──────────────────────────────┘
                          pinned H2D 26 GB/s，24 槽环形
                       ┌───────────────▼──────────────────────────────┐
                       │ VRAM (8 GB, ~6.6 GB 可用)                    │
                       │  ├─ 注意力/非专家 fp8 常驻 29–34 层 = 4.9–5.8 GB│
                       │  ├─ 专家槽 24 × 18.8 MB = 451 MB（暂存，不落盘）│
                       │  ├─ 激活 + KV 窗口 + 临时 = ~0.42 GB           │
                       │  └─ head：默认 CPU 算；可选 fp8 0.66 GB         │
                       └──────────────────────────────────────────────┘
~~~

设计规则：
1. **VRAM 只放"每 token 必用且复用率高"的东西**：注意力权重（常驻越多，PCIe 流越少）、专家槽（暂存）、激活/KV。
2. **RAM 放"每 token 必用但 VRAM 放不下"的东西**：非专家权重、专家 cache、长上下文 KV、staging。
3. **NVMe 只承担专家和 Engram 表**：这两者合计 491.5 GB，无法常驻；非专家永远不从 NVMe 读（17.7 GB 全驻 RAM）。
4. **专家槽是暂存不是缓存**：槽被换出时不丢数据（RAM cache 还有），这样 VRAM 占用可解释、可上限。
5. **反量化在消费端**：NVMe/RAM/PCIe 上永远是 18.8 MB 的原始 fp4 blob；GPU 拿到后直接喂 fp4 反量化 kernel（tilelang/torch），不生成 bf16 副本常驻。

---

## 3. 预算模型（8 GB VRAM / 47.2 GB RAM）

### 3.1 VRAM 预算

| 项 | 大小 | 备注 |
|---|---:|---|
| 桌面/WDDM 预留 | 0.9 GB | 本次实测桌面占 1.7 GB，需按实际调整；公式自适应 |
| CUDA/runtime 预留 | 0.35 GB | context、allocator、NCCL 无 |
| 注意力/非专家 fp8 常驻 | 34 × 170.06 MB = **5.78 GB** | 每层 170 MB；常驻层数 = floor(可用/170 MB) |
| 未常驻注意力层（按 token 流） | 6 × 170 MB = 1.02 GB/token | H2D 26 GB/s → 39 ms，可被 NVMe 掩盖 |
| 专家槽 | 24 × 18.80 MB = **451 MB** | 当前层 6 + 双缓冲 6 + 预取 12 |
| 激活 + KV 窗口 | 0.42 GB | prefill 按 512 token 块；KV 窗口 5.24 MB/会话 |
| **合计** | **6.65 GB** | 余量 1.35 GB；桌面占 1.7 GB 时常驻层数自动降到 29 |

自适应公式（TierBudget.plan 已实现）：

    usable_vram = vram_gb − desktop_reserve − runtime_reserve
    attn_layers = min(40, floor((usable_vram − activations − slots×18.8MB − KV) / 170MB))
    streamed_layers = 40 − attn_layers
    streamed_bytes_per_token = streamed_layers × 170MB

开关：--vram-reserve-mib（默认 900+350）、--attn-resident N（默认 auto）、--expert-slots N（默认 24）、--head cpu|fp8|bf16。

### 3.2 RAM 预算（三种情景）

| 项 | A：无 MTP（推荐默认） | B：MTP 开 | C：RAM 紧张（其他程序占 17 GB） |
|---|---:|---:|---:|
| OS + 应用预留 | 6.0 | 6.0 | 6.0 |
| 非专家常驻 | 9.76 | 17.78 | 9.76 |
| 其中：40 层注意力/shared/HC | 6.80 | 6.80 | 6.80 |
| 其中：embed + head | 2.65 | 2.65 | 2.65 |
| 其中：Engram 投影 | 0.31 | 0.31 | 0.31 |
| 其中：MTP 3 层 | — | 8.02 | — |
| KV（长上下文） | 0.25 | 0.25 | 0.25 |
| pinned staging | 0.8 | 0.8 | 0.8 |
| headroom | 2.0 | 2.0 | 2.0 |
| **专家缓存** | **22.2**（41 GB 可用时） | **14.2** | **1.2**（20 GB 可用时） |
| 专家条目（18.8 MB/条） | 1,180（7.7%） | 756（4.9%） | 63（0.4%） |

说明：
- "可用 RAM"必须**启动时实测**（psutil.virtual_memory().available），不能写死 47.2。本次测量时下载器把 page cache 占满，available 只有 19.85 GB；重启后通常 40–42 GB。TierBudget.plan(ram_available_gb=...) 支持。
- **Windows page-lock 上限约为物理 RAM 的一半（WDDM 共享预算）**（Strata DETAILS 的实测经验）。非专家 9.76 GB + staging 0.8 GB + KV pinned 0.25 GB = 10.8 GB < 23.6 GB，安全。专家 cache 用普通（非 pinned）内存，靠页表换入换出；不要 pin 20 GB 专家，会触发 pinned 耗尽并挤掉 OS。
- MTP 开 = 用 8 GB 专家缓存换投机解码。带宽受限时 MTP 的净收益存疑（flash-moe 实测 break-even），**默认关闭**，等 O5 实测后再决定。

### 3.3 NVMe 预算

| 项 | 值 |
|---|---|
| 专家总量 | 288.78 GB |
| Engram 总量 | 202.76 GB |
| 非专家总量 | 17.78 GB（全驻 RAM，不读盘） |
| 解码每 token 冷读 | 4.512 GB |
| 预填充 512-token chunk | 288.7 GB（触达全部专家一次） |
| 读放大（2 段/专家） | +1.1 MB scale / 18.8 MB = +5.9%（已计入 18.8 MB） |
| 4 KB 对齐浪费（O_DIRECT） | 每段 ≤8 KB → 480 × 8 KB = 3.8 MB/token = 0.08% |
| 规划带宽 | 4.5 GB/s 保守，8.8 GB/s 实测上界 |

---

## 4. 每 token 时间与吞吐模型

### 4.1 解码（单会话，8.8 GB/s，命中率 h）

    read_expert = 4.512 × (1−h) GB
    t_nvme = read_expert / 8.8
    t_pcie = (read_expert + 1.02) / 25            # 与 NVMe 并行，只算 ~35% 暴露
    t_token ≈ t_nvme + 0.35 × t_pcie + 0.060 s
    tok/s = 1 / t_token

### 4.2 预填充（逐 token 块 vs 逐层）

| 策略 | 1K prompt 专家 I/O | 时间 @8.8 | 预填充吞吐 | TTFT |
|---|---:|---:|---:|---:|
| 按 512-token 块顺序跑（每块重读全部专家） | 2 × 288.7 GB = 577 GB | 65.6 s | ~15.6 tok/s | 65 s |
| **逐层预填充：整 prompt 一次，专家每层读一遍** | **288.7 GB** | **32.8 s** | **~31 tok/s** | **33 s** |
| 逐层 + 8K prompt | 288.7 GB | 32.8 s | ~250 tok/s | ~33 s + 计算 |
| 逐层 + 32K prompt | 288.7 GB | 32.8 s | ~1000 tok/s | ~33 s + 计算/注意力 |

逐层预填充（layer-major prefill，O9 核心）：
1. 把整个 prompt 的 hidden（hc_mult × dim × 2 B = 40 KB/token）放 RAM；8K → 328 MB，32K → 1.3 GB。
2. for layer in 40: 先算该层 attention（按 token 块，KV 流式），得到 router 输入；按 expert 分组所有 token；对 384 专家分 wave（wave_size = (slots − 6)/2 = 9），每个专家只读一次，做 grouped GEMM；写回 hidden。
3. 结果：专家 I/O 与 prompt 长度基本无关（每层 7.22 GB 封顶），预填充从"每 token 4.5 GB"降到"每层 7.2 GB"。
4. 代价：hidden 全量驻 RAM（8K 0.33 GB）、attention 需要按块重算/缓存、实现复杂度高；需要 tinyinfer.model 改成 layer-major 调度。这是 O5/O9 的主要工程量。

### 4.3 多 agent 并发（随机路由，K=6，384 专家）

| 同时解码会话 | 每层并集专家 | 每 step 读盘 | 每 token 均摊 | 聚合 tok/s @8.8 |
|---:|---:|---:|---:|---:|
| 1 | 6.0 | 4.51 GB | 4.512 GB | 1.95 |
| 2 | 11.9 | 8.95 GB | 4.477 GB | 1.97 |
| 4 | 23.4 | 17.63 GB | 4.407 GB | 2.00 |
| 8 | 45.5 | 34.18 GB | 4.273 GB | 2.06 |

结论：**随机路由下 batching 几乎不减少 I/O**（并集近似相加）；它省的是 GPU 启动开销与注意力/head 的固定成本。真正的收益来自：
- 同任务/同上下文的 agent（路由相关）→ 并集显著小于相加；
- 预填充与解码混合调度：一个会话 prefill（I/O 密集、专家复用高）与另一个 decode 交错，可掩盖 GPU/PCIe 等待；
- prompt cache / 前缀复用（O8）。

因此并发策略：**2–4 个活跃会话为默认上限**，更多会话排队；每个会话独立 KV 与采样状态；专家 cache 全局共享。

---

## 5. 权重选择与专家缓存（O4/O7）

### 5.1 缓存内容与单元

- 缓存对象：**原始 18.8 MB fp4 blob**（w1/w2/w3 权重 + 3 个 scale 原样拼接）。不做 dtype 转换。
- 单元：18.8 MB；20 GB = 1,086 条；30 GB = 1,595 条；总量 15,360 条（40×384），覆盖率 7–10%。
- 命中定义：请求的 (layer, expert) 在 RAM 中；命中只省 NVMe，不省 PCIe（仍需上卡）。

### 5.2 替换策略：LRU / ARC / S3-FIFO + pin

| 策略 | 特点 | 本模型适配 |
|---|---|---|
| LRU | 实现简单；对扫描型访问差 | 基线 |
| **ARC** | T1/T2 + ghost B1/B2，自动平衡 recency/frequency；**模拟中最好** | **默认** |
| S3-FIFO | small 10% + main 90%，2-bit freq，工程简单、抗扫描 | 备选；模拟中略差（本分布） |
| 热度 pin | 按每层访问计数 pin 前 N 个专家，永不被 LRU 驱逐 | 与 ARC 叠加：top-20/层 = 800 条 = 15 GB |

模拟结果（128 token × 40 层 × 6 = 30,720 次专家请求；合成 Zipf 0.9 + topic + 25% temporal；脚本 scripts/prototypes/loading_research/sim_router.py）：

| 缓存 | LRU | ARC | S3-FIFO | ARC+pin20 |
|---:|---:|---:|---:|---:|
| 8 GB | 11.0% | **14.8%** | 8.1% | 13.6% |
| 16 GB | 19.1% | **22.5%** | 13.1% | 22.5% |
| 20 GB | 22.8% | **24.7%** | 15.3% | 24.9% |
| 24 GB | 24.9% | **26.8%** | 17.5% | 26.9% |
| 32 GB | 28.0% | **30.5%** | 21.0% | 30.4% |

Zipf 指数 1.2（更集中）+ temporal 15% 时：ARC 在 20 GB 命中 **32.1%**、24 GB **34.1%**、32 GB **37.4%**。结论：**命中率强烈依赖真实路由集中度**；必须用真实 trace 替换合成分布（第 10 节实验）。

pin 的注意点：pin 会减少可替换容量，模拟中收益仅 +0.1–0.3 个百分点；不要一次 pin 太多。建议 pin 预算 = 缓存 20%，且只 pin 观测 ≥3 个 token 的专家；每 128 token 重算一次，采用"晋升/降级"而不是整批替换。

### 5.3 热度自适应（Strata 的 --adapt-swaps 思路）

Strata 在 RAM↔VRAM 间按热度搬专家。本项目的对应设计：
- **RAM 内部**：ARC 自动做 recency/frequency；pin 集合每 K=128 token 用指数衰减计数重算（新专家晋升、旧专家降级）。
- **RAM↔VRAM**：不做主动搬迁（PCIe 26 GB/s、VRAM 只有 451 MB 槽），而是"槽轮换 + 预取"；VRAM 槽始终只放当前/下一层需要的专家。
- **可选 STRATA 式交换**：当专家在 VRAM 槽且将被换出时，若它不在 RAM cache 中，可先写回 RAM（避免二次读盘）。代价是 PCIe 18.8 MB 写回；收益仅当该专家未来会复用。默认关，作为 E-L6 实验项。

### 5.4 非专家权重（权重选择加载的另一半）

- 17.78 GB 非专家全驻 RAM（pinned 10.8 GB 以内），每 token 从 RAM 取。
- VRAM 常驻 29–34 层注意力（fp8 原样），剩余 6–11 层按 token 流：每层 170 MB，26 GB/s → 6.5 ms/层；11 层 = 72 ms/token，可与 NVMe 并行。
- **不要**把注意力反量化成 bf16 常驻 VRAM（340 MB/层 × 40 = 13.6 GB，不可能）。需要 bf16 计算时，在 kernel 内联反量化或每层临时 dequant 到 scratch（340 MB 峰值）。
- head：129280×5120 bf16 = 1.32 GB。默认 **CPU bf16 matvec**，实测 39–41 ms/次（16 线程；读 1.32 GB RAM，有效 33 GB/s），见 5.4.1。
- embedding：只做行 gather（10 KB/token），放 RAM。

#### 5.4.1 head 放哪里：实测与 fp8/int8 备选

实测条件：随机 bf16 [129280, 5120]（1.324 GB），torch 2.9 CPU 16 线程，RTX 5060 Laptop；与真实 head.weight 的 dtype/形状一致（header：BF16 [129280,5120]）。误差 ±2 ms。

| 方案 | 实测/估算 | 结论 |
|---|---:|---|
| **CPU bf16 matvec（默认）** | **39–41 ms**（8/16/32 线程 40.8/39.8/40.3） | 推荐：0 VRAM，速度可接受 |
| CPU fp32 matvec（当前 model.py 路径 .float()） | 80.9 ms + 2.65 GB 转换 | 不要；除非 logits 精度验证必须 |
| GPU bf16 常驻 VRAM | GEMV 3.6 ms，但占 1.32 GB → 挤掉 8 层注意力 ≈ +52 ms PCIe/token | 不划算 |
| GPU bf16 每 token pinned 上卡 | H2D 46.3 ms + GEMV 3.6 ms ≈ 50 ms | 不如 CPU |
| fp8 head 0.66 GB（per-row scale）常驻 RAM | H2D 28.2 ms + torch dequant+GEMV 29.4 ms ≈ 58 ms；若自定义 fused fp8 GEMV（~3 ms）≈ 31 ms | 备选 B：需 tilelang 内核；torch._scaled_mm 在 sm_120 不支持 rowwise scale（实测报错） |
| fp8 head 常驻 VRAM 0.66 GB | 挤掉 4 层注意力 ≈ +26 ms + GEMV ~3 ms ≈ 29 ms | 比 CPU 略快，但 0.66 GB VRAM + 量化质量风险；8 GB 卡不推荐 |
| int8 head 0.66 GB | 与 fp8 同尺寸：H2D 28.0 ms；torch 无 rowwise int8 GEMM | 备选 C：收益与 fp8 同量级，质量更差，暂不做 |

**拍板**：默认 CPU bf16；把 model.py 的 head 计算从 h.float()/head.float() 改为 bf16（否则 81 ms + 2.65 GB）。fp8/int8 只有在 (a) 自定义 fused GEMV 内核落地且 (b) 真实权重质量验证通过后才考虑；在 8 GB 卡上收益仅 5–10 ms，不值得抢先做。若未来换 12–16 GB 卡，可把 fp8 head 常驻 VRAM 作为省 CPU 的方案。

### 5.5 Engram 行读与缓存

- 每 token 每 Engram 层 24 行 × 256 B = 6 KB；2 层 = 12 KB。哈希 id 在**当前 token 已知时**就能算（NgramHashState 只依赖 token id），所以可以在 step 开始时预取，延迟可忽略。
- 但 24 个随机 256 B 读在 4 KB 页粒度下是 24 个页 = 96 KB/层，2 层 192 KB/token（相对 4.5 GB 是 0.004%）。**不需要专门大缓存**；建议 128–256 MB 行缓存 + OS page cache 兜底，命中率约 15%（随机行）。
- 真正要小心的是 Engram 表的**首次访问抖动**：随机行在 NVMe 上 ~100 μs 级，24 行并行可压到 ~200 μs；放 step 开始（第一个 token 的 embedding 之后）异步预取，避免在 layer 1/14 处串行等待。
- 建议开关：--engram-row-cache 256MB（默认）、--engram-prefetch on。

### 5.6 KV 分层（Strata KV streaming）

- 窗口注意力的 KV 只有 128 行 × 512 × 2 B = 131 KB/层 → 5.24 MB/会话，**常驻 VRAM**。
- 长上下文压缩 KV（compress_ratio 层）随上下文增长，放 **RAM**；VRAM 只保留当前 attention 读的尾部窗口（--kv-resident N tokens，Strata 默认 32K cells）。
- 估算（待实测）：压缩 KV ~1.3 KB/token → 128K 上下文 166 MB/会话，4 会话 664 MB；如果按 13.7 KB/token（Strata 的模型）则 128K = 1.75 GB/会话，必须限制并发或开 KV 量化。
- 开关：--kv-resident 32768（tokens 在 VRAM）、--kv-q8/--kv-q4（量化，质量待测）。
- tinyinfer 当前 attention.py 只实现窗口路径；压缩层的 KV 分层由 attention owner 接入，本模块预留 RAM 预算 0.25–1 GB。

---

## 6. 预取与分支预测（O5/O6）

### 6.1 先讲清楚数学：预取为什么经常亏

设单层 K=6 个专家、每专家 18.8 MB、B=8.8 GB/s → 每专家读 2.14 ms，单层读 12.8 ms；设该层 GPU 计算（attention+FFN）3 ms，且计算期间 NVMe 空闲。那么：

- 无预取：t = 3 + 12.8 = 15.8 ms/层。
- 预取 k 个专家（在上一层/本层计算期间读），命中 h 个：t = 3 + (6−h) × 2.14 + max(0, k×18.8MB − 3ms×8.8GB/s)/B。
- 免费窗口 = 3 ms × 8.8 GB/s = 26.4 MB = 1.4 个专家。
- k=2 时溢出 11 MB → +1.3 ms；h=1 时 t = 3 + 10.7 + 1.3 = 15.0（+5%）；h=1.4 时 14.7（+7.5%）。
- **预测错误会同时增加总字节**：k=2、precision 0.6 → 多读 1.5 GB/token（+33% 字节），但其中大部分落在空闲窗口内，净收益仍为正；若计算时间只有 1 ms（窗口 8.8 MB），k=2 溢出 29 MB → 预取变成 +3.3 ms，**负收益**。

因此预取的硬条件：
1. **GPU 计算窗口 ≥ 预取字节/B**（本模型 3 ms 窗口 → k≤1.4；k=2 略溢出）。窗口大小需用 nsys 实测每层 compute。
2. **precision 必须高**：模拟给出 k=2、recall=0.5 时的净加速：precision 0.4 → +2.8%，0.5 → +5.8%，0.6 → +8.9%，0.7 → +12.2%，0.8 → +15.7%（脚本 sim_router.py 输出）。
3. **必须 gate**：连续 16 层统计"预取命中率 <10%"就关，隔 256 层再探测。Strata 的 decode gate 就是这个思路，flash-moe 的 -18% 是没 gate 的代价。
4. **预取目标只放 RAM**（不占 VRAM 槽），槽只给当前层。

### 6.2 可用的预测器（按性价比排序）

| 预测器 | 输入 | 提前量 | 预期 precision@k | 成本 | 结论 |
|---|---|---|---:|---|---|
| **Temporal** | 上一 token 同层 top-6 | 1 token（可提前 40 层） | 命中 1–2 个/6；集合重叠 ~25% | 0 | 基线；单独用亏（flash-moe -18%），只作为 gate 参考 |
| **N-gram router** | 最近 4 个 token id → 历史路由表 | 1 token | 重复 agent 循环里 60–80%，一般文本 10–20% | 极低（CPU 查表） | O5 亮点：适合 agent/tool 重复调用；默认开，带 gate |
| **Lookahead 线性** | 当前层 hidden (5120) → 目标层 gate logits (384) | 1 层 | 未实测；预期 top-1 25–35%，precision@2 40–60% | 2M 参数/层（8 MB fp32），CPU GEMV 0.05 ms | 主力实验对象；用真实 trace 训练后才知道 |
| **Lookahead MLP** | 同上，2 层 hidden | 1–2 层 | 文献 31%（flash-moe，同层）；跨层更低 | 中 | 先不做，线性不够再上 |
| **Oracle** | 真实路由 | — | 100% | — | 只用于上限评估 |

CPU 决策点（O6）：
- **lead=1 层**：在 layer L 的 attention 输出（router 输入）拿到后，CPU 立刻用 lookahead 预测 L+1 的 6 个专家；此时 GPU 正在算 L 的 FFN（3–5 ms），NVMe 空闲，正好读 L+1 的 2 个高置信专家。**这是唯一真正能隐藏 I/O 的窗口**。
- **lead=2 层**：只有当每层计算 ≥6 ms（PyTorch fallback/长上下文）时才值得；否则预取字节超过窗口，负收益。
- **transformer 依赖决定了不可能提前更多**：layer L+1 的 router 输入依赖 layer L 的 FFN 输出，而 FFN 输出依赖专家本身。任何"提前 2 层以上"的预测都是对 hidden 的猜测，误差累积。
- CPU 还负责：采样（head 1.3 GFLOP，CPU bf16 39–41 ms，见 5.4.1）、n-gram 查表、PCIe/CPU 分流决策（6.4）、gate 统计。

### 6.3 预测器实现（loading.py 已提供）

- TemporalPredictor：上一 token 同层路由 + 在线命中率。
- NgramRouterPredictor：最近 n 个 token id 的哈希 → 每层路由表（LRU 限 20 万条），完全 CPU，无 hidden 依赖。
- LookaheadRouterPredictor：线性 W[384, 5120] + bias，fit() 用 ridge 回归，predict() 取 top-k + softmax 置信度；save/load npz。
- OraclePredictor / EnsemblePredictor：上限与融合。
- Prefetcher：lead_layers、max_experts_per_layer、min_confidence、free_compute_ms、bandwidth_gbs、gate 窗口；统计 issued/hits/precision/wasted_gb/gate_trips。**默认 gate=True，lead=1，k=2，min_confidence=0.5，free_compute_ms=3**。

### 6.4 O5 投机解码对 I/O 的影响

- DSpark/MTP 3 层（~8 GB，128 专家/层）：每轮 draft 需要额外专家 I/O ~3 专家 × 3 层 × 18.8 MB = 169 MB；verify k 个 draft token 会让目标模型每层并集从 6 增到 ~6+3(k−1)。
- 在带宽受限的解码里，**投机解码基本 break-even**（flash-moe 实测；每 token 的专家 I/O 无法摊薄）。只有当接受率高且 cache 命中率高（专家复用）时才可能正收益。
- 结论：**默认关闭 MTP**；O5 实验先做 n-gram draft（无额外模型权重，CPU 侧 0 I/O），再评估 MTP。预填充侧可以用 speculative prefill（对长 prompt 提前算多个 token 的 attention，但专家仍按层读一遍，收益有限）。

### 6.5 预取实验方案（可验证）

1. 采集真实路由（第 10 节 E-L3）。
2. 用 train_lookahead.py 拟合 W，报告 top-k precision/recall。
3. 用 sim_router.py 把真实 trace 代入，比较：
   - 无预取、temporal、n-gram、lookahead、oracle 的 precision 与净加速；
   - gate 开/关；
   - k=1,2,3；free_compute_ms=1,3,5。
4. 端到端 A/B：同一 prompt、同一 seed，对比 tokens/s 与输出 token 一致性（预取不得改变数值结果；只影响时序）。
5. 通过标准：**precision@2 ≥ 0.6，端到端解码 ≥ +5%，浪费字节 < 40%**。

---

## 7. 多 agent 并发与跨会话共享（O8）

### 7.1 设计

- **全局共享**：一个 ExpertCache（RAM）、一个 VramSlotCache、一个 ShardReader 线程池、一个热度统计表。会话只持有 KV、采样状态、路由历史。
- **调度**：SessionScheduler.plan(sessions, phase) 返回每层并集与 wave 划分；解码时把 2–4 个会话的 token 合成一个 micro-batch 过一次模型（每层并集专家只读一次）。
- **公平性**：按轮转（round-robin）组 batch；单会话连续 decode 上限 8 token，避免长会话饿死其他会话；p99 延迟 / p50 ≤ 3。
- **缓存配额**：全局 ARC；每个会话可 pin 自己的 top-N（N=8/层）作为"会话亲和"，总量不超过缓存的 20%；会话结束后释放。
- **prompt cache（O8）**：相同前缀的请求复用 KV 与已算 hidden；专家热度按 prompt 前缀聚类（同一 agent 的 system prompt + 工具 schema 固定 → 路由高度重复），这会让 n-gram/热度缓存命中率显著高于随机文本。
- **API**：OpenAI 兼容层只负责请求排队、会话 ID、流式返回；模型执行器无状态，所有缓存进程内共享。

### 7.2 内存分配（4 会话场景，无 MTP）

| 项 | 大小 |
|---|---:|
| 全局非专家常驻 | 9.76 GB |
| 全局专家 cache | 20.0 GB |
| 每会话 KV（8K 上下文） | ~10 MB（窗口 5.24 MB + 压缩 KV ~5 MB） |
| 4 会话 KV | 40 MB |
| staging + headroom | 2.8 GB |
| **合计** | **~32.6 GB**（余量给 OS/浏览器） |

注意：4 会话 batching 的每 step I/O 是 17.6 GB（表 4.3），单步耗时 2.0 s → 每会话 0.5 tok/s、聚合 2.0 tok/s。**并发不会提高单会话速度**；如果用户要交互体验，默认只跑 1 个活跃会话，其余排队。

---

## 8. 异构流水线（O9）：一层之内做什么

目标：让 NVMe、PCIe、GPU、CPU 尽量同时忙。单层时间线（解码）：

    时间 →
    GPU:   [attn_L 计算 1.5ms]──[FFN_L 计算 1.5ms]────────────[attn_{L+1}]…
    CPU:        [router 0.3ms][预测 L+1 0.1ms][gate/统计 0.05ms][采样/head 39ms(每 token 一次)]
    NVMe:                 [读 L+1 预测 2 专家 4.3ms][读 L+1 缺失 4.8 专家 10.2ms]
    PCIe:                          [L 的 6 专家上卡 4.3ms]──…

要点：
1. **双缓冲**：两个专家槽组，A 组给当前层 FFN，B 组接收下一层预取/缺失读入；槽数 24 足够（6 当前 + 6 双缓冲 + 12 预取/流水）。
2. **读顺序**：按 layer 递增；同层内按文件 offset 排序以利用 NVMe 顺序性；480 次 pread/ token，8 线程，队列深度 ≈ 8。
3. **O_DIRECT/NO_BUFFERING**：专家流用 FILE_FLAG_NO_BUFFERING（Windows O_DIRECT 等价），避免 4.5 GB/token 的读把非专家常驻页和 OS 缓存冲掉；非专家用 buffered（17.7 GB 常驻 RAM 后不会再读盘）。实测 direct 3.47 GB/s vs buffered（热）5.0 GB/s；冷态下 direct 更稳。
4. **fp4 原样搬运**：H2D 只搬 18.8 MB/专家，不搬 70.8 MB bf16；GPU 侧用 fp4 反量化 + GEMM（tilelang 或 torch._scaled_mm）。这是 3.8× PCIe 带宽节省。
5. **PCIe/CPU 分流（pcie-frac，O6）**：missed expert 可以选择"上卡算"（18.8 MB / 26 GB/s = 0.72 ms + GPU）或"CPU 算"（读 RAM + fp4 反量化 + GEMV ≈ 0.7–1.4 ms/专家）。默认 pcie_frac=0.5，按实测调；GPU 空闲时优先上卡，NVMe 忙时 CPU 算以重叠。
6. **注意力与非专家流**：未常驻的 6–11 层注意力权重在 token 开始时预取到 VRAM（H2D 与 NVMe 并行），被 FFN I/O 掩盖。
7. **Engram 行**：token 开始时知道 ids → 异步读 48 行（12 KB），在到达 layer 1/14 之前完成。
8. **head/采样**：放最后，CPU 算或 GPU 流式；与下一 token 的 layer-0 输入准备重叠。

预期（8.8 GB/s，h=27%）：
- 解码：0.37 s NVMe + 0.06 s 暴露 PCIe + 0.06 s fixed ≈ **0.49 s/token ≈ 2.0 tok/s**（预取 +5–10% → 2.1–2.2）。
- 预填充（layer-major）：1K prompt ≈ 33 s（31 tok/s），8K ≈ 35–40 s（200+ tok/s）。
- 4 会话并发：聚合 ~2.0–2.2 tok/s，单会话 0.5 tok/s。

---

## 9. 接口设计（loading.py）

### 9.1 数据流

    ShardIndex ── 解析 index.json + safetensors header（lazy）
       │            region(name) / expert_blob(layer, expert) -> 2 段
       ▼
    ShardReader ── 每线程句柄；buffered 或 Windows NO_BUFFERING
       │            read_blob / read_many
       ▼
    ExpertCache ── RAM：ARC/LRU/S3-FIFO + pin + prefetch 线程池
       │            get / prefetch / pin_hot / hot_experts / stats
       ▼
    VramSlotCache ─ 24 个 uint8 设备槽；acquire/copy_in/release
       │
       ▼
    ExpertStore ── fetch(layer, ids) -> list[ExpertRecord]
       │            ExpertRecord.tensor("w1.weight"|"w1.scale"|...)
       ▼
    tinyinfer.moe：dequant_fp4_block(rec.tensor("w1.weight"), rec.tensor("w1.scale"))

    Prefetcher ── Predictor（Temporal / NGram / Lookahead / Oracle）
    SessionScheduler ── 多会话 union + prefill wave + dedupe 统计

### 9.2 接入 moe.py 的最小改动

当前 tinyinfer.moe.MoE._expert(i) 自己用 ShardStore.get 读 6 个张量，再 dequant 成 bf16 并缓存在 dict（70.8 MB/专家，无限增长）。替换为：

    # 初始化（每个 MoE 层共享同一个 ExpertStore）
    self._store = ExpertStore.from_store(store, cache_bytes=20<<30, vram_slots=24, direct=True)

    # forward 中
    with self._store.experts(self.layer_id, needed_experts) as recs:
        for rec in recs:
            w1 = rec.tensor("w1.weight")     # uint8 [2304, 2560]，CPU 或 CUDA
            s1 = rec.tensor("w1.scale")      # float8_e8m0fnu [2304, 160]
            e = Expert(dequant_fp4_block(w1, s1), ...)

注意：
- 不在 MoE 内部缓存反量化结果；反量化每 token 做一次（当前实现也是每 token 反量化，但把 bf16 常驻了）。
- ExpertRecord 的生命周期由 with 块管理；底层 RAM cache 与 VRAM 槽不因 release 而丢失。
- Lead 负责把 MoE.forward 的 expert 循环改成"先 fetch 全部 6 个，再并行 dequant/GEMM"，以及 layer-major prefill 的调度。

### 9.3 关键 API 一览

| 对象 | 方法 | 语义 |
|---|---|---|
| ShardIndex | region(name), expert_blob(l,e) | 文件偏移、两段 blob、张量短名映射 |
| ShardReader | read_blob, read_many, stats, close | 同步/并行字节读取 |
| ExpertCache | get, prefetch, contains, pin, unpin, pin_hot, hot_experts, stats | RAM 缓存 |
| VramSlotCache | acquire, copy_in, release, stats | VRAM 暂存环 |
| ExpertStore | fetch, experts, release, prefetch, pin_hot, stats | 门面 |
| Prefetcher | note_layer, note_step_start, wait, stats | 预测/预取/gate |
| SessionScheduler | plan(sessions, phase) -> StepPlan | 多会话并集/wave |
| TierBudget | plan(...), table(), io_scenarios() | 内存/IO 预算 |
| IoBudget | compute(...) | 命中率→tok/s |

### 9.4 接入 moe.py 最关键的 3 个 API（短清单）

1. **ExpertStore.from_store(...)**：`store = ExpertStore.from_store(store, cache_bytes=20<<30, policy="arc", vram_slots=24, direct=True, device="cuda")`。进程内只建一次，传给所有 MoE 层/会话；不要每层 new 一个（重复文件句柄+缓存+线程池，RAM 会翻倍）。
2. **store.experts(layer_id, ids) + rec.tensor(name)**：推荐 `with store.experts(layer_id, ids) as recs:`，先 fetch 全部 6 个，再并行 `dequant_fp4_block(rec.tensor("w1.weight"), rec.tensor("w1.scale"))`。视图零拷贝，uint8 packed fp4 + float8_e8m0fnu scale，直接喂现有 quant 函数；with 退出自动 release。
3. **store.prefetch(layer, ids) / store.pin_hot(per_layer=20) / Prefetcher.note_layer(...)**：预取与热度 pin 是可选加速；预测逻辑放在 Prefetcher，不要塞进 MoE 循环。集成顺序：先正确性（1+2），再 gate（3）。

**陷阱（按踩坑概率排序）**
- 不要把 rec.tensor(...) 的视图带到 with 之外使用：VRAM 槽会被复用（换 owner 后同一 slot 是脏数据）；RAM 条目也可能被驱逐。跨层/跨 token 要用时先复制到自有 buffer。
- dtype/形状别转错：weight 是 **uint8 packed fp4（每字节 2 个 nibble）**，不要 .float()、不要 .view(int8) 后按 fp4 解；scale 是 **float8_e8m0fnu [out, in/32]**。dequant_fp4_block 期望 (packed [out,in//2], scale [out,in//32])，别 reshape 成 (in,out)。
- 并发引用数不能超过 vram_slots：6 专家/层 + 双缓冲 6 + 预取 12 = 24；多会话时按 union 计算，否则抛 "all VRAM expert slots are referenced"。
- fetch 异常也要 release（务必用 with）；close() 前等所有 prefetch future 结束（当前 close() 内部 shutdown）。
- direct=True 只用于专家流；非专家权重仍用 buffered（17.7 GB 常驻 RAM）。ShardReader 每线程一个文件句柄，线程数默认 6。
- 集成前先跑 `python -m tinyinfer.loading selftest` 与 `budget`；cache/gb 为 0 或分片缺失时会显式抛异常。

---

## 10. 实验计划（按依赖顺序）

| 编号 | 实验 | 方法/脚本 | 指标 | 通过标准 |
|---|---|---|---|---|
| E-L1 | I/O 微基准 | scripts/prototypes/loading_research/bench_io.py；真实分片 | 2 段/专家、480 pread/token、GB/s、p50/p95 | ≥4.5 GB/s（保守）；单专家 2 段成立 |
| E-L2 | 缓存策略模拟 | sim_router.py（合成分布） | 命中率 vs 容量/策略 | ARC ≥ LRU；20 GB ≥ 22% |
| E-L3 | 真实 router trace | collect_router_trace.py（需全 40 分片） | 每层 top-6、gate logits、hidden | 30K+ 位置；分布统计归档 |
| E-L4 | 预测器训练 | train_lookahead.py | precision@k / recall@k（k=1,2,4） | precision@2 ≥ 0.6（否则只保留 n-gram/temporal） |
| E-L5 | 预取端到端 A/B | Prefetcher gate 开/关；同 prompt/seed | tok/s、浪费字节、输出一致性 | tok/s +≥5%，浪费 <40%，输出逐 token 一致 |
| E-L6 | 多会话调度 | SessionScheduler；2/4/8 会话 | 聚合 tok/s、单会话 p50/p99、dedupe | 聚合 ≥1.8× 单会话；p99/p50 ≤3 |
| E-L7 | 逐层预填充 | 改造 model.py 为 layer-major | TTFT、prefill tok/s、RAM 峰值 | 1K TTFT ≤40 s；prefill ≥20 tok/s |
| E-L8 | 长上下文 KV 分层 | --kv-resident 32K；128K 上下文 | RAM、tok/s 下降 | 128K 会话 RAM ≤1 GB；速度下降 ≤20% |
| E-L9 | 命中率真实化 | E-L3 trace 代入 sim_router | 命中率/预取收益 | 给出 20/24/32 GB 三档真实命中率 |

实验记录建议写入 docs/loading-experiments.md（由 Lead 决定负责人），本文件只保留设计与结论。

### E-L3 预检（2026-10-21，仅 layer 0，真实权重）

条件：model-00003（layer 0，7.39 GB）已完成；159-token 混合代码/文本 prompt；只跑 block0 的 attention+router（跳过 FFN），CPU 0.46 s；hook 真实 gate。

- 触达 232/384 专家；top-6 share 24.3%、top-24 41.9%、top-48 57.8%、top-96 77.5%；有效专家数 131。
- 相邻位置 top-6 交集均值 **0.85/6（14.2%）**，P(≥1)=47%。
- 单层 LRU（逐层配额 27 条 ≈ 20 GB/40）：真实命中 **34.4%**；cap 12/20/40/64 → 22.1/28.8/42.0/58.1%。
- 含义：layer 0 真实集中度不低于合成分布（20 GB 全局 ARC 24.7%），**24.7% 可视为保守下限**；但只是单层 prompt 内统计，decode 跨 token / 40 层仍需 E-L3 全量验证（需全部分片）。

### E-L3/E-L4 阻塞状态

- 下载日志：2026-10-21 00:24 重启后 baseline 11.13/510.3 GB，速率约 12 MB/s（≈11 h 到全量）；当前仅 layer 0 完整，layer 1–39、engram、MTP 未齐。
- 因此 E-L3 只能先做 layer 0 预检；E-L4 的 lookahead 需要多层 hidden→logits 对，暂不能训练。
- 权重齐后执行：collect_router_trace.py（128 token × 40 层，1–2 h）→ train_lookahead.py → sim_router.py --trace，替换本文 24.7%/32.1% 的合成数字。

---

## 11. 风险、失败模式与开关

| 风险 | 影响 | 缓解/开关 |
|---|---|---|
| Windows pinned 内存上限（~50% RAM） | 无法 pin 全部非专家+staging；驱动拒绝 | 只 pin 10.8 GB；--pin-budget；失败降级为普通内存 |
| O_DIRECT 对齐 | 偏移非 4 KB 对齐 → 每段多读 8 KB；实现复杂 | 已实现按 4 KB 对齐读 + memmove；480 段 ×8 KB = 0.08% 浪费 |
| mmap 整文件 | page fault 风暴（flash-moe 冷数据 -5×） | **禁止**；专家流只 pread |
| 预取错误浪费带宽 | 可能 -18%（flash-moe） | gate（16 层窗口，命中 <10% 关闭）；k≤2；只放 RAM |
| 计算窗口估错 | k=2 溢出 → 负收益 | 先用 nsys 实测每层 GPU 时间；--free-compute-ms 可调 |
| 缓存污染（预取挤掉热点） | 命中率下降 | 预取只读 RAM cache，不碰 VRAM；pin 保护热专家；gate |
| 多会话内存膨胀 | OOM/RAM 抖动 | 会话上限 4；每会话 KV 配额；全局 cache 上限 |
| 下载/其他程序占 RAM | available 从 41 GB 掉到 20 GB | 启动时实测 available，动态缩减 cache；最低 4 GB（212 条）仍可运行 |
| 桌面占 VRAM 1.7 GB | 常驻层数下降 | 自适应公式；--vram-reserve-mib |
| 专家文件读取与下载器竞争 | 带宽/队列被抢 | 下载期间不做性能结论；测量前暂停下载 |
| bf16 反量化常驻（当前 MoE 实现） | RAM 爆炸 70.8 MB/专家 | 集成时必须移除 dict 缓存；只保留 fp4 blob |
| MTP 打开 | +8 GB RAM、每轮 +169 MB I/O | 默认关；E-L5 后再评估 |
| 逐层预填充实现复杂 | 可能引入数值差异 | 以现有逐层 chunk 实现为 reference，对比 logits/hidden |

---

## 12. 给 Lead 的集成建议与待确认

1. **先接 ExpertStore，不要接 Prefetcher**：ExpertCache + ExpertStore 是正确性相关（缓存、生命周期、VRAM 槽），Prefetcher 是纯可选加速。建议集成顺序：ExpertStore → 预算/开关 → 多会话 scheduler → prefetcher gate → layer-major prefill。
2. **moe.py 的改动点**：删除 _expert_cache: dict[int, Expert]；_expert(i) 改为从 ExpertStore 取 rec；6 个专家先全部 fetch 再 dequant（并行）；每层结束 release。Gate.forward 返回值不变。
3. **数值一致性**：loading 层只改变"字节从哪里来"，不改变 dtype/布局；dequant 输入仍是原 HF 的 uint8/fp8 张量，因此输出应与现在逐层验证结果 bit-exact（除 GPU/CPU kernel 顺序差异）。
4. **真实 router 分布（进行中）**：需要全部 40 层分片 + tokenizer；只给 40 层 gate.weight（~160 MB）不够，因为路由输入是完整前向的 hidden。截至 2026-10-21 00:29 下载 baseline 11.13/510.3 GB、~12 MB/s（≈11 h），仅 layer 0 完整。已完成 layer-0 真实预检（见 §10）：232/384 专家、top-24 41.9%、单层 LRU(27) 命中 34.4%，不低于合成的 24.7%。权重齐后我按 E-L3/E-L4 跑全 40 层（1–2 h）替换合成数字。
5. **已拍板默认值（2026-10-21 Lead）**：
   - MTP/DSpark 默认关（省 8 GB 给专家缓存，作为可选档，用真实接受率再评估）；
   - head 默认 CPU bf16（实测 39–41 ms；fp8/int8 备选见 5.4.1，8 GB 卡不划算）；
   - 专家缓存 20 GB 起步（可配，CLI --expert-cache-gb）；非专家/attn 常驻策略做成配置（auto 29–34 层 / 可全驻 / 可部分驻）；
   - 多 agent 默认 2 活跃（上限 4）、全局 ExpertCache、会话 pin ≤20%；
   - 集成顺序：ExpertStore → 预算开关 → SessionScheduler → Prefetcher gate；layer-major prefill 由 Lead 改 model.py。
6. **接口冻结建议**：loading.py 的 ExpertStore/ExpertRecord/ShardReader 接口先冻结；Prefetcher/Scheduler 可在 E-L5/E-L6 后调整。

---

## 附录 A：公式速查

    expert_bytes = 3 × dim × moe_inter_dim / 2 + 3 × dim × moe_inter_dim / 32
                 = 17,694,720 + 1,105,920 = 18,800,640 B
    token_expert_bytes = 40 × 6 × 18,800,640 = 4,512,153,600 B
    nvme_time = token_expert_bytes × (1 − hit) / B_nvme
    pcie_time = (nvme_bytes + streamed_attn_bytes) / B_pcie
    tok/s ≈ 1 / (nvme_time + 0.35 × pcie_time + 0.06)
    prefill_layer_major_bytes = 40 × 384 × 18,800,640 = 288.78 GB（与 prompt 长度基本无关）
    prefill_chunked_bytes ≈ ceil(n_tokens / chunk) × 288.78 GB（每块重读，避免！）
    hot_pin_budget = 0.2 × expert_cache_bytes
    free_prefetch_bytes = free_compute_ms / 1000 × B_nvme

## 附录 B：配置开关草案

| 开关 | 默认 | 说明 |
|---|---|---|
| --expert-cache-gb | **20**（可配；0=按 available 自动） | Lead 拍板 20 GB 起步；上限 24；最低 4 |
| --cache-policy | arc | lru/arc/s3fifo |
| --pin-per-layer | 0（实验 20） | 热度 pin |
| --session-pin | 0.2 | 单会话 pin 不超过缓存的 20% |
| --adapt-interval | 128 token | 重算热度/pin |
| --expert-slots | 24 | VRAM 暂存槽 |
| --attn-resident | auto（29–34） | 可配 full（全 40 层）/ partial（auto）/ N |
| --direct-io | on | 专家流 NO_BUFFERING |
| --io-threads | 6 | 读取线程 |
| --prefetch | on（gate） | 总开关 |
| --prefetch-lead | 1 | 1 或 2 |
| --prefetch-k | 2 | 每层预测专家数 |
| --prefetch-min-conf | 0.5 | 置信度阈值 |
| --prefetch-free-ms | 3 | 计算窗口 |
| --gate-window | 16 | 统计窗口 |
| --gate-min-useful | 0.10 | 命中率下限 |
| --mtp | **off** | DSpark/MTP 可选档（+8 GB RAM） |
| --kv-resident | 32768 | VRAM KV tokens |
| --kv-q8 / --kv-q4 | off | KV 量化 |
| --engram-row-cache-mb | 256 | Engram 行缓存 |
| --pcie-frac | 0.5 | missed expert GPU/CPU 分流 |
| --max-sessions | 2 | 活跃解码会话 |
| --head | **cpu**（bf16） | cpu/fp8/bf16；fp8/int8 见 5.4.1，默认不启用 |
| --prefill-mode | layer-major | layer-major/chunked |

---

## 附录 C：与参考实现的对应关系

| 参考 | 结论 | 本项目采用 |
|---|---|---|
| DeepInfer hugging_face/weights.py | 专家按层聚合、engram 按行 chunk 读、避开 mmap | 采用：按 expert 两段聚合；Engram 行读 + 预取；不用 mmap |
| DeepInfer eplb | 冗余专家/负载均衡针对多 GPU 副本 | 不适用单机；借鉴"热度统计"用于 pin |
| flash-moe（58 实验） | 信任 OS cache、temporal -18%、mmap -5×、预取无 gate 亏 | 采用 O_DIRECT + 自管 cache；预取必须 gate；禁止 mmap |
| ds4 SSD_STREAMING | bounded expert cache、按字节预算、非专家常驻 | 采用：RAM cache 字节预算 + 非专家全驻 |
| Strata DETAILS | RAM+VRAM 总预算、KV streaming、adapt-swaps、pcie-frac、decode gate、多客户端 | 采用：预算模型、KV 分层、热度 pin、pcie-frac、gate、会话 slot |
| llama.cpp PR #25294 | 每层 n_slots 专家 cache、O_DIRECT、Wave-Partitional Prefill、单 context 限制 | 采用：O_DIRECT、prefill wave、全局槽；并发用全局 cache（已支持） |

---

*本文与 loading.py 同属 loading-researcher 写作用域；修改请同步更新 scripts/prototypes/loading_research/README.md 的实验口径。*

---

## 附录 D：工具输出快照（2026-10-21，RTX 5060 Laptop 8 GB / 47.2 GB RAM / Windows）

复现命令（在 tinyinfer 目录）：

~~~text
PS> .\.venv\Scripts\python.exe -m tinyinfer.loading selftest
PS> .\.venv\Scripts\python.exe -m tinyinfer.loading budget --expert-cache-gb 20
PS> .\.venv\Scripts\python.exe -m tinyinfer.loading sim --steps 128
~~~

说明：selftest 内部用 16 token 的模拟只是冒烟检查，命中率数字有噪声；正式扫描以 D.3 的 128-token 输出为准。测量期间下载器仍在写盘，NVMe 数字偏保守。

### D.1 selftest（摘录，完整输出以命令为准）

~~~text
budget plan:
VRAM total                                                     8.00 GB
  - desktop/WDDM reserve                                       0.90 GB
  - CUDA/runtime reserve                                       0.35 GB
  attention weights pinned (fp8, 170 MB/layer)                 34 layers = 5.78 GB
  attention weights streamed per token                         6 layers = 1.02 GB
  expert slots (18.8 MB each)                                  24 = 0.45 GB
  activations + KV window                                      0.42 GB
VRAM used                                                      6.65 GB (free 1.35)
RAM total                                                      47.20 GB
  - OS + apps reserve                                          6.00 GB
  non-expert resident (40 layers + embed + head + engram proj) 9.76 GB
  MTP/DSpark layers (3)                                        0.00 GB
  KV cache (RAM)                                               0.25 GB
  pinned staging buffers                                       0.80 GB
  headroom                                                     2.00 GB
  expert cache                                                 28.39 GB = 1509 experts (9.8% of all 15360)

IO scenarios (8.8 GB/s):
  hit=0.0% read= 4.51 GB nvme=   513 ms pcie=  213 ms ~1.55 tok/s
  hit=20.0% read= 3.61 GB nvme=   410 ms pcie=  178 ms ~1.88 tok/s
  hit=30.0% read= 3.16 GB nvme=   359 ms pcie=  161 ms ~2.10 tok/s
  hit=50.0% read= 2.26 GB nvme=   256 ms pcie=  126 ms ~2.77 tok/s

SELFTEST PASSED
~~~

（selftest 内默认 cache=auto：MTP 关、47.2 GB 全额可用时 28.39 GB；产品默认按 Lead 拍板用 20 GB，见 D.2。）

### D.2 budget（推荐默认：20 GB cache、MTP 关）

~~~text
VRAM total                                                     8.00 GB
  - desktop/WDDM reserve                                       0.90 GB
  - CUDA/runtime reserve                                       0.35 GB
  attention weights pinned (fp8, 170 MB/layer)                 34 layers = 5.78 GB
  attention weights streamed per token                         6 layers = 1.02 GB
  expert slots (18.8 MB each)                                  24 = 0.45 GB
  activations + KV window                                      0.42 GB
VRAM used                                                      6.65 GB (free 1.35)
RAM total                                                      47.20 GB
  - OS + apps reserve                                          6.00 GB
  non-expert resident (40 layers + embed + head + engram proj) 9.76 GB
  MTP/DSpark layers (3)                                        0.00 GB
  KV cache (RAM)                                               0.25 GB
  pinned staging buffers                                       0.80 GB
  headroom                                                     2.00 GB
  expert cache                                                 20.00 GB = 1063 experts (6.9% of all 15360)

hit=0.0%  read/token= 4.51 GB  NVMe=   513 ms  PCIe=  213 ms  ~1.55 tok/s
hit=20.0%  read/token= 3.61 GB  NVMe=   410 ms  PCIe=  178 ms  ~1.88 tok/s
hit=30.0%  read/token= 3.16 GB  NVMe=   359 ms  PCIe=  161 ms  ~2.10 tok/s
hit=50.0%  read/token= 2.26 GB  NVMe=   256 ms  PCIe=  126 ms  ~2.77 tok/s
~~~

### D.3 sim（合成 Zipf 0.9 + topic + 25% temporal，128 token × 40 层）

~~~text
policy           cache GB  hit rate  disk GB/token  speedup
lru                   4.0     0.000          4.512     0.00
arc                   4.0     0.039          4.336     0.00
s3fifo                4.0     0.042          4.324     0.00
arc+pin20             4.0     0.026          4.395     0.00
lru                   8.0     0.110          4.017     0.00
arc                   8.0     0.148          3.845     0.00
s3fifo                8.0     0.081          4.148     0.00
arc+pin20             8.0     0.136          3.899     0.00
lru                  16.0     0.191          3.651     0.00
arc                  16.0     0.225          3.497     0.00
s3fifo               16.0     0.131          3.922     0.00
arc+pin20            16.0     0.225          3.496     0.00
lru                  24.0     0.249          3.390     0.00
arc                  24.0     0.268          3.302     0.00
s3fifo               24.0     0.175          3.724     0.00
arc+pin20            24.0     0.269          3.297     0.00
lru                  32.0     0.280          3.249     0.00
arc                  32.0     0.305          3.137     0.00
s3fifo               32.0     0.210          3.563     0.00
arc+pin20            32.0     0.304          3.142     0.00
prefetch@0.4          0.0     0.400          0.902     1.11
prefetch@0.5          0.0     0.500          0.752     1.14
prefetch@0.6          0.0     0.600          0.602     1.17
prefetch@0.7          0.0     0.700          0.451     1.20
~~~

读法：命中率按 expert 请求计；disk GB/token 是 NVMe 实际读；prefetch 行的 "hit rate" 实际是预测 precision，"disk GB/token" 是被浪费的预取字节（落在计算窗口内的部分不算关键路径），speedup 是层时间模型（compute=3 ms、B=8.8 GB/s、k=2）。

### D.4 真实 layer-0 预检与 head 实测

- layer 0 真实：232/384 专家，top-24 41.9%，单层 LRU(27) 34.4%，相邻位置重叠 0.85/6（14.2%）。
- head CPU bf16：[129280,5120] 39–41 ms（16 线程）；fp32 80.9 ms；H2D bf16 pinned 46.3 ms、fp8 pinned 28.2 ms；torch rowwise fp8 _scaled_mm 在 sm_120 报 "Rowwise scaling is not currently supported on your device"。
- 详见 §5.4.1、§10 E-L3 预检。
