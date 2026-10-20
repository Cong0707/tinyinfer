# tinyinfer × llama.cpp / GGUF 参考研究

> 作者：llama-cpp-researcher · 2026-10-21
> 代码基线：`repositories/llama.cpp` @ `5ba3b9f`（master）、`repositories/ds4`、`repositories/flash-moe`
> 范围：GGUF 格式与加载、量化布局、ggml CPU/GPU 算子、流式专家、DeepSeek-V4 实现、失败经验、EXTREME 近似
> 证据等级：代码阅读 + 小规模原型；**未**下载/运行任何模型权重，未在本机编译 CUDA 后端做性能实测。文中所有性能数字若来自上游文档/PR/仓库，均已标注来源；未标注者是解析或设计估算。

---

## 0. 结论速览（先看这一页）

### 0.1 六个关键事实

1. **llama.cpp master 已经原生实现 DeepSeek-V4（不是 V4.1）**：`LLM_ARCH_DEEPSEEK4` 有完整推理图（Hyper-Connections + Sinkhorn、CSA/HCA 压缩 KV、lightning indexer top-k、hash routing、MTP）、CPU/CUDA 后端和 HF→GGUF 转换器。它是目前最完整的“可执行对照实现”，但仓库里找不到任何 `V4.1`/`deepseek41` 支持；V4.1 的 GGUF/tokenizer/图是 ds4 自己的一套（`general.architecture = "deepseek41"`）。**不能直接拿 llama.cpp 跑 tinyinfer 的目标权重，但可以拿它当 V4 的结构/算子 oracle。**
2. **DeepSeek-V4 的专家 fp4 与 GGUF MXFP4 是同一格式**：都是 E2M1 值 + 每 32 个元素一个 E8M0 scale；HF 与 GGUF 只差 nibble 排列（HF：偶数列低 4 位；GGUF：低 16 个元素低 4 位）。llama.cpp 的转换器只做位重排，**无损**。本报告原型已逐步验证（`selftest.py` 全绿）。
3. **PR #25294（MoE 专家磁盘流式）仍是 open，未合并**，但设计完整：每层 slot cache、router 后 id→slot remap 自定义 op、O_DIRECT 绕过 page cache、Wave-Partitional Prefill。它与 tinyinfer `ExpertStore` 的差距主要是“**把全局缓存改成按层 slot + 计算图内 remap**”，而不是缓存策略本身。
4. **ggml 的 CPU K-quant dot 与 CUDA mmq/mmvq 是现成蓝本**：AVX2 Q4_K×Q8_K 用 `maddubs` + 预计算 bsums 做 int32 累加；CUDA 对 MXFP4/NVFP4 在 Blackwell（sm_120a）走 **native W4A4 FP4 MMA**，llama.cpp 默认编译 `120a-real`。RTX 5060 Laptop 理论可用，但 26 SM 上的收益需实测。
5. **失败经验高度一致**：冷专家数据用 `mmap` 会因每页缺页而灾难性变慢；OS page cache 与自管 cache 的胜负取决于“工作集 / 可用 RAM”比值；预取在路由命中率低且窗口小时净负收益；O_DIRECT 必须处理对齐、短读、EINVAL/EFAULT 回退。
6. **EXTREME 档在 ggml 里有现成参考**：SIMD `expf` 是“Cody-Waite 范围缩减 + 5 阶多项式 + scalef”，不是低精度 Taylor；GELU quick 用 64K 项 fp16 查表（128 KB）；CUDA 全量开 `-use_fast_math`；MXFP4 的激活也能量化到 FP4（W4A4），这是比“换多项式”更大的杠杆。

### 0.2 给 tinyinfer 的三条最高优先级建议

| 优先级 | 建议 | 为什么 | 验收 |
|---|---|---|---|
| **P0** | 把 GGUF 适配器定义为“**只读索引 + 按需 slab 读取**”，先支持 llama.cpp 的 `deepseek4` 张量命名与 MXFP4；用 ds4 V4.1 GGUF 的元数据做交叉校验。**主线仍走 HF safetensors**，GGUF 只做参考/可选存储。 | GGUF 不能解决 V4.1 Engram/attention 的差异；但它能提供现成的量化/流式/算子对照，降低“自己发明格式”的风险。 | 解析任意 vocab GGUF + 合成 3D 专家 GGUF；MXFP4 重排与 HF 数值逐位一致。 |
| **P1** | 把 **Wave-Partitional Prefill** 的思路落到 `PrefillPlanner`：按“每层去重专家数 / slot 容量”切 wave，wave 间只预载下一 wave，当前 wave 的 masked pair 用 parking slot 占位；同时补 **O_DIRECT 探测 + 对齐 + 回退**。 | tinyinfer 已有 layer-major 预填充计划，但缺“超出 slot 时如何分波”的可执行算法；PR #25294 给出了 cap=(n_slots-n_expert_used)/2 的推导和 masked-pair 处理。 | 用合成 routes 跑 planner：每专家每 ubatch 只读一次；slot 数不满足 3×top-k 时明确报错。 |
| **P2** | 在 `VramSlotCache`/`ExpertStore` 上增加 **slot 代际(generation) + claimed 状态 + 热度衰减**，并把统计口径对齐 PR（hit/miss/cold miss、stall、wave stall、preload ready）。 | 这是支持异步预取/多 wave 时不读脏数据的最小机制；也是定位“到底慢在 I/O 还是等待”的必要观测。 | 并发/中断场景下不出现旧请求覆盖新 slot；stats 能算出每 token 磁盘字节与等待时间。 |

---

## 1. 基线与证据

### 1.1 本报告实际读了什么

| 仓库/路径 | 关键文件 | 用途 |
|---|---|---|
| llama.cpp | `ggml/include/gguf.h`、`gguf-py/gguf/{gguf_reader,constants,quants}.py` | GGUF 二进制格式、类型表、参考反量化 |
| llama.cpp | `ggml/src/ggml-common.h`、`ggml-quants.c`、`ggml-cpu/arch/x86/quants.c` | K-quant 块布局、标量/AVX2 dot |
| llama.cpp | `ggml-cpu/ggml-cpu.c`、`ggml-cpu/ops.cpp`、`ggml-cpu/vec.h` | mul_mat 调度、RMSNorm/RoPE/SwiGLU/FlashAttention、SIMD exp |
| llama.cpp | `ggml-cuda/mmq.cu`、`mmq.cuh`、`mmq-load-tiles.cuh`、`mmvq.cu`、`fattn*.cu(h)` | 量化矩阵乘、W4A4、FlashAttention |
| llama.cpp | `src/models/deepseek4.cpp`、`llama-kv-cache-dsv4.*`、`conversion/deepseek.py`、`conversion/base.py` | V4 图、压缩 KV、MXFP4 转换 |
| llama.cpp PR #25294 | diff（`llama-moe-stream.{h,cpp}`、`llama-graph.cpp`、`llama-model.cpp`） | 流式专家与 wave prefill |
| ds4 | `docs/SSD_STREAMING.md`、`docs/MODELS.md`、`gguf-tools/README.md`、`gguf-tools/deepseek41_metadata.py`、`ds4_ssd.c` | V4.1 GGUF/量化配方/流式经验 |
| flash-moe | `docs/io-and-gpu-exploration.md`、`docs/optimization-experiments-q4.md`、`paper/flash_moe.tex` | page cache/mmap/预取/压缩的失败与成功 |
| tinyinfer | `docs/loading.md`、`src/tinyinfer/{loading,quant,moe,tiers,approx_math}.py` | 现有设计、命名、接口 |

### 1.2 版本与支持矩阵（只列与本任务相关的）

| 能力 | llama.cpp master | ds4 | tinyinfer 现状 |
|---|---|---|---|
| DeepSeek-V4 文本 | ✅ `LLM_ARCH_DEEPSEEK4` | ✅ | 目标模型是 V4.1 |
| DeepSeek-V4.1 文本 | ❌ 未找到 | ✅ `deepseek41` | 目标 |
| V4.1 Engram 表 | ❌ | ✅ 行读、常驻磁盘 | 计划行读 |
| MXFP4 专家 | ✅ 转换 + CPU/CUDA | ✅ 原生 | HF fp4（同构） |
| FP8 非专家 | 转换时 dequant 到 bf16/Q8_0 | 自产 Q8_0/Q4_K 等 | HF fp8 直读 |
| 压缩 KV（CSA/HCA） | ✅ V4 | ✅ V4.1 | 设计 |
| Hyper-Connections | ✅ 融合算子 CPU/CUDA | ✅ | 参考实现 |
| MTP | ✅ V4（单 block） | ✅ DSpark/V4.1 | DSpark 计划 |
| MoE 专家磁盘流式 | ⏳ PR #25294 open | ✅ SSD streaming | ExpertStore |

> 结论：**llama.cpp 是 V4 的“结构 + 算子”参考，ds4 是 V4.1 的“格式 + 运行”参考**。两者不能互相替代。

### 1.3 一个必须说清的差异：V4 vs V4.1

- llama.cpp 的转换器明确按 `DeepseekV4ForCausalLM` 注册（`conversion/deepseek.py`），架构字符串为 `deepseek4`；仓库全文检索不到 `V4.1` 或 `deepseek41`。
- ds4 的 `deepseek41_metadata.py` 写入 `general.architecture = "deepseek41"`，并且使用自定义键：`deepseek41.config`、`deepseek41.engram.*`、`general.alignment = 16384`（Apple 16 KiB 页）等；`docs/MODELS.md` 明确写“V4.1 需要自己的 GGUF、tokenizer 和推理图；V4 Flash 权重与 DSpark 支持文件不可互换”。
- tinyinfer 的 `TinyConfig.arch_family()` 返回 `deepseek_v41`，所以任何 GGUF adapter 都必须把 `deepseek41` 与 `deepseek4` 分开处理，不能共用一个 tensor map。

---

## 2. GGUF 格式与 tinyinfer GGUF adapter 候选设计

### 2.1 二进制布局（以 `gguf.h` 头注释为准）

| 段 | 字段 | 类型/说明 |
|---|---|---|
| 0 | magic | 4 B，`"GGUF"`（小端 `0x46554747`） |
| 4 | version | u32；当前 3（reader 兼容 2/3） |
| 8 | tensor_count | u64 |
| 16 | kv_count | u64 |
| 20… | KV 列表 | 每项：key(string) + type(u32) + value |
| … | tensor info 列表 | 每项：name(string)、n_dims(u32)、dims(u64×n_dims)、type(u32)、offset(u64) |
| … | padding | 对齐到 `general.alignment`（默认 32，必须是 2 的幂） |
| … | tensor data | tensor info 中的 offset 是**相对 data 段起点**的偏移；每个 tensor 的字节数由 type/block 算 |

关键细节：

- string = `u64 length + bytes`（无 NUL）；bool 存 1 字节；枚举存 int32。
- dims 的顺序是 **ne0 最先**（最内层、连续维）。所以 2D 权重 `[ne0, ne1]` 在 NumPy/torch 里的形状是 `(ne1, ne0)`；3D 专家 `[ne0, ne1, ne2]` 是 `(ne2, ne1, ne0)`。这一点是 adapter 最容易写错的地方（本报告原型专门测了逻辑形状）。
- tensor 的 `n_bytes = n_elements / block_size × type_size`。量化类型的 ne0 必须整除 block（K-quant 256，MXFP4 32，Q8_0 32）。
- 对齐只保证 data 段起点；llama.cpp 的 writer 还会把每个 tensor 也 pad 到 alignment，但 reader **不能假设**每个 offset 都对齐（尤其第三方 GGUF）。O_DIRECT 读取要自己向上/向下取整并拷贝。
- 分片 GGUF 用 `split.no` / `split.count` / `split.tensors.count` 元数据；tensor info 仍分散在各分片，data offset 各自独立。

### 2.2 读取方式：先索引，后按需读；不要 mmap 整个 510 GB

- gguf-py 的 `GGUFReader` 用 `np.memmap` 把整个文件映射进来，再按 offset 切 tensor。这对“小模型/全驻留”很方便，但 tinyinfer 的 510 GB 权重**不能**整文件 mmap：地址空间和 page cache 都会被拖垮。
- llama.cpp 自己在 `llama-mmap.cpp` 中提供 `llama_mmap`：`MAP_SHARED`、可选 `MAP_POPULATE`、按 range `madvise(WILLNEED/RANDOM)`、加载后 `munmap` 未使用片段；但 **开启 MoE 流式时会自动关闭 mmap**（PR #25294 在 `llama.cpp` 里直接置 `params.use_mmap = false`，因为 mmap 的预取会把整个模型拉进 page cache，破坏流式）。这与 tinyinfer `loading.md` 的结论一致。
- 推荐 tinyinfer 的 reader 分两级：
  1. **Index 级**：只读文件头 + KV + tensor info（几十 KB 到几 MB，取决于 tokenizer）；用 `pread`/`ReadFile`，不 mmap。
  2. **Data 级**：按 tensor/slab 的 `(file, offset, nbytes)` 直接 `pread`（Linux）或 `FILE_FLAG_NO_BUFFERING`（Windows），复制到 RAM/VRAM；非专家常驻权重可以用 mmap（它们会被反复访问且总量可控），专家流式永远不要 mmap。

### 2.3 候选 adapter 设计（不改变主线 HF 加载）

目标不是“用 GGUF 替代 HF”，而是给 tinyinfer 增加一个**只读参考/可选存储层**，同时能把 ds4 的 V4.1 GGUF 当作对照输入。建议接口：

```python
@dataclass(frozen=True)
class TensorRef:
    name: str
    file: Path
    offset: int          # absolute file offset
    nbytes: int
    ggml_type: int       # 12=Q4_K, 39=MXFP4, ...
    dims: tuple[int, ...]  # GGUF order, ne0 first

class GGUFIndex:
    metadata: dict[str, Any]
    tensors: dict[str, TensorRef]
    expert_names: dict[int, dict[str, str]]  # layer -> {"w1"/"w2"/"w3": fused 3D name}
    tokenizer: GGUFVocab | None

class GGUFTensorReader:
    def read_into(self, ref: TensorRef, dest: memoryview, *, direct: bool = False) -> None: ...
    def dequantize(self, ref: TensorRef) -> np.ndarray: ...
    def expert_slab(self, layer: int, expert: int) -> list[ReadSegment]: ...
```

与 tinyinfer 现有结构的对接点：

| tinyinfer 模块 | 对接方式 | 注意 |
|---|---|---|
| `loading.ShardIndex` | 可新增 `GGUFIndex`，暴露同样的 `expert_blob(layer, expert)` / `region(name)` 接口 | expert blob 在 GGUF 里是 3D 大 tensor 的一个 slab：`offset + expert * slab_bytes`，不需要拼多段（与 HF 的两段式不同） |
| `loading.ShardReader` | 新增 `GGUFReader.read_into`；`direct=True` 走 O_DIRECT/NO_BUFFERING | 对齐、短读、回退逻辑见 `6.2` |
| `loading.ExpertStore` | 复用缓存/槽/预取；把 `_materialize` 的数据源换成 `GGUFIndex` | GGUF MXFP4 slab 可以直接作为 `uint8` 进入现有 `quant.dequant_fp4_block` 的输入（见 `3.4） |
| `quant.py` | 新增 `dequant_gguf_mxfp4`（或复用 `dequant_fp4_block`，因为语义相同） | 不要先 dequant 成 bf16 再缓存 |
| `tokenizer` | 优先用 HF tokenizer；GGUF 只作为 fallback/离线工具 | GGUF 的 tokenizer 是 `tokens + merges + pre`，不等于 HF `tokenizer.json` 的全部行为 |
| `server.py` | 不需要 GGUF 参与；模型名/元数据可以来自 GGUF | 不要把 API 层和格式层耦合 |

### 2.4 HF → GGUF 张量映射（DeepSeek-V4 系）

llama.cpp 的转换器（`conversion/deepseek.py` + `gguf-py/gguf/constants.py`）给出了完整映射。下表是 tinyinfer 最常用的子集（HF 名 → GGUF 名；`N` 为层号，`blk.N`）：

| HF（tinyinfer） | GGUF | 形状/说明 |
|---|---|---|
| `embed.weight` | `token_embd.weight` | `[vocab, dim]` → dims `(dim, vocab)` |
| `norm.weight` | `output_norm.weight` | |
| `head.weight` | `output.weight` | |
| `hc_head_fn/base/scale` | `output_hc_fn/base/scale` | V4 |
| `layers.N.hc_attn_fn/base/scale` | `blk.N.hc_attn_fn/base/scale` | |
| `layers.N.hc_ffn_fn/base/scale` | `blk.N.hc_ffn_fn/base/scale` | |
| `attn.wq_a / wq_b / q_norm` | `blk.N.attn_q_a / attn_q_b / attn_q_a_norm` | |
| `attn.wkv / kv_norm` | `blk.N.attn_kv / attn_kv_a_norm` | 命名有点反直觉 |
| `attn.wo_a / wo_b` | `blk.N.attn_output_a / attn_output_b` | wo_a 在 GGUF 里是 3D，按 o_groups reshape |
| `attn.attn_sink` | `blk.N.attn_sinks` | |
| `attn.compressor.*` | `blk.N.attn_compressor_{kv,gate,ape,norm}` | |
| `attn.indexer.*` | `blk.N.indexer.*` / `indexer_compressor_*` | |
| `ffn.gate.weight` | `blk.N.ffn_gate_inp.weight` | |
| `ffn.gate.bias` | `blk.N.exp_probs_b.bias`（hash 层可无） | |
| `ffn.gate.tid2eid` | `blk.N.ffn_gate_tid2eid.weight` | I32 |
| `ffn.shared_experts.w1/w2/w3` | `blk.N.ffn_{gate,down,up}_shexp.weight` | |
| `ffn.experts.i.w1/w2/w3` | `blk.N.ffn_gate_exps / ffn_down_exps / ffn_up_exps.weight` | 3D，`ne2 = n_expert`；`w1=gate, w2=down, w3=up` |
| `mtp.*` / `nextn.*` | `blk.N.nextn.*` | V4 的 MTP 单独处理 |

专家张量的关键点：

- GGUF 把**所有专家合并成一个 3D tensor**：`dims = (ne0, ne1, n_expert)`，数据按 expert 连续（expert-major 在最外层）。读取一个专家就是读一个连续 slab：`offset + expert * (ne0*ne1 bytes)`；这比 HF 的“6 个张量、weight/scale 两段”更适合流式。
- MXFP4 专家没有独立 scale tensor：scale 已经内嵌在每 17 字节的 block 里。HF 的 `w1.scale` 在转换时被“打包进” `block_mxfp4` 的 scale 字节。
- V4 的 hash routing 层可能没有 `exp_probs_b`，但有 `ffn_gate_tid2eid`；adapter 要按层判断，不能硬编码。

### 2.5 tokenizer 集成

GGUF 的 tokenizer 元数据（`gguf-py/gguf/constants.py` 的 `Keys.Tokenizer`）：

| 键 | 含义 | 备注 |
|---|---|---|
| `tokenizer.ggml.model` | `llama`(SPM) / `gpt2`(BPE) / `bert` / `t5` / `rwkv` 等 | DeepSeek V4 用 `gpt2` + BPE |
| `tokenizer.ggml.pre` | pre-tokenizer 类型 | 缺失会导致生成质量下降（llama.cpp 会告警） |
| `tokenizer.ggml.tokens` | 词表 string 数组 | 可能 10 万级 |
| `tokenizer.ggml.token_type` | 每个 token 的属性（normal/unknown/control/user_defined/byte） | 决定 special 处理 |
| `tokenizer.ggml.merges` | BPE merge 列表，`"a b"` 形式 | |
| `tokenizer.ggml.scores` | SPM score | |
| `tokenizer.ggml.bos/eos/unk/pad_token_id` | 特殊 id | |
| `tokenizer.ggml.add_bos_token` 等 | 是否自动加 | |

llama.cpp 在 `src/llama-vocab.cpp` 里按 `tokenizer.ggml.model` 分派到 SPM/BPE/UGM/WPM，并按 `pre` 选正则/归一化。**tinyinfer 不建议自己实现完整 tokenizer**：

- 主线用 HF `AutoTokenizer`（与训练/评测一致，风险最低）。
- GGUF 适配器只提供“只读 token 表 + merges”，用于：
  1. 离线工具/对照实验；
  2. 在没有 HF tokenizer 的 GGUF-only 场景做最小 BPE；
  3. 校验 GGUF 与 HF 的 token id 是否一致（这是最容易出静默错误的地方）。
- V4.1 的 tokenizer 还有 Engram 的 compressed vocab/prime 分配（`ds4/gguf-tools/deepseek41_metadata.py`）；这部分不要试图从 GGUF 通用键推导，直接读 ds4 的自定义键或 HF config。

### 2.6 原型验证结果

本报告附带 `scripts/prototypes/gguf/`：

- `gguf_reader.py`：无第三方依赖（只用 NumPy）的 header/KV/tensor 目录解析器；支持 13 种 KV 类型、数组、对齐、边界检查、按需 slab 读取、反量化。
- `quants_ref.py`：Q4_K/Q5_K/Q6_K/Q8_0/MXFP4 反量化 + HF→GGUF MXFP4 重排。
- `make_sample_gguf.py`：生成带 3D 专家/Q4_K/Q6_K/MXFP4/F32 的合成 GGUF。
- `selftest.py`：解析 llama.cpp 仓库里的真实 `ggml-vocab-deepseek-llm.gguf`（102400 tokens、0 tensors），并验证 5 种量化反量化与 gguf-py **逐位一致**。

运行结果（本机，2026-10-21）：

```text
[ok] real GGUF: ggml-vocab-deepseek-llm.gguf: version=3, kv=23, tensors=0, tokens=102400, arch='llama', alignment=32
[ok] synthetic GGUF: metadata, tensor directory, Q4_K/Q6_K/MXFP4 dequant
[ok] Q4_K: bit-exact vs gguf-py, shape=(8, 256)
[ok] Q5_K: bit-exact vs gguf-py, shape=(8, 256)
[ok] Q6_K: bit-exact vs gguf-py, shape=(8, 256)
[ok] Q8_0: bit-exact vs gguf-py, shape=(8, 32)
[ok] MXFP4: bit-exact vs gguf-py, shape=(8, 32)
[ok] MXFP4 repack: HF nibble order -> ggml blocks, 6x256 values bit-identical
SELFTEST PASSED
```

这说明“GGUF 头解析 + MXFP4 重排”不是纸上设计，已经可以运行；剩下的工作是把它接到 `ExpertStore` 和 tokenizer 上。

---

## 3. 量化格式对照

### 3.1 GGUF 量化类型总表（与本任务相关的子集）

| 类型 | block | 字节/block | bpw | scale 结构 | 典型用途 |
|---|---:|---:|---:|---|---|
| F32 | 1 | 4 | 32 | 无 | 小张量/参考 |
| F16 | 1 | 2 | 16 | 无 | 小张量 |
| BF16 | 1 | 2 | 16 | 无 | 小张量 |
| Q8_0 | 32 | 34 | 8.5 | fp16 | 高质量/非专家 |
| Q4_0 | 32 | 18 | 4.5 | fp16 | 老式 4-bit |
| Q2_K | 256 | 84 | 2.625 | fp16 d + fp16 dmin + 4-bit packed scales/mins | 2-bit 混合 |
| Q4_K | 256 | 144 | 4.5 | fp16 d/dmin + 6-bit scales/mins | Q4_K_M 主体 |
| Q5_K | 256 | 176 | 5.5 | 同 Q4_K + 1 个 high bit | Q4_K_M 的 qkv/敏感张量 |
| Q6_K | 256 | 210 | 6.5625 | fp16 d + int8 scales | 输出/敏感层 |
| IQ2_XXS | 256 | 66 | 2.0625 | fp16 d + 4-bit sub-scale + codebook+sign | ds4 Q2 / 极端压缩 |
| IQ4_XS | 256 | 136 | 4.25 | fp16 d + 6-bit scales + codebook | 4-bit 非均匀 |
| IQ4_NL | 32 | 18 | 4.5 | fp16 d + 非均匀 4-bit kvalues | 4-bit 非均匀 |
| MXFP4 | 32 | 17 | 4.25 | E8M0/32 + E2M1 | DeepSeek-V4 专家、GPT-OSS |
| NVFP4 | 64 | 36 | 4.5 | UE4M3/16 + E2M1 | 需要 per-16 scale 的 FP4 |

> bpw 按 `type_size * 8 / block_size` 计算；Q4_K_M 是**文件级配方**，不是 tensor 类型，平均通常在 4.8–4.9 bpw（依模型张量占比）。

### 3.2 Q4_K / Q5_K / Q6_K 的字节布局与公式

#### Q4_K（144 B / 256 值）

```text
offset  size  field
0       2     d      fp16  (super-block scale)
2       2     dmin   fp16  (super-block min scale)
4       12    scales 6-bit packed (8 sub-block scales + 8 mins)
16      128   qs     4-bit quants: 4 chunks of 32 bytes
```

数值公式：`x_i = d * sc[g] * q_i - dmin * mn[g]`，其中 `g = i/32`，`q_i ∈ [0,15]`。

- `qs` 的一个 32 字节 chunk 覆盖 64 个元素：低 nibble 是前 32 个，高 nibble 是后 32 个；对应两个 sub-block（scale 索引 2j / 2j+1）。
- 6-bit scales/mins 的打包：
  - 前 4 个 scale 的低 6 位在 `scales[0..3]`，前 4 个 min 的低 6 位在 `scales[4..7]`；
  - 后 4 个 scale 的低 4 位在 `scales[8..11]` 的低 nibble，后 4 个 min 的低 4 位在高 nibble；
  - 后 4 个的高 2 位分别藏在 `scales[0..3]` 和 `scales[4..7]` 的 bit6-7。
- 解包就是 `get_scale_min_k4()`（C）或本报告 `quants_ref._q4k_scales_mins()`（NumPy）。

#### Q5_K（176 B / 256 值）

比 Q4_K 多一个 `qh[32]`（`offset 16..48`），其余后移：`qs[128]` 在 `48..176`。每个 32 值 sub-block 的 5th bit 来自 `qh[l]` 的第 `2j/2j+1` 位；数值公式同 Q4_K，只是 `q_i ∈ [0,31]`。

#### Q6_K（210 B / 256 值）

```text
offset  size  field
0       128   ql    低 4 位
128     64    qh    高 2 位
192     16    scales int8 (16 个/16 值)
208     2     d     fp16
```

数值：`x = d * sc[g] * (q - 32)`，`q ∈ [0,63]`，`g = i/16`。

- 每 128 个元素为一组：`q1` 用 `ql[l]&0xF + (qh[l]&3)<<4`，`q2` 用 `ql[l+32]&0xF + ((qh[l]>>2)&3)<<4`，`q3` 用 `ql[l]>>4 + ((qh[l]>>4)&3)<<4`，`q4` 用 `ql[l+32]>>4 + ((qh[l]>>6)&3)<<4`；顺序是 `q1,q2,q3,q4`（各 32 个），不是交错。
- 本报告原型一开始就是在这里写错过（用了 `np.stack` 而不是 `np.concatenate`），selftest 立刻抓出 234/256 个元素不一致。**这类错误必须用参考实现逐位对照，不能靠“看起来对”。**

#### Q8_0（34 B / 32 值）

`d(fp16) + 32×int8`，`x = d*q`。简单、快，适合非专家/输出，但不适合作为“压缩率”手段。

### 3.3 Q4_K_M / Q5_K_M 的混合策略（llama.cpp `llama-quant.cpp`）

Q4_K_M 不是“所有张量都用 Q4_K”，而是在 base type 上按张量类别和层号升级：

| 张量类别 | Q4_K_M | Q5_K_M |
|---|---|---|
| 路由专家 `ffn_*_exps` | Q4_K（或按 `--expert-tensor-type`） | Q5_K |
| `attn_v` / `attn_qkv` / `attn_kv_b` | `use_more_bits` 的层 → Q6_K，否则 Q4_K | `use_more_bits` 的层 → Q6_K，否则 Q5_K |
| `ffn_down` | `use_more_bits` 的层 → Q6_K，否则 Q4_K | `use_more_bits` 的层 → Q6_K，否则 Q5_K |
| `output` | Q6_K（若 ne0 能被 256 整除） | Q6_K |
| `token_embd` | Q4_K（除 tied/特殊） | Q5_K |

其中 `use_more_bits(i, n) = i < n/8 || i >= 7n/8 || (i - n/8) % 3 == 2`。直觉是：首尾层和每 3 层里的第 3 层更敏感，多给 1–2 bit。注意 `category_is_attn_v()` 把 `attn_qkv`、`attn_kv_b` 也归入 attn_v，所以它们与 `attn_v` 同分支；文件里另有一个单独的 `ATTENTION_QKV` 分支，但当前分类下 Q4_K_M/Q5_K_M 不会命中它。

**对 tinyinfer 的启示**：

- 如果要做一个“GGUF-only 对照”或离线压缩，Q4_K_M 的“按张量类别 + 层号升级”比“全模型统一 4-bit”更划算；但 tinyinfer 主线已经是 fp4 专家 + fp8 非专家，**没有必要再引入 Q4_K**（多一套 dequant/布局/性能路径）。
- 如果未来要做 2-bit 兜底，ds4 的配方更贴近我们的模型：`Q2 = IQ2_XXS gate/up + Q2_K down + Q8_0 attention/shared/head`；`Q4 = Q4_K experts + Q8_0 其余`；MXFP4 = 原生专家 + Q8_0 其余。见 `ds4/gguf-tools/deepseek41_quantize.py` 和 `download_model.sh`。

ds4 的 V4.1 发布配方（来自 `download_model.sh` / `gguf-tools/README.md`）：

| 发布 | 文件大小 | 主权重 | 专家配方 | 备注 |
|---|---:|---:|---|---|
| `ds41f-q2` | 341 GiB | 152 GiB | IQ2_XXS gate/up + Q2_K down + Q8_0 attention/shared/head | 189 GiB Engram；imatrix 校准 |
| `ds41f-q4` | 483 GiB | 294 GiB | Q4_K 专家 + Q8_0 其余 | 两个分片，脚本自动合并 |
| `ds4f-mxfp4`（V4） | ~156 GB | — | 原生 MXFP4 专家 + Q8_0 其余 | V4 不是 V4.1 |

注意：这些是 ds4 的“整文件发布”大小（GiB），不是 tinyinfer 的 HF 510.286 GB 直接换算；V4.1 的 Engram 表在 ds4 中始终按行从磁盘读，不常驻。

### 3.4 MXFP4 / NVFP4 与 tinyinfer 的 fp4 / fp8

#### 3.4.1 结论：DeepSeek-V4 专家 fp4 ≡ GGUF MXFP4（无损）

- HF 存储：`packed uint8 [rows, cols/2]`，元素 `2i` 在低 nibble、`2i+1` 在高 nibble；scale `uint8 [rows, cols/32]`，每 32 个元素一个 E8M0 字节。
- GGUF `block_mxfp4`：每 32 个元素 `1 byte scale + 16 bytes codes`；byte `j` 的低 nibble = 元素 `j`，高 nibble = 元素 `j+16`。
- 两者 codebook 相同（E2M1：`0,0.5,1,1.5,2,3,4,6` 及负值）。llama.cpp 的 `repack_mxfp4_blocks()` 只做 nibble 重排，不 requantize；本报告 `repack_hf_mxfp4()` 在 6×256 随机数据上验证输出**逐位相同**。
- 因此：**tinyinfer 的 fp4 专家可以直接作为 GGUF MXFP4 的 slab 输入/输出**，不需要改数值；只需要重排 nibble 和把 scale 插入每个 block。

#### 3.4.2 与 NVFP4 的差异

| | MXFP4 | NVFP4 |
|---|---|---|
| block | 32 | 64（4 个 16 值 sub-block） |
| scale | E8M0/32 | UE4M3/16（无全局 scale） |
| 值 | E2M1 | E2M1 |
| 字节 | 1 + 16 = 17 B | 4 + 32 = 36 B |
| 适用 | DeepSeek-V4 专家、GPT-OSS | 需要更细粒度 scale 的 FP4 |

tinyinfer 的专家是 MXFP4 语义；**不要**用 NVFP4 的 per-16 scale 去套，除非重新量化。

#### 3.4.3 我们的 FP8 非专家没有 GGUF 对应

- tinyinfer 的 dense/attention 权重是 FP8 E4M3 + 32×32 block E8M0（`quant.dequant_fp8_block`）。
- GGUF 没有“FP8 + 2D block scale”类型；llama.cpp 的 DeepSeek-V4 转换器在 `dequant_model()` 里先把 FP8 反量化成 bf16，然后 `tensor_force_quant()` 对 2D 张量强制 Q8_0（`DeepseekV4Model.tensor_force_quant`）。这是**有损**的（FP8 的 32×32 block scale 被 Q8_0 的 per-32 单 scale 取代）。
- 建议：主线保留 HF FP8 直读；GGUF 对照版本接受 Q8_0 的损失，并在报告里标注“对照版本不是逐位一致”。
- 若一定要在 GGUF 里无损保留 FP8 block scale，只能自定义 tensor type / sidecar，代价是 llama.cpp 生态不认；不值得。

### 3.4.4 一页对照：我们的格式 vs GGUF

| tinyinfer 格式 | 最接近的 GGUF 类型 | 布局差异 | 能否直接互转 | 结论 |
|---|---|---|---|---|
| 专家 FP4 E2M1 + E8M0/32（packed 2/byte） | **MXFP4** | 仅 nibble 顺序（偶/奇 vs 低 16/高 16）；scale 同为 1 byte/32 | ✅ 无损位重排 | 直接作为 GGUF MXFP4 使用/存储 |
| 非专家 FP8 E4M3 + E8M0 32×32 block | 无；最接近 Q8_0/BF16 | FP8 的 2D 32×32 scale 在 Q8_0 里变成 per-32 单 scale | ❌ 有损 | 主线保留 HF FP8；GGUF 对照接受 Q8_0 |
| KV FP8/FP4（模型设计） | Q8_0/Q4_0 KV | scale 粒度/布局不同 | ❌ 需重排/重算 | 用 llama.cpp KV 量化做参考，不复用数值 |
| 2-bit 兜底（如果要做） | IQ2_XXS（+Q2_K） | codebook+sign vs E2M1 | ❌ 需重新量化 | 仅 CPU/极端压缩；必须 imatrix |
| 4-bit 非均匀（如果要做） | IQ4_XS/IQ4_NL | codebook vs E2M1 | ❌ | 不适用于 FP4 硬件路径 |
| Q4_K/Q5_K/Q6_K | 自身 | 8 个 32 值 sub-block + 6-bit scale/min | — | 不建议引入主线；可做对照 |

### 3.5 IQ2_XXS / IQ4_XS 的 codebook 思路

IQ2_XXS 用 `grid[256][8]`（8 个 2-bit 值）+ 128 项 sign 表 + 每 32 值 4-bit scale + fp16 super-scale，做到 2.0625 bpw。IQ4_XS 用 16 项非均匀 4-bit 码本 + 6-bit scale，4.25 bpw。

IQ 系列其余类型（IQ1_S/IQ1_M/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/IQ4_NL）的工程套路相同：**grid 码本 + ksigns 符号表 + 分组 scale +（通常）imatrix**；差别只在码本维数、符号位/高比特打包和 scale 位数。它们的解码都是查表/移位，不走 FP4 tensor core。

可借鉴的只有“**码本 + 符号 + 分组 scale**”这一思想，不适合直接替换我们的 FP4：

- 我们的硬件路径（Blackwell FP4 tensor core / W4A4）要求 E2M1 硬件格式；IQ codebook 需要查表/移位，无法走 FP4 MMA。
- 如果 EXTREME 档要 2-bit CPU 回退，IQ2_XXS 是成熟选项；但必须配 imatrix（llama.cpp 的 `--imatrix` / ds4 的 `.dat`）。没有 imatrix 的 IQ2_XXS 质量不可接受。
- imatrix 的思想对 tinyinfer 有用：按激活重要性给专家/通道加权；但它需要一次校准前向，成本不低。

### 3.6 给 tinyinfer 的量化选择（建议）

| 权重类别 | 当前/建议 | GGUF 对照 | 理由 |
|---|---|---|---|
| 路由专家 | HF MXFP4 原样，零拷贝 | MXFP4（无损） | 与硬件 FP4 对齐；不要 dequant 常驻 |
| 共享专家 | FP8 block → bf16 计算 | Q8_0 | 占比小，精度优先 |
| attention 投影 | FP8 block，常驻/流式 | Q8_0（有损） | 8 GB 下常驻层数有限 |
| head | bf16 CPU 或 FP8 | Q8_0/Q6_K | 实测 CPU bf16 37.8–40.8 ms，见 loading.md |
| KV | FP8/FP4（模型设计） | q8_0/q4_0 KV | llama.cpp 支持 `--cache-type-k/v` |
| 输出/embedding | bf16/fp8 | Q6_K/Q8_0 | 不参与专家流式 |

### 3.7 W4A4：最大的精度/速度旋钮，也是最大的风险

- llama.cpp 的 MXFP4/NVFP4 在 Blackwell 上可以走 **W4A4**（激活也量化到 FP4，每 32/16 值一个 scale），走 native FP4 MMA；否则走 W4A8（激活 Q8_1 + DP4A/MMA）。
- 控制开关：`GGML_CUDA_MMQ_PREC=q4|q8|auto`（`docs/build.md`）；默认 auto 会按 per-tensor 元数据选择，而很多模型标注为 W4A16，默认实际走 W4A8。要强制 W4A4 必须显式 `q4`。
- 对 tinyinfer：先用 W4A8 做正确性基线，再在 EXTREME 档尝试 W4A4；必须测 PPL/长文/工具调用，不能只看 tok/s。
- 26 SM 的 RTX 5060 上，W4A4 的 tensor core 吞吐是否被带宽/occupancy 吃掉，需要实测；llama.cpp 的 mmq 已为小 batch 做了 tile 选择（`J=8..128`）和 fallback。


---

## 4. ggml 算子：CPU/GPU 可复用点

### 4.1 CPU `mul_mat`：调度比指令更值得抄

`ggml-cpu/ggml-cpu.c:1255` 的 `ggml_compute_forward_mul_mat`：

1. 查 `type_traits_cpu[src0->type]` 得到 `vec_dot`、`vec_dot_type`、`nrows`。K-quant 的 `vec_dot_type` 是 **Q8_K**；MXFP4/NVFP4 是 **Q8_0**；也就是“权重保持量化，激活量化到 8-bit 后做整数点积”。
2. 如果 `src1` 类型不是 `vec_dot_type`，先把激活量化/拷贝到 `wdata`（多线程按 block 切分）。
3. 用 `atomic_fetch_add` 抢 chunk：`chunk_size=16`（当 `nr0==1` 或 `nr1==1` 时 64），按 `nchunk0 × nchunk1` 二维分块；如果块数不够（`< nth*4`）或 NUMA，退化为按行/按列分线程。
4. 每个 chunk 内部再做 16×16 的 block tiling，调用 `vec_dot`，把结果写回 `dst`。`nrows=2` 的 ARM MMLA 路径一次算两行/列，其他架构为 1。

对 tinyinfer 的启示：

- **不要按“层”粗粒度并行**，而是按 16×16 tile 做 work-stealing；否则 26 SM/多核的尾延迟会很差。
- 激活量化（Q8_K/Q8_0）是每次 matmul 的固定成本；在 decode 的 GEMV 里，权重读取是主成本，激活量化可以忽略；在 prefill 的 GEMM 里，激活量化可以摊薄。
- `ggml-cpu/repack.cpp`、`tiled/tiled.cpp` 是 AVX512/AMX 的“预重排 + 分块 GEMM”路径；tinyinfer 的 CPU fallback 不必到这一步，先实现 `vec_dot` 级即可。

### 4.2 Q4_K × Q8_K 的 AVX2 点积（可直接照搬的整数技巧）

`ggml-cpu/arch/x86/quants.c:2102` 的 `ggml_vec_dot_q4_K_q8_K`：

- 把 12 字节 `scales` 解包成 8 个 scale 和 8 个 min（同 `_q4k_scales_mins`）。
- 每个 super-block 的 `d = y.d * x.d`、`dmin = -y.d * x.dmin`；其中 `y.d` 是激活 Q8_K 的 per-256 scale。
- 权重 4-bit × 激活 int8：用 `_mm256_maddubs_epi16`（无符号×有符号，得到 int16 对和），再用 `_mm256_madd_epi16` 乘 scale 并累加 int32。
- min 项不逐元素乘：利用 Q8_K 预计算的 `bsums`（每 16 个激活的和）与 min 做点积，最后一次性加 `dmin * Σ(min*bsum)`。
- 整个 super-block 用整数累加，最后才转 float 乘 `d`；避免每元素浮点乘。

这个结构对 tinyinfer 的 CPU 专家计算非常合适：MXFP4 的 E2M1 值可以映射成整数码本（0..12 带符号），与 Q8_0 激活做整数点积，再乘 `e8m0_half * y.d`。**CPU 端不需要把 fp4 反量化成 bf16 再 matmul**（见 `4.3 的 MXFP4 dot）。

### 4.3 CPU 端 MXFP4：现成的 int8 点积

`ggml-cpu/quants.c:296` 的 `ggml_vec_dot_mxfp4_q8_0_generic`：

```c
const float d = y[ib].d * GGML_E8M0_TO_FP32_HALF(x[ib].e);
for (int j = 0; j < QK_MXFP4/2; ++j) {
    sumi1 += y[ib].qs[j + 0]          * kvalues_mxfp4[x[ib].qs[j] & 0xf];
    sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
}
sumf += d * (sumi1 + sumi2);
```

其中 `kvalues_mxfp4` 是“加倍后的 E2M1 整数码本”（0,1,2,3,4,6,8,12 及负值），`E8M0_TO_FP32_HALF` 是 `2^(e-128)`；两者相乘正好等于真实值。**这是把 FP4 变成“整数 4-bit 权重 + per-block scale”的最干净写法**，也适合用 torch/int8 kernel 复刻，不必依赖 FP4 tensor core。

### 4.4 RMSNorm / RoPE / SwiGLU

- **RMSNorm（CPU）**：`ggml-cpu/ops.cpp:4191` 按行切线程，两遍扫描（平方和 + scale），支持 `F32`；另有 `ggml_compute_forward_rms_norm_mul_fused()` 把“RMSNorm × 权重”融合成一次写出（`ops.cpp:4276`）。tinyinfer 的 tilelang 融合 RMSNorm+RoPE 思路更激进，但注意 ggml 只融合了“norm + mul”，没有把 RoPE 也塞进去；RoPE 需要 cos/sin cache，融合的收益取决于 cache 复用。
- **RoPE（CPU）**：`ggml-cpu/ops.cpp:6287` 的 `ggml_rope_cache_init()` 为每个位置预计算 `cos/sin`，支持 `freq_factors`、YaRN 的 `corr_dims`、`ext_factor/attn_factor/beta_fast/beta_slow`；`rotate_pairs()` 支持 NORMAL（相邻对）、NEOX（前后半对）、MROPE、VISION。**DeepSeek-V4 的 q/k 是“前 nope 不转、后 rope 转”**：llama.cpp 用 `ggml_rope_set_offset(cur, n_embd_head_nope)` 让 RoPE 只作用在最后 `n_rot` 维（见 `src/models/deepseek4.cpp`）。tinyinfer 的 `attention.py` 应该按同样的“partial RoPE + offset”实现，而不是把整个 head_dim 都转。
- **SwiGLU（CPU）**：`ggml-cpu/ops.cpp:3342` 的 `ggml_compute_forward_swiglu_f32` 按行切线程，调用 `ggml_vec_swiglu_f32()`（`vec.cpp:418`）；该函数用 SIMD 的 `ggml_v_silu()`（多项式 exp）再乘 gate。另有 `swiglu_oai`、`swiglu_clamp` 变体；**DeepSeek-V4 的 `swiglu_limit` 对应 `GGML_GLU_OP_SWIGLU_CLAMP`**（`ops.cpp:3655`），不是普通 SwiGLU。tinyinfer 的 `moe.Expert.forward` 已实现 clamp，保持即可。

### 4.5 CPU FlashAttention

`ggml-cpu/ops.cpp:8933` 起有 `ggml_compute_forward_flash_attn_ext_f16`（one chunk）与 `ggml_compute_forward_flash_attn_ext_tiled`：

- 支持 `F16` Q/K/V、`F32` 输出、`sinks`（attention sink，`9130` 起对第一个 KV chunk 做修正）、`logit_softcap`、mask。
- tiled 版本按 KV chunk 分块，适合长上下文；one-chunk 版本适合短上下文。
- 这说明“CPU 也能跑带 sink 的 flash attention”，但性能有限；tinyinfer 的 CPU fallback 可以先用它做正确性参照（不是性能路径）。

### 4.6 CUDA：mmq / mmvq 的选择逻辑

llama.cpp 在 `ggml-cuda.cu` 里按“类型 + batch + 架构”选 kernel：

- **MMQ**（`mmq.cu`）：把激活量化到 Q8_1（或 FP4），权重 tile 进 shared memory，用 DP4A/MMA 做矩阵乘。`ggml_cuda_should_use_mmq()`：
  - Turing 及以上的 NVIDIA：只要 smem ≥ 48 KiB，**总是 true**（即 prompt/prefill 优先 MMQ）；
  - 老的 NVIDIA：没有 DP4A 时，MoE（`n_experts>0`）仍优先 MMQ；
  - 其他：`ne11 < MMQ_DP4A_MAX_BATCH_SIZE(64)` 时用 MMQ，否则回 cuBLAS。
- **MMVQ**（`mmvq.cu`）：decode 的 GEMV 路径，`MMVQ_MAX_BATCH_SIZE=8`；Blackwell 上按类型细化：Q4_K 在 `ne11<=5`、Q5_K `<=6`、Q6_K `<=7` 时用 MMVQ，超过就 MMQ。
- **MMVF**（`mmvf.cu`）：F16 权重的 GEMV。
- **MMQ 的 W4A4**：MXFP4/NVFP4 在 `blackwell_mma_available()`（编译了 `120a` 及以下、非 Rubin）时，若精度选 Q4，则走 native FP4 MMA；否则 Q8 激活走 W4A8。`mmq.cuh` 的 `block_fp4_mmq` 把 MXFP4 的 2 个 E8M0 打包进一个 u32、NVFP4 的 4 个 UE4M3 打包进一个 u32，与 Q8_1 的 `block_q8_1_mmq` 同大小（方便共用 tile 代码）。

对 tinyinfer：

- decode 的专家 GEMV（M=1..8）应优先 **MMVQ 风格**（每线程读连续权重、激活向量在 shared/L1 复用）；prefill 的 GEMM（M≥16）用 **MMQ/tile**。这与 tinyinfer 当前“专家矩阵乘用小 batch”一致。
- 如果走 tilelang 自研 kernel，可复用 mmq 的“激活 Q8/FP4 量化 + 权重 tile + int32 累加 + per-block scale”结构；不要每个元素先 dequant 成 bf16。
- Blackwell 的 native FP4 需要 `sm_120a` 编译目标；llama.cpp 默认 arch 列表包含 `120a-real`（`ggml-cuda/CMakeLists.txt`）。tinyinfer 的 tilelang 0.1.15 是否支持 `sm_120a` 未验证，这是 P2 的实验项。

### 4.7 CUDA FlashAttention 与 DSV4 融合算子

- `ggml-cuda/fattn.cu` 按 head_dim 分派到 `fattn-vec`（decode，D=64/128/256/512）、`fattn-tile`、`fattn-mma-f16`（prefill/大 batch）。**D=512 和 D=576（QK 576/V 512）都有专门分支**，支持 sinks、softcap、mask；这正是 MLA/DeepSeek 的 head 结构。对 tinyinfer 的 sparse MLA（D=512）是一个可参考的 kernel 结构（尤其 vec 路径的 online softmax + sink）。
- DSV4 专用融合（`src/models/deepseek4.cpp` + `ggml-cuda/dsv4-hc.cu` + `lightning-indexer.cu`）：
  - `DSV4_HC_PRE/COMB/POST`：把 HC 的“RMSNorm + 线性 mix + sigmoid + Sinkhorn 迭代 + 加权残差”融成 3 个 op；CPU 也有实现（`ggml-cpu/ops.cpp:11427` 起）。
  - `LIGHTNING_INDEXER`：把 indexer 的 Q/K/weights 乘、mask、top-k 前的 score 融合；CPU 也有实现（`ops.cpp:12445`）。
  - 这些融合的收益是“少写中间张量 + 少 kernel launch”，对 8 GB 卡尤其重要。tinyinfer 的 tilelang 可以把 HC 的 3 个阶段分别融合，但不要把 Sinkhorn 的 20 次迭代全塞进一个 kernel（会占用过多寄存器/shared）。
- MoE：`ggml-cuda/mmid.cu`、`topk-moe.cu`、`moe-weighted-reduction.cu` 是 MUL_MAT_ID / top-k / combine 的现成实现；tinyinfer 的 `moe.py` 仍是 Python 循环，性能瓶颈在 kernel launch，后续可以按“group tokens by expert + 一次 MUL_MAT_ID + weighted reduction”重构。

### 4.8 与 tinyinfer 模块的对应表

| tinyinfer 模块 | 可借鉴 | 不建议 |
|---|---|---|
| `attention.py` | partial RoPE + offset、sink 的 online softmax 修正、D=512 的 tiled/vec 结构、top-k mask 生成 | 直接照搬 ggml graph（Python 里没必要） |
| `kernels.py` | mmq 的激活量化/整数累加/scale 应用；W4A4 的量化/反量化 | 把整个 mmq tile 调度搬进 tilelang（先做单 tile） |
| `gemm.py` | MMVQ vs MMQ 的 batch 阈值；专家 GEMV 用 Q8 激活 | 在 decode 用大 tile GEMM |
| `nn.py` | fused RMSNorm×weight、SwiGLU clamp、残差/combine 融合 | 过度融合 Sinkhorn |
| `moe.py` | group-by-expert、MUL_MAT_ID、weighted reduction | Python 逐专家循环（launch 开销） |
| `loading.py` | slot remap、wave partition、O_DIRECT 探测（见 `5） | 用 mmap 读冷专家 |
| `approx_math.py` | ggml 的 SIMD exp 多项式、fp16 查表（见 `7） | 无范围缩减的 Taylor |

---

## 5. 流式专家 PR #25294 vs tinyinfer ExpertStore

### 5.1 PR 机制（按执行顺序）

1. **模型加载**：每个被流式的专家 tensor（`ffn_gate_exps` / `up` / `down`）不 materialize，而是创建一个 3D cache tensor `{ne0, ne1, n_slots}`（与源 tensor 同 quant 类型），并记录 `(file_idx, offset, nb_expert)`。这一层的 cache 在 GPU/CPU 后端 buffer 上分配。
2. **router 之后**：在 top-k ids 后插入一个自定义 op `llama_moe_stream_remap`（`llama-moe-stream.cpp`）：
   - 收集本 ubatch 触达的 distinct experts（按首次出现顺序）；
   - 对每个 expert：命中则取 slot；未命中则选 victim（空 slot 优先，然后 route-hotness 最小、last_use 最旧），标记 LOADING、`slot_gen++`，把 load 请求推给 I/O 线程池；
   - 等待本批所有 demand load 完成（`cv_done`），然后把每个 id 改写成 slot id。
3. **expert GEMM**：后续 `MUL_MAT_ID` 用 slot id 索引 cache tensor；因为 expert 集合与顺序未变，输出与不流式版本一致（CUDA 上 bit-exact；CPU repack 路径可能末位不同）。
4. **I/O**：默认 buffered pread；`--moe-stream-direct` 时用 O_DIRECT。O_DIRECT 的 buffer/offset/length 都按 4096 对齐；读取时向上/向下取整，返回 staging 中间指针；短读/EINVAL/EFAULT 回退 buffered（重新 open，因为 O_DIRECT 是 fd 属性）。
5. **eviction**：`route_hotness` 饱和递增，每 64×streamed_layers 次 remap 全部右移 1 位（衰减）；victim 选择“最小 hotness，last_use 最旧”。LOADING/keep slot 不 evict。
6. **stats**：remap 次数、hit/miss/cold miss、stall 总时间/每次调用、wave 次数、preload issued/ready、wave stall。
7. **限制**：同一 model 的多个 `llama_context` 共享一个 cache，**并发 decode 会互相破坏**（PR 明确写 single-context only）；单 context 内 `--parallel N` 安全。

### 5.2 Wave-Partitional Prefill 的推导

当 prefill 的一个 ubatch 触达的 distinct experts `n_touch_max > n_slots` 时，不能一次性把所有专家放进 cache。PR 把专家分成 `n_waves` 个 wave：

```text
cap = (n_slots - n_expert_used) / 2
n_waves = ceil(n_touch_max / cap)
约束：cap >= n_expert_used  =>  n_slots >= 3 * n_expert_used
```

为什么是 `(n_slots - n_expert_used)/2`：cache 同时要容纳

1. 当前 wave 的专家（cap 个）；
2. 下一 wave 预载的专家（cap 个，用于和当前 wave 计算重叠）；
3. 当前 token 行里被 mask 掉的 pair 需要“停车位”（parking slots），最多 `n_expert_used` 个。

每个 wave 的图：

- 自定义 op `wave_ids`：把本 wave 的专家载入 cache，并写出“每个 (token, expert) pair 应索引的 slot”；
- 自定义 op `wave_mask`：本 wave 的 pair 输出 1.0，其他 wave 输出 0.0；
- expert GEMM 的输出乘 mask 后累加；
- 每个专家在每个 ubatch **只加载一次**，所以长 prompt 的预填充 I/O 从“按 block 重读”变成“每层扫一遍专家”。

PR 的 benchmark（GB10，GLM-5.2 Q2，254 GB，cache 64/90 slots）：prefill 从 2.28→5.65 tok/s、2.88→5.69 tok/s（waved vs no waves）；decode ~1.83/2.20 tok/s，hit 73%/79%。这说明 wave 分区对“专家数 >> slot 数”的预填充是数量级收益。

### 5.3 与 tinyinfer ExpertStore 的对比

| 维度 | PR #25294 | tinyinfer `ExpertStore` |
|---|---|---|
| cache 单元 | 每层一个 slot tensor；按 expert slab | 全局 RAM blob cache（18.8 MB/专家）+ VRAM 固定槽 |
| 命中方式 | router id → slot 重映射（计算图内） | Python 侧 `expert_slot` / `VramSlotCache` 查找 |
| 替换策略 | route-hotness + LRU tie-break（每层独立） | ARC / LRU / S3-FIFO + pin（全局） |
| I/O | 9 线程池（默认，最大 18），O_DIRECT 可选 | 6 线程 `ThreadPoolExecutor`，Windows NO_BUFFERING 可选 |
| 预取 | 下一 wave 预载；不预测 token | 支持预测器 + `prefetch_to_vram`；有 gate |
| 预填充 | wave 分区 + parking slot + mask | `PrefillPlanner` 有 wave 计划，但无“超 slot 时计算图内分波” |
| 缓存预算 | `--moe-stream-cache NG` 或 `Ns` 自动换算 slot 数 | `cache_bytes` / `vram_slots` |
| 统计 | hit/miss/cold、stall、wave/preload | stats 有 hit/miss/bytes，缺 wave stall/ preload ready |
| 并发 | 单 context 限制 | 多会话共享全局 cache（设计上更强） |
| mmap | 开启流式自动关闭 | 主线不用 mmap 读专家 |

结论：**PR 的“计算图内 slot remap + 逐层 cache”适合 GPU 端零拷贝计算；tinyinfer 的“全局 RAM cache + VRAM 槽”适合多会话/大 RAM 的 Python 运行时。** 两者可以组合：RAM cache 继续全局 ARC；VRAM 槽按层划分 slot；在 MoE 计算前做一次“expert id → VRAM slot”的 remap（Python 或自定义 kernel），并用 generation 防止异步加载读脏。

### 5.4 具体可借鉴/落地

1. **slot 状态机**：`EMPTY/LOADING/RESIDENT` + `generation` + `claimed`；异步 pread 完成时检查 generation，旧请求直接丢弃。tinyinfer 的 `VramSlotCache` 目前是同步 copy_in，建议补这三个字段。
2. **wave 分区**：在 `PrefillPlanner.plan_layer()` 里增加“如果 `n_unique > slots.prefill_compute`，按 `cap = max(1, (slots - n_activated)/2)` 切 wave”；每个 wave 只保留当前专家 + 下一 wave 预载；输出 mask。`loading.md` 的 `pseudo_code` 已有雏形，但缺停车位和 mask 的明确语义。
3. **O_DIRECT 探测/回退**：启动时对每个文件做一次 4 KiB 对齐读探测；若失败，关闭 direct 并重开 buffered。Linux 用 `O_DIRECT`；Windows 用 `FILE_FLAG_NO_BUFFERING`（tinyinfer 已有 `_WindowsDirect`，但需要 per-thread handle；PR 在 Windows 上退化为“加锁 seek+read”，因为缺 `pread`）。
4. **热度和衰减**：PR 的 `route_hotness` 是每层独立、每 64 token 衰减；tinyinfer 的 `pin_hot` 是全局频率、每 128 token 重算。两者可以并存：slot victim 用“层内 hotness”，pin 用全局。
5. **统计口径**：至少记录 `miss_cold`（首次见到的专家）、`stall_ms`、`wave_count`、`preload_ready`；否则无法区分“I/O 慢”和“等 I/O”。
6. **不要照搬**：PR 的“每层 cache 固定 slot 数”在 8 GB 卡上会浪费（24 槽 × 18.8 MB = 451 MB 已接近预算）；tinyinfer 的全局槽更合适。PR 的“单 context”限制也不应继承。

### 5.5 风险

- **3×top-k 约束**：`n_slots >= 3 * n_expert_used` 在 top-k=6 时是 18 槽；tinyinfer 默认 24 槽，够，但 prefill 时如果 `n_slots` 被降到 16 就会 abort。必须在 CLI 层校验。
- **mask 的数值**：wave mask 用 0/1 乘在 GEMM 输出上；如果使用 bf16/fp16 累加，跨 wave 求和顺序会变，可能引入末位差异。正确性测试要固定顺序。
- **parking slot 重复**：Metal 要求一个 token 行内不能重复用同一个 slot；PR 用 `pool_used` 保证。tinyinfer 若在 CUDA 上做，需确认 `MUL_MAT_ID` 是否允许重复 id；不允许就必须同样做 parking。
- **单 context 共享 cache 的坑**：tinyinfer 是多会话，不能直接抄“全局 slot”；必须把 slot 归属到 session/step，或者串行化同一层的 remap。

---

## 6. 失败经验与风险清单

### 6.1 mmap 冷专家：灾难性的 5×（flash-moe 实测）

- flash-moe 用 `mmap + memcpy` 读 3.9 MB 专家：0.56 tok/s，比 `pread` 慢 5×。原因是每个 3.9 MB 专家跨 240 个 16 KiB 页，冷数据需要 240 次缺页/内核陷入；一次 `pread` 只发一个大 NVMe 命令。
- 但 mmap 对**常驻的非专家权重**很好（5.5 GB 在 48 GB 机器上完全命中）；tinyinfer 可以把非专家 mmap、专家 `pread`，不要“一刀切”。
- llama.cpp 开 MoE 流式时直接禁用 mmap（PR #25294），因为 `MAP_POPULATE`/`MADV_WILLNEED` 会把整个模型拉进 page cache。**这和 tinyinfer 的“专家不用 mmap”一致。**

### 6.2 O_DIRECT 的坑（PR 实现细节）

- 对齐：offset、length、buffer 都必须是 4096 的倍数（4096 是 512/4096 逻辑块大小的公倍数）。读一个 18.8 MB 的专家会多读头尾各最多 4 KiB，可忽略。
- 短读：`pread` 可能返回 < 请求长度（尤其 signal/EOF），必须循环读；PR 用 `EINTR` 重试 + 循环。
- 回退：某些文件系统/网络盘接受 `O_DIRECT` 但拒绝对齐读（EINVAL/EFAULT）；PR 在启动时做一次探测，失败就**重开 buffered**（O_DIRECT 是 fd 属性）。
- Windows：没有 `pread`，PR 在 Windows 上用“加锁的 `seek + read_raw`”串行化（I/O 线程池退化为串行）；tinyinfer 的 `_WindowsDirect` 是更好的方案（`FILE_FLAG_NO_BUFFERING` + 每线程 scratch），但要注意每线程独立 handle，不能共享 scratch。
- 文件系统缓存：O_DIRECT 绕过 page cache 不等于绕过控制器缓存；对“模型远大于 RAM”的场景，PR 的结论是 **O_DIRECT 是必须的**（否则 4.5 GB/token 会把非专家和 OS 缓存全部冲掉）。对 34 GB RAM / 4.5 GB per token 的 tinyinfer，20 GB 专家 cache + O_DIRECT 是合理组合。

### 6.3 page cache vs 自管 cache：不是非黑即白

- flash-moe（48 GB RAM / 209 GB 专家）删掉 9.8 GB Metal LRU 后 +38%，因为 Metal buffer 会 wired 内存，挤掉 OS page cache，且 OS 的 CLOCK-Pro 更聪明。**结论是“小 RAM + 大工作集 + 统一内存”下 Trust the OS 更好。**
- llama.cpp PR（GB10 128 GB / 254 GB 模型）用 O_DIRECT + 自管 slot cache，因为模型远大于 RAM，page cache 无法提供有效命中，反而和专家流争内存。**结论是“远大于 RAM”时自管 + 绕过 page cache 更好。**
- tinyinfer（34 GB RAM / 510 GB 模型 / 4.5 GB/token）：模型远大于 RAM，但 20 GB 缓存能覆盖约 25% 的专家读取；**自管 ARC + O_DIRECT** 是合理折中（与 loading.md 一致）。不要因为 flash-moe 的“Trust the OS”就删掉 ARC——两者的 RAM/working-set 比例不同。
- 经验公式：若 `cacheable_working_set / RAM > 2`，page cache 基本无用，走 O_DIRECT；若 `< 1`，让 OS 管更划算；中间地带用“自管 cache + buffered”并测命中率。

### 6.4 预取：命中率 × 窗口，缺一不可

- flash-moe：temporal 预测 25.6% 命中，但 K=4 全中的概率 `0.256^4 ≈ 0.4%`；预测失败还会浪费带宽，端到端 **-18%**。MLP 预测 31%，仍不够。
- tinyinfer `loading.md` 的数学：预取的硬条件是“GPU 计算窗口 ≥ 预取字节 / 带宽”；本模型窗口约 3 ms → k≤1.4，k=2 已略溢出。**预取必须 gate**（连续 16 层命中率 <10% 就关）。
- llama.cpp PR 的“预取”不是 token 预测，而是**同一 ubatch 内下一 wave 的专家**：命中率接近 100%（因为 wave 计划已知），所以没有预测风险。这是更稳的“预取”。
- 对 tinyinfer：优先做“wave 内预载下一 wave”（确定性），token 级预测只作为可选加速，并保留 gate。

### 6.5 压缩、GPU 缓存、其他负结果

- LZ4 专家压缩：flash-moe -13%（解压成本 > 冷读节省）；2-bit 时 F_NOCACHE +3% 但质量崩（JSON 引号变反斜杠）。**压缩不是 free lunch。**
- GPU private buffer 压缩：-20% 流水线（blit 4×7 MB 成本 > matvec 节省）。
- Expert file clustering：0%（NVMe 在 7 MB 粒度不在乎 scatter）。
- 投机解码：MoE 流式下每个候选 token 都要自己的路由/I/O，MTP 基本 break-even（flash-moe 分析）；tinyinfer 默认关 MTP 是合理的。
- 多 context 共享流式 cache：PR 明确会破坏输出；tinyinfer 必须做 session 归属或串行化。

### 6.6 准确性风险

- **FP8→Q8_0**：GGUF 对照版本与 HF 不是逐位一致；不要把对照版本的 PPL 当作主线指标。
- **W4A4**：激活量化误差会放大；必须用长文/工具调用/数学题测，不能只看短 prompt。
- **IQ2_XXS**：无 imatrix 不可用；ds4 的 Q2 配方也是“gate/up IQ2_XXS + down Q2_K + attention Q8_0”的混合，不是全 IQ2。
- **MXFP4 的 E8M0=0xFF**：llama.cpp 不处理 NaN（注释明确“we don't need NaNs”）；如果 HF 权重有 NaN scale，转换/反量化会得到 inf。tinyinfer 应在加载时校验 scale 字节。
- **tokenizer 不一致**：GGUF 与 HF 的 token id 错位是静默错误；必须用同一段文本逐 token 对比。

---

## 7. EXTREME 档（近似数学）的 ggml 参考

### 7.1 SIMD exp：范围缩减 + 多项式 + scalef

`ggml-cpu/vec.h:1191`（AVX512）/`1234`（AVX2）：

```c
// exp(x) = 2^n * poly(b), n = round(x*log2e), b = x - n*ln2
z = fma(x, 1.442695, 0x1.8p23);  // round-to-nearest trick
n = z - 0x1.8p23;
b = fnmadd(n, ln2_hi, fnmadd(n, ln2_lo, x));
u = b*b;
j = fma(fma(fma(c7,b,c5),u,fma(c3,b,c1)),u,fma(c1,b,1)); // 5 阶
res = scalef(j, n);
```

- 误差上界注释：`1.45358 + 0.5 ulp`；>88.38 溢出为 inf，<-103.97 下溢为 0。
- 这不是“低精度近似”，而是**用 FMA 换 SFU**；在 CPU 上比 `expf` 快，在 GPU 上 `-use_fast_math` 已经把它变成 `__expf`。
- tinyinfer 的 `approx_math.fast_exp` 是“`2^(x*log2e)` + 3 阶多项式”，误差 ~5e-3；如果 EXTREME 需要更低误差，可以照抄 ggml 的 5 阶 + hi/lo 拆分，但 torch 没有 `scalef`，需要 `ldexp` + 位操作，成本可能不划算。**更现实的做法是把 fast_exp 写进 tilelang/CUDA kernel，用 `ex2.approx` 或 `__expf`。**

### 7.2 fp16 查表

`ggml-cpu/vec.cpp:6`：

```c
ggml_fp16_t ggml_table_gelu_f16[1 << 16];       // 128 KB
ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16]; // 128 KB
```

`ggml_vec_gelu_quick_f32()` 把 float 转 fp16 取 16-bit 索引，查表后再转回 float（`vec.h:1047`）；误差受 fp16 量化限制（~1e-3 量级），但比多项式更省指令。**适合 SiLU/GELU 这种“逐元素、形状固定”的激活**；softmax 的 exp 输入动态范围大，查表容易溢出/精度差，不建议。

### 7.3 CUDA 的 fast math 与 W4A4

- `ggml-cuda/CMakeLists.txt:196`：`set(CUDA_FLAGS -use_fast_math -extended-lambda)`。这意味着 `expf/sqrtf/...` 被替换为 `__expf` 等；llama.cpp 默认接受这个误差。
- **比多项式更大的杠杆是 W4A4**：MXFP4/NVFP4 的激活也量化到 FP4，直接走 native FP4 MMA；`GGML_CUDA_MMQ_PREC=q4` 强制。代价是激活量化误差，收益是 tensor core 吞吐/带宽。
- KV 量化：llama.cpp 支持 `--cache-type-k q8_0` / `q4_0` 等；对 tinyinfer 的 KV8/KV4 是现成参考（但布局/scale 与 DeepSeek 的 FP8/FP4 KV 不同，不能直接复用数值）。
- FlashAttention 本身是“用 online softmax 换内存”的近似（数学等价，数值不同）；`fattn-vec` 的 sink 修正是 `exp(sink - max)` 加进分母，tinyinfer 的 sink 实现要对齐这个公式。

### 7.4 对 tinyinfer EXTREME 的建议

| 手段 | 参考 | 预期 | 风险 |
|---|---|---|---|
| 多项式 exp/silu | ggml vec.h | 中 | 误差累积到 softmax/路由 |
| fp16 LUT（SiLU/GELU） | ggml table | 中 | 输入范围/NaN |
| W4A4 专家 | llama.cpp mmq | 高 | 精度，需长文评测 |
| KV4/KV8 | llama.cpp cache-type | 高 | 已有模型设计，别重复量化 |
| partial RoPE / 融合 norm | ggml | 低-中 | 实现复杂度 |
| 层跳过/低秩 | tinyinfer `approx.py` | 高 | 质量崩，需逐层验证；ggml 没有可抄的 |

**原则**：先做“数学等价但实现不同”的融合（RoPE/norm/SwiGLU/FA），再做“有界误差”的近似（exp/LUT），最后才做“改变模型”的近似（层跳过/低秩/FFT）。ggml 能提供前两类的工程参考，第三类没有现成答案。


---

## 8. server / tokenizer / MTP 附注

### 8.1 llama.cpp server 的可借鉴点（api-engineer 可直接对照）

`tools/server/README.md` 的 OpenAI 兼容层不是本任务重点，但有几处与 tinyinfer 的 `server.py` 直接相关：

| 能力 | llama.cpp | 对 tinyinfer 的启示 |
|---|---|---|
| OpenAI 端点 | `/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings`、SSE | 保持 OpenAI 兼容；错误码/流式结束 event 要一致 |
| slot | `--parallel N`、`/slots` 监控、`id_slot` 指定、`--slot-save-path` 保存/恢复 | tinyinfer 已有会话 slot；补“保存/恢复”和监控端点 |
| prompt cache | `--slot-prompt-similarity`（默认 0.10）、`--cache-idle-slots`、`--ctx-checkpoints` | 相同前缀复用 KV/缓存；idle 时把 slot 存盘 |
| 统一 KV | `--kv-unified`、`--kv-unified-per-slot N` | 多会话共享 KV 池，按 slot 限额 |
| 投机/推理控制 | `--reasoning-control`、`/v1/chat/completions/control` | DeepSeek 思考模式需要可中断/可调 effort（V4.1 的 `--think-level` 在 ds4 中） |
| 指标 | `/metrics`（Prometheus）、`/props`、`/health` | 至少暴露 tok/s、cache hit、队列长度、slot 状态 |
| KV 量化 | `--cache-type-k/v q8_0/q4_0` | 与 tinyinfer 的 KV8/KV4 设计对照 |
| 模板/工具 | `--jinja`、`--chat-template-file`、工具调用 parser | V4.1 的 DSML/工具调用模板需要单独处理 |

tinyinfer 的 `server.py` 已经有 OpenAI 兼容层；建议只补“slot 级 prompt cache + 统计端点”，不要为了 GGUF 改 server。

### 8.2 tokenizer 的最终建议

- **主线**：HF `AutoTokenizer` + 原始 `tokenizer.json`，与训练/评测完全一致。
- **GGUF 适配器**：只读 token 表/merges，用于：
  - 校验 GGUF 与 HF 的 token id 是否一致（同一段文本逐 token 对比）；
  - GGUF-only 的离线实验；
  - 从 ds4 V4.1 GGUF 里提取 `deepseek41` 的自定义 tokenizer 元数据。
- **不要**用 `gguf-py` 的 `vocab.py` 直接替代 HF tokenizer：它覆盖了很多 BPE 变体，但 V4.1 的 tokenizer 还有额外规则（Engram compressed vocab、special token 处理），HF 是权威。
- 如果将来要“只发布 GGUF”，则必须把 `tokenizer.ggml.pre` 写对，并用 llama.cpp 的 `llama-tokenize` 与 HF 逐 token 对齐；这是发布前的硬门槛。

### 8.3 MTP / DSpark

- llama.cpp 的 DSV4 有 `graph_mtp`（`src/models/deepseek4.cpp:1359`），但只支持 **单个 MTP block**；tinyinfer 的模型有 3 个 MTP 层 + DSpark 头。
- ds4 的 V4.1 DSpark 是独立支持 GGUF + 单独的 drafter 图（`ds4_deepseek41_dspark.h`），不能和主模型 GGUF 混用。
- 对 tinyinfer：MTP 默认关闭是合理的（`loading.md` 的结论：MoE 流式下每个候选 token 都要自己的专家 I/O，收益可能 break-even）。如果要做，先做“draft 不触发专家 I/O”的版本（例如只 draft 用共享专家/小模型），否则会放大磁盘流量。
- llama.cpp 的 MTP 实现可以作为“如何把 MTP 层接进主图”的参考，但数值/张量布局要按 tinyinfer 的 DSpark 重新对。

---

## 9. 原型：`scripts/prototypes/gguf/`

### 9.1 文件

| 文件 | 作用 |
|---|---|
| `quants_ref.py` | Q4_K/Q5_K/Q6_K/Q8_0/MXFP4 反量化；HF→GGUF MXFP4 重排；E2M1/E8M0 表 |
| `gguf_reader.py` | 无依赖 GGUF header/KV/tensor 目录解析；按需 `raw()` / `dequantize()`；CLI |
| `make_sample_gguf.py` | 生成带 3D 专家、Q4_K/Q6_K/MXFP4/F32 的合成 GGUF |
| `selftest.py` | 真实 vocab GGUF + 合成 GGUF + 与 gguf-py 逐位对比 + MXFP4 重排 |

### 9.2 运行

```powershell
cd E:\Develop\Deepseek\tinyinfer
.\.venv\Scripts\python.exe scripts\prototypes\gguf\selftest.py

# 解析任意 GGUF
.\.venv\Scripts\python.exe scripts\prototypes\gguf\gguf_reader.py path\to\model.gguf --tensors

# 生成合成样本并反量化
.\.venv\Scripts\python.exe scripts\prototypes\gguf\make_sample_gguf.py --out sample.gguf
.\.venv\Scripts\python.exe scripts\prototypes\gguf\gguf_reader.py sample.gguf --dequant blk.0.ffn_gate_exps.weight
```

### 9.3 已验证 / 未验证

已验证：

- GGUF v3 头、KV（含数组/字符串）、tensor info、对齐、data offset；
- 真实 llama.cpp vocab GGUF（102400 tokens）；
- IQ2_XXS 等 IQ 类型的 header 解析（只校验 nbytes；反量化未实现）；
- Q4_K/Q5_K/Q6_K/Q8_0/MXFP4 与 gguf-py **逐位相同**；
- HF MXFP4 packed+scale → GGUF block，数值逐位相同；
- 3D 专家张量的逻辑形状（`(n_expert, ne1, ne0)`）。

未验证（有意留给后续）：

- 多分片 GGUF 的合并/去重；
- 真实 DeepSeek-V4/V4.1 GGUF 的完整张量名/形状（需要下载权重，本任务禁止）；
- tokenizer 的 BPE 解码与 HF 对齐；
- O_DIRECT 在本机 NVMe 上的对齐读性能（已有 tinyinfer 的 `_WindowsDirect` 实测）；
- 解析速度/内存（当前是 Python 逐字段，10 万 token 的 vocab 文件约 1–2 秒，模型文件只有 KB 级元数据）。

### 9.4 如果要接到 `ExpertStore`

最小改动：

1. `GGUFIndex` 提供 `expert_slab(layer, expert) -> (file, offset, nbytes)`；
2. `ShardReader` 增加 `read_slab(ref, dest)`，复用现有 `_WindowsDirect`/`pread`；
3. `ExpertStore._materialize` 根据 `index.kind` 选择 HF 两段拼 blob 或 GGUF 单段 slab；
4. `quant.dequant_fp4_block` 直接接收 GGUF `block_mxfp4` 的 `uint8`（语义相同）；若使用 llama.cpp 的 `MXFP4` tensor type，注意 nibble 顺序已是 GGUF 的（低 16/高 16），与 HF 的（偶/奇）不同——**在 adapter 层做一次重排，不要在 dequant 里做分支**；
5. 加一个 `selftest`：同一专家分别从 HF 和 GGUF 读取，dequant 后逐位比较（MXFP4 应完全相等）。

---

## 10. 可执行清单（按 tinyinfer 模块）

### P0（正确性/可行性，1–2 天）

| # | 模块 | 动作 | 验收 |
|---|---|---|---|
| P0.1 | `scripts/prototypes/gguf` | 已完成的 parser/dequant/selftest 接入 `tests` 或 `scripts` 的 CI（不依赖大文件） | `selftest.py` 绿；有真实 vocab GGUF 的回归 |
| P0.2 | `quant.py` | 增加 `repack_hf_mxfp4()` 的 torch 版本（或调用 NumPy 后转 tensor） | 与 `base.py` 的 `repack_mxfp4_blocks` 逐位一致 |
| P0.3 | `loading.py` | 在 `ExpertStore` 里加 `index.kind`（`"hf"` 或 `"gguf"`）与 GGUF slab 读取路径 | 同一专家 HF/GGUF dequant 逐位一致（MXFP4） |
| P0.4 | `tokenizer` | 写一个“HF vs GGUF token id 对齐”脚本（读 GGUF tokens/merges，只做对照，不替换主线） | 同一段文本 token id 完全一致；不一致时打印首个差异 |

### P1（性能/稳定性，3–5 天）

| # | 模块 | 动作 | 验收 |
|---|---|---|---|
| P1.1 | `loading.py` | O_DIRECT/NO_BUFFERING **探测 + 回退**；4 KiB 对齐 staging；每线程 handle | 在 NTFS/ReFS 上 direct 开/关都能读；短读/EINVAL 有日志与回退 |
| P1.2 | `loading.py` | `VramSlotCache` 增加 `generation/claimed/state`；异步 `copy_in` 完成时校验 | 并发预取不覆盖已消费 slot；取消/关闭不崩 |
| P1.3 | `loading.py` / `PrefillPlanner` | wave 分区：`cap=(slots - topk)/2`，parking slot + mask；每专家每 ubatch 只读一次 | 合成 routes 下 I/O 字节 = 每层去重专家数 × 18.8 MB；slot 不足时明确报错 |
| P1.4 | `loading.py` | stats 对齐 PR：hit/miss/cold/stall/wave/preload | 能输出“每 token 磁盘字节、命中率、等待 ms、wave 数” |
| P1.5 | `moe.py` | group tokens by expert + 一次批量 expert GEMM（减少 Python 循环） | 数值与逐专家循环一致；kernel launch 数下降 |

### P2（研究/优化，按需）

| # | 模块 | 动作 | 验收 |
|---|---|---|---|
| P2.1 | `kernels.py` | tilelang 实现 MXFP4 W4A8（Q8 激活）单 tile；对照 llama.cpp `kvx` 的整数点积 | 数值在 1e-3 内；RTX 5060 上比 bf16 dequant+mm 快 |
| P2.2 | `kernels.py` | `sm_120a` W4A4 native FP4 实验（若 tilelang 支持） | 速度/精度报告；默认关闭 |
| P2.3 | `attention.py` | partial RoPE offset + sink 的 online softmax；对照 ggml `fattn-vec` | 与参考实现误差 <1e-3；decode 延迟下降 |
| P2.4 | `approx_math.py` | f16 LUT SiLU/GELU；5 阶 exp（如需更低误差） | 误差上界/速度报告；只在 EXTREME 开 |
| P2.5 | `server.py` | slot 级 prompt cache + `/metrics` | 同前缀 TTFT 下降；指标可观测 |

### 不建议做（明确排除）

- 为了 GGUF 而放弃 HF 直载：V4.1 的 Engram/attention/tokenizer 差异会让 GGUF 版成为另一个项目。
- 把 GGUF 的 Q4_K/Q5_K 作为主线专家格式：tinyinfer 已经是 FP4，重新量化没有收益，且会丢掉硬件 FP4 路径。
- 在没有 imatrix/评测的情况下上 IQ2_XXS：质量风险不可控。
- 把 MTP 默认打开：MoE 流式下 I/O 放大，先做“draft 不触发专家 I/O”的方案。
- 用 mmap 读冷专家：已有 5× 反例。

---

## 11. 参考索引（文件:位置）

### llama.cpp

- `ggml/include/gguf.h`：GGUF 头注释、`gguf_type`、API
- `gguf-py/gguf/gguf_reader.py`：reader 实现；`constants.py`：`GGML_QUANT_SIZES`、`GGUFValueType`、`GGMLQuantizationType`
- `gguf-py/gguf/quants.py`：Q4_K/Q5_K/Q6_K/Q8_0/MXFP4 参考反量化
- `ggml/src/ggml-common.h`：`block_q4_K`/`q5_K`/`q6_K`/`q8_0`/`mxfp4`/`nvfp4`
- `ggml/src/ggml-quants.c`：`dequantize_row_q4_K`（1530）、`q5_K`（1732）、`q6_K`（1940）、`get_scale_min_k4`（880）
- `ggml/src/ggml-cpu/ggml-cpu.c`：`ggml_compute_forward_mul_mat`（1255）、work-stealing（1460 附近）
- `ggml/src/ggml-cpu/arch/x86/quants.c`：`ggml_vec_dot_q4_K_q8_K`（2102）
- `ggml/src/ggml-cpu/quants.c`：`ggml_vec_dot_mxfp4_q8_0_generic`（296）
- `ggml/src/ggml-cpu/ops.cpp`：RMSNorm（4191/4276）、RoPE cache（6287）、SwiGLU（3342）、CPU FA（8933）
- `ggml/src/ggml-cpu/vec.h` / `vec.cpp`：`ggml_v_expf`（1191/1234）、`ggml_vec_silu_f32`（380）、fp16 表（vec.cpp:6）
- `ggml/src/ggml-cuda/mmq.cu`：W4A4 选择（117）、`should_use_mmq`（355）；`mmq.cuh`：`block_fp4_mmq`、派发（580/740）
- `ggml/src/ggml-cuda/mmvq.cu`：`should_use_mmvq`（318）、`MMVQ_MAX_BATCH_SIZE=8`
- `ggml/src/ggml-cuda/fattn.cu`：D=512/576 分支、sinks；`fattn-vec.cuh`：sink 修正（381）
- `ggml/src/ggml-cuda/CMakeLists.txt`：`120a-real`（62–67）、`-use_fast_math`（196）
- `src/models/deepseek4.cpp`：V4 图（HC/LID/attention/MTP）、张量列表（82–185）
- `src/llama-kv-cache-dsv4.h` / `.cpp`：压缩 KV/状态
- `conversion/deepseek.py`：V4 转换、MXFP4、MTP；`conversion/base.py`：`repack_mxfp4_blocks`（918）
- `docs/build.md`：`GGML_CUDA_MMQ_PREC`（311 附近）；`docs/ops.md`：后端支持矩阵
- PR #25294（open）：`src/llama-moe-stream.{h,cpp}`、`src/llama-graph.cpp`、`src/llama-model.cpp`、`common/arg.cpp`

### ds4

- `docs/SSD_STREAMING.md`：缓存预算/自动 sizing/measurements
- `docs/MODELS.md`：V4.1 Q2/Q4 大小、Engram 常驻磁盘
- `gguf-tools/README.md`：Q2/Q4/MXFP4 配方、imatrix、V4.1 converter
- `gguf-tools/deepseek41_metadata.py`：`general.architecture="deepseek41"`、16 KiB 对齐、Engram 元数据
- `gguf-tools/deepseek41_quantize.py`：Q2/Q4 配方（IQ2_XXS/Q2_K/Q4_K/Q8_0）
- `download_model.sh`：模型族与大小（ds41f-q2 341 GiB / q4 483 GiB）
- `ds4_ssd.c` / `ds4_gpu.h`：缓存计划与流式 API、hotlist

### flash-moe

- `docs/io-and-gpu-exploration.md`：mmap 5×、自定义 cache -38%、F_RDADVISE 0%、dispatch_io -70%
- `docs/optimization-experiments-q4.md`：FMA dequant +2.6%、LZ4 -13%、temporal -18%、MTP break-even
- `paper/flash_moe.tex`：Trust the OS（`sec:trust_os`）、page cache/预取分析

### tinyinfer

- `docs/loading.md`：三级内存、缓存策略、预取数学、O_DIRECT 结论
- `src/tinyinfer/loading.py`：`ExpertStore`/`ExpertCache`/`VramSlotCache`/`PrefillPlanner`
- `src/tinyinfer/quant.py`：fp8/fp4 反量化；`moe.py`：gate/expert/MoE
- `src/tinyinfer/approx_math.py`：fast_exp/silu/rsqrt；`tiers.py`：EXTREME 配置

---

## 附录 A：一页纸公式

- GGUF tensor bytes = `nelem / block × type_size`；dims 顺序 ne0 最内。
- Q4_K: `x = d·sc·q − dmin·m`；Q5_K 同 + 5th bit；Q6_K: `x = d·sc·(q−32)`。
- MXFP4: `x = 2^(e−127) · E2M1(code)`；HF nibble：偶/奇；GGUF：低 16/高 16。
- Expert slab（GGUF 3D）：`offset = tensor.offset + expert × (ne0×ne1×bytes_per_elem / block)`。
- Wave: `cap = (n_slots − n_expert_used)/2`；`n_slots ≥ 3·n_expert_used`。
- 预取窗口：`bytes_prefetch ≤ compute_window × BW`。

## 附录 B：术语

- **slab**：一个专家在 3D 专家 tensor 里的连续字节段。
- **slot**：VRAM/设备上的固定专家缓存位置；不是数据源。
- **wave**：prefill 时按 slot 容量切分的专家子集。
- **remap**：把 router 输出的 expert id 改写成 cache slot id；不改变专家选择。
- **W4A4/W4A8**：权重 4-bit，激活 4-bit/8-bit。
- **MXFP4/NVFP4**：OCP 微缩放 FP4 格式；前者 E8M0/32，后者 UE4M3/16。
