# GGUF 原型（tinyinfer）

本目录是 `docs/llama_ref.md` 的可运行附件，用来验证：

1. 不依赖 gguf-py 也能解析 GGUF v3 的 header / KV / tensor 目录；
2. Q4_K / Q5_K / Q6_K / Q8_0 / MXFP4 的反量化与 gguf-py **逐位一致**；
3. HF/compressed-tensors 的 MXFP4（E2M1 + E8M0/32）重排成 ggml `block_mxfp4` 是**无损**的；
4. 3D 专家张量（`dims = (ne0, ne1, n_expert)`）的逻辑形状与 slab 偏移。

## 文件

| 文件 | 作用 |
|---|---|
| `quants_ref.py` | NumPy 反量化：Q4_K/Q5_K/Q6_K/Q8_0/MXFP4；HF→GGUF MXFP4 重排 |
| `gguf_reader.py` | 无第三方依赖的 GGUF 解析器；CLI 可列 tensor / 反量化 |
| `make_sample_gguf.py` | 生成带 Q4_K/Q6_K/MXFP4/F32 与 3D 专家的合成 GGUF |
| `selftest.py` | 真实 vocab GGUF + 合成 GGUF + 与 gguf-py 对比 + MXFP4 重排 |

## 运行

```powershell
cd E:\Develop\Deepseek\tinyinfer
.\.venv\Scripts\python.exe scripts\prototypes\gguf\selftest.py

# 解析真实 GGUF（需要 llama.cpp 仓库里的 vocab 文件，或设置 LLAMA_CPP_DIR）
.\.venv\Scripts\python.exe scripts\prototypes\gguf\gguf_reader.py ..\..\repositories\llama.cpp\models\ggml-vocab-deepseek-llm.gguf --tensors

# 生成合成样本并反量化
.\.venv\Scripts\python.exe scripts\prototypes\gguf\make_sample_gguf.py --out sample.gguf
.\.venv\Scripts\python.exe scripts\prototypes\gguf\gguf_reader.py sample.gguf --dequant blk.0.ffn_gate_exps.weight
```

如果 llama.cpp 不在默认位置：

```powershell
$env:LLAMA_CPP_DIR = 'D:\path\to\llama.cpp'
```

## 已验证的输出（2026-10-21）

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

## 关键约定

- **dims 顺序**：GGUF 的 `ne0` 是最内层；`[ne0, ne1]` 的逻辑形状是 `(ne1, ne0)`；3D 专家是 `(n_expert, ne1, ne0)`。
- **tensor offset**：相对 data 段起点；每个 tensor 不一定按 alignment 对齐（llama.cpp writer 会 pad，reader 不能假设）。
- **MXFP4**：GGUF 每个 block 是 `1 字节 E8M0 + 16 字节 code`；低 nibble 是元素 `j`，高 nibble 是元素 `j+16`；HF 是“偶数列低、奇数列高”。重排在 adapter 层做一次，不要写进 dequant 分支。
- **Q4_K/Q5_K**：scale 和 min 各 8 个、6-bit；Q5_K 多一个 high bit；Q6_K 的 q1..q4 是每 32 个一组连续排列（不是交错）。
- **Q4_K/Q5_K/Q6_K 的 d/dmin**：本原型生成样本时固定为 1.0/0.0，避免随机 fp16 出现 NaN；真实权重的 d 是正常 fp16。
- **IQ 类型**：header 解析支持 IQ2_XXS/IQ2_XS/IQ3_XXS/IQ1_S/IQ4_NL/IQ3_S/IQ2_S/IQ4_XS/IQ1_M；反量化只实现了本报告对照的 5 种（Q4_K/Q5_K/Q6_K/Q8_0/MXFP4）。

## 不做什么

- 不下载模型权重；只解析仓库里的 vocab GGUF 和合成文件。
- 不实现 tokenizer BPE；tokenizer 对齐建议用 HF 作为权威（见 `docs/llama_ref.md` 2.5）。
- 不实现 mmap/O_DIRECT 读取；这部分接 `tinyinfer.loading.ShardReader`，且 O_DIRECT 的对齐/回退见 `docs/llama_ref.md` 6.2。
- 不实现 Q4_K_M 选择器；Q4_K_M 是文件级混合配方，不是 tensor 类型（见 `docs/llama_ref.md` 3.3）。

## 接到 ExpertStore 的最小改动

1. `GGUFIndex` 提供 `expert_slab(layer, expert) -> (file, offset, nbytes)`；
2. `ShardReader` 增加 `read_slab`，复用现有 direct/buffered 路径；
3. `ExpertStore._materialize` 按 `index.kind` 选 HF 多段拼 blob 或 GGUF 单段 slab；
4. MXFP4 的 `uint8` 直接进 `quant.dequant_fp4_block`（语义相同，nibble 顺序在 adapter 层已转换）；
5. 用同一专家的 HF/GGUF 双读做逐位回归。

更完整的背景、对比与风险见 `docs/llama_ref.md`。
