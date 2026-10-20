# loading_research：分层加载/专家预取原型实验

本目录是 `tinyinfer/docs/loading.md` 的实验脚本，不参与推理主路径。

| 脚本 | 作用 | 是否需要权重 |
|---|---|---|
| `sim_router.py` | 用真实 router trace（若有）或合成 Zipf+topic+temporal 分布，评估 LRU/ARC/S3-FIFO/pin 命中率与预取净收益 | 否 |
| `collect_router_trace.py` | 跑 40 层前向记录每步每层 hidden / gate logits / routes，存 npz（E-L3） | 是（全部 40 层分片；`--smoke` 只需 layer 0） |
| `train_lookahead.py` | 从 trace 拟合 hidden(L) -> gate logits(L+lead)，评估 precision@k / recall@k | 否（只用 trace） |
| `bench_io.py` | 用 `tinyinfer.loading.ShardReader` 在真实分片上测 buffered/O_DIRECT 读取 | 是（分片已下载部分即可） |
| `bench_head.py` | head 放 CPU/GPU/fp8 的实测（CPU bf16 37.8–40.8 ms vs H2D bf16 46.3 ms） | 否（随机同形状张量） |
| `precheck_layer0.py` | 仅需 layer 0 分片即可采集真实 router 分布/相邻重叠/LRU 命中 | 是（layer 0 + embed） |

## 运行

```powershell
cd E:\Develop\Deepseek\tinyinfer
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/sim_router.py --out sim.md
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/bench_io.py --layer 0 --experts 6
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/bench_head.py --threads 16
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/precheck_layer0.py --max-tokens 160
```

规划工具（loading.py 自带）：

```powershell
.\.venv\Scripts\python.exe -m tinyinfer.loading prefill --tokens 1024,8192,32768
.\.venv\Scripts\python.exe -m tinyinfer.loading vram
```

`collect_router_trace.py` 冒烟（layer 0、2 步、约 3 秒）与全量运行：

```powershell
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/collect_router_trace.py --smoke
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/collect_router_trace.py `
    --model-dir cache/weights --layers 0-39 --max-steps 32 --prompt "Hello" --out router_trace.npz
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/train_lookahead.py `
    --trace router_trace.npz --out lookahead.npz --lead 1
```

## 结论口径

- 命中率实验的分布参数是**先验**（Zipf s、topic 复用、temporal 复用 25%），不是本模型实测；真实 trace 到位后用 `--trace` 覆盖。
- 预取收益模型：每层 GPU compute（默认 3 ms）期间 NVMe 空闲，可免费读 ~26 MB；预取 k=2 个专家 37.6 MB 会溢出 ~11 MB；净加速由命中数与溢出共同决定，脚本输出 0.4–0.8 precision 的扫描。
- layer-0 真实预检（2026-10-21，159 token）：232/384 专家、top-24 41.9%、单层 LRU(27) 34.4%、相邻重叠 0.85/6；合成 20 GB ARC 24.7% 可当保守下限。全 40 层需等分片。
- layer-major prefill 规划（1K/8K/32K prompt）：专家 I/O 固定 288.78 GB，@8.8GB/s = 32.9 s；吞吐 31/249/994 tok/s；hidden 0.04/0.34/1.34 GB；expert wave 0.23 GB；attention scratch 83.9 MB。
- 24 个 VRAM 专家槽 = 451 MB：decode 6+6、prefill 12+12；12 槽是无双缓冲底线，16 槽（8+8）可用，24 槽最稳。
- head：默认 CPU bf16（37.8–40.8 ms）；fp8 0.66 GB 在 VRAM 常驻需挤掉 4 层注意力（~26 ms）+ per-tensor fp8 GEMM 2.0 ms ≈ 28 ms，质量风险，8 GB 卡不划算；rowwise fp8 GEMM 在 sm_120 不支持。
