# loading_research：分层加载/专家预取原型实验

本目录是 `tinyinfer/docs/loading.md` 的实验脚本，不参与推理主路径。

| 脚本 | 作用 | 是否需要权重 |
|---|---|---|
| `sim_router.py` | 用真实 router trace（若有）或合成 Zipf+topic+temporal 分布，评估 LRU/ARC/S3-FIFO/pin 命中率与预取净收益 | 否 |
| `collect_router_trace.py` | 跑 tinyinfer 前向，记录每层 router top-k / gate logits / 子层 hidden，存 npz | 是（全部 40 层分片） |
| `train_lookahead.py` | 从 trace 拟合 hidden(L) -> gate logits(L+1) 的线性预测器，评估 precision@k / recall@k | 否（只用 trace） |
| `bench_io.py` | 用 `tinyinfer.loading.ShardReader` 在真实分片上测 buffered/O_DIRECT 读取 | 是（分片已下载部分即可） |

## 运行

```powershell
cd E:\Develop\Deepseek\tinyinfer
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/sim_router.py --out sim.md
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/bench_io.py --layer 0 --experts 6
```

`collect_router_trace.py` 需要完整权重；权重齐了以后：

```powershell
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/collect_router_trace.py `
    --model-dir cache/weights --layers 0-39 --prompt-file deepseek/prompt.txt --out router_trace.npz
.\.venv\Scripts\python.exe scripts/prototypes/loading_research/train_lookahead.py `
    --trace router_trace.npz --out lookahead.npz --lead 1
```

## 结论口径

- 命中率实验的分布参数是**先验**（Zipf s、topic 复用、temporal 复用 25%），不是本模型实测；真实 trace 到位后用 `--trace` 覆盖。
- 预取收益模型：每层 GPU compute（默认 3 ms）期间 NVMe 空闲，可免费读 ~26 MB；预取 k=2 个专家 37.6 MB 会溢出 ~11 MB；净加速由命中数与溢出共同决定，脚本输出 0.4–0.8 precision 的扫描。
