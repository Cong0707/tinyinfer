# 示例

先启动引擎，模型由启动参数决定：

```powershell
.\start.ps1 -Model "E:\Models\Qwen3.5-0.8B" -Bg
```

## 通过 HTTP（推荐）

```powershell
.\venv\Scripts\python.exe repl.py "你的问题"       # 单次
.\venv\Scripts\python.exe repl.py                  # 交互
```

或直接用标准库调 OpenAI 兼容接口：

```powershell
python examples\openai_client.py --prompt "Once upon a time"
```

这个脚本只用标准库，任意 Python 都能跑。

## 直接调引擎（进程内，跳过 HTTP）

需要 `PYTHONPATH=src`：

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe examples\generate.py --model "E:\Models\Qwen3.5-0.8B"
.\venv\Scripts\python.exe examples\stream.py   --model "E:\Models\Qwen3.5-0.8B" --prompt "Once upon a time"
```

| 文件 | 用途 |
| --- | --- |
| generate.py | 最小端到端：加载 + 生成 |
| stream.py | 边生成边打印 |
| multi_turn.py | 多轮且不重复 prefill（存 KV 快照复用） |
| tiers.py | 同一 prompt 在不同优化级别下的对比 |
| openai_client.py | 走 HTTP，只用标准库 |

注意：这些示例走引擎的 `Engine` 接口，主要面向 DeepSeek。Qwen 走 server 的
`QwenBackend`，用上面的 `repl.py`。
