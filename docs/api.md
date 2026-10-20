# tinyinfer OpenAI 兼容 API

tinyinfer 的对外服务层：把后端（MockBackend / TinyModelBackend / 后续真正的引擎）包装成
**OpenAI 兼容的 HTTP API**，供一个或几个 agent 并发使用。后端只负责"生成 token"，
服务层负责协议、SSE、usage、错误格式、request id、API key、限流、超时和会话/槽位。

- 代码：`tinyinfer/src/tinyinfer/server.py`（FastAPI 应用）、`tinyinfer/src/tinyinfer/cli.py`（serve / generate）
- 测试：`tinyinfer/tests/test_server.py`（40 个用例，MockBackend，无需权重/GPU）
- 依赖：`pyproject.toml` 的 `server` extra（fastapi、uvicorn、httpx）

---

## 1. 安装与启动

在仓库根目录执行（Windows PowerShell）：

```powershell
# 1) 安装服务端依赖（清华镜像；已装过可跳过）
$env:UV_CACHE_DIR='E:\Develop\Deepseek\deepseek\uvcache'
uv pip install --python .\tinyinfer\.venv\Scripts\python.exe `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple `
  -e .\tinyinfer[server]

# 2) 启动（Mock 后端，不需要模型权重）
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.server --backend mock --port 8000

# 或者走 CLI（等价）
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.cli serve --backend mock --port 8000

# 3) 健康检查
.\tinyinfer\.venv\Scripts\python.exe -c "import httpx; print(httpx.get('http://127.0.0.1:8000/health').json())"
```

如果当前环境没有把 `src` 加入路径，可以加 `PYTHONPATH`，或像上面一样做一次
`uv pip install -e`（只写 venv，不污染仓库）：

```powershell
$env:PYTHONPATH = "E:\Develop\Deepseek\tinyinfer\src"
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.server --help
```

服务默认监听 `127.0.0.1:8000`，交互式文档在 `http://127.0.0.1:8000/docs`，
OpenAPI 在 `/openapi.json`。启动参数（`--help` 可看全量）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host / --port` | `127.0.0.1 / 8000` | 监听地址 |
| `--backend` | `mock` | `mock` 或 `tiny` |
| `--model-dir` | – | HF safetensors 目录（`--backend tiny` 必填） |
| `--layers` | 全部 40 层 | 逗号分隔的层号，做分段验证时用，如 `--layers 0,1` |
| `--device` | `auto` | 预留；当前 TinyModelBackend 是 CPU 参考路径 |
| `--api-key` | 无 | 逗号分隔多个 key；也支持环境变量 `TINYINFER_API_KEY` |
| `--max-concurrency` | `4` | 同时在跑的 HTTP 请求数，超过返回 429 |
| `--max-slots` | `4` | 会话槽位数（KV/专家缓存的归属单位） |
| `--max-context-tokens` | `1048576` (1M) | 上下文上限 |
| `--max-tokens-limit` | `1048576` | 单次输出上限 |
| `--default-max-tokens` | `256` | 请求没带 max_tokens 时的默认值 |
| `--fit-max-tokens` | 关 | 超上下文时把 max_tokens 压到剩余空间，而不是 400 |
| `--request-timeout` | `600` | 单请求超时（秒），超时 504（流式中为 error 事件） |
| `--sse-keepalive` | `15` | 流式 keep-alive 注释间隔（秒），0=关闭 |
| `--log-level` | `info` | `debug/info/warning/error` |

---

## 2. 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活、模型、ready、槽位、并发（无需 API key） |
| GET | `/v1/models` | 列出模型（含别名、上下文/输出上限） |
| GET | `/v1/models/{id}` | 单个模型元数据；未知 404 |
| POST | `/v1/completions` | 文本补全（流式/非流式） |
| POST | `/v1/chat/completions` | 聊天补全（流式/非流式，tools 占位） |
| GET | `/v1/slots`、`/slots` | 槽位/会话状态（llama.cpp 风格字段） |
| POST | `/v1/messages` | Anthropic Messages 预留接口（当前 501） |
| GET | `/docs`、`/openapi.json` | 交互文档 / OpenAPI |

---

## 3. 认证与 request id

- 配置了 `--api-key` 后，所有 `/v1/*` 与 `/slots` 都要求 `Authorization: Bearer <key>`；
  `/health` 保持公开。多个 key 用逗号分隔，任一匹配即可。
- 每个响应都带 `x-request-id`（请求头传了 `x-request-id` 会原样回显，否则生成 `req_<hex>`）。
- 流式响应同样是先回 header，再回 SSE 数据。

```powershell
curl.exe http://127.0.0.1:8000/v1/models -H "Authorization: Bearer sk-demo"
```

---

## 4. Completions

### 4.1 非流式（curl）

```powershell
curl.exe http://127.0.0.1:8000/v1/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"tinyinfer-mock","prompt":"hello tiny world","max_tokens":8,"temperature":0,"seed":7}'
```

响应（节选）：

```json
{
  "id": "cmpl-4f8c...",
  "object": "text_completion",
  "created": 1792512960,
  "model": "tinyinfer-mock",
  "choices": [
    {"index": 0, "text": "echo: hello tiny world edge memory", "logprobs": null, "finish_reason": "length"}
  ],
  "usage": {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11}
}
```

支持的请求字段：
`prompt`（字符串；列表仅允许单元素）、`model`（任意名字都接受，响应原样回显）、
`max_tokens`、`temperature` (0–2)、`top_p` (0–1]、`top_k`（扩展）、`stop`（字符串或最多 4 个字符串）、
`seed`、`stream`、`stream_options.include_usage`、`session_id`（扩展）。
`n` 只支持 1；`echo`、`logprobs`、`suffix` 会被忽略（保持请求可解析）。

### 4.2 流式（SSE）

```powershell
curl.exe -N http://127.0.0.1:8000/v1/completions `
  -H "Content-Type: application/json" `
  -d '{"prompt":"stream me","max_tokens":6,"stream":true,"stream_options":{"include_usage":true}}'
```

```text
data: {"id":"cmpl-...","object":"text_completion","created":...,"model":"tinyinfer-mock","choices":[{"index":0,"text":"","logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion",...,"choices":[{"index":0,"text":"echo:","logprobs":null,"finish_reason":null}]}

: keep-alive

data: {"id":"cmpl-...","object":"text_completion",...,"choices":[{"index":0,"text":"","logprobs":null,"finish_reason":"length"}]}

data: {"id":"cmpl-...","object":"text_completion",...,"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":6,"total_tokens":8}}

data: [DONE]
```

- 一行一个 `data:` 帧，空行分隔；结束固定 `data: [DONE]`。
- 模型读长 prompt 时服务端会发 `: keep-alive` 注释（SSE 注释，客户端可忽略），避免 agent 超时。
- `stream_options.include_usage=true` 时，在 finish 帧之后多一帧 `choices:[]` + `usage`。

---

## 5. Chat Completions

### 5.1 curl

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"tinyinfer-mock","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"write a haiku about GPUs"}],"max_tokens":16,"seed":1}'
```

响应：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1792512960,
  "model": "tinyinfer-mock",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "echo: system be brief ..."},
    "logprobs": null,
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 9, "completion_tokens": 16, "total_tokens": 25}
}
```

`messages` 支持 `system/user/assistant/tool/developer/function`；`content` 可以是字符串，也可以是
`[{"type":"text","text":"..."}]` 文本分片（目前只支持文本分片，图片分片会 400）。
`max_tokens` 与 `max_completion_tokens` 等价（后者是给新客户端用的别名）。

### 5.2 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="tinyinfer-mock",
    messages=[{"role": "user", "content": "hello tinyinfer"}],
    max_tokens=16,
    temperature=0.7,
)
print(resp.choices[0].message.content)
print(resp.usage)
```

流式：

```python
stream = client.chat.completions.create(
    model="tinyinfer-mock",
    messages=[{"role": "user", "content": "stream, please"}],
    max_tokens=32,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.usage:
        print("\nusage:", chunk.usage)
```

### 5.3 原始 httpx（不装 SDK 也能用）

```python
import json, httpx

with httpx.Client(base_url="http://127.0.0.1:8000/v1", timeout=600) as c:
    with c.stream("POST", "/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "stream": True,
    }) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
```

### 5.4 tools 占位

`tools` / `tool_choice` / `parallel_tool_calls` 会被**接受并忽略**：响应仍是普通
`assistant` 文本，不会返回 `tool_calls`，`finish_reason` 也不会是 `tool_calls`。
这是为将来接入结构化输出留的协议位；MockBackend 和 TinyModelBackend 目前都不执行工具。

---

## 6. 会话、槽位与并发（tinyinfer 扩展）

tinyinfer 把"一条对话的 KV cache / 专家缓存"归到一个**槽位（slot）**，HTTP 层只负责把请求
路由到槽位并把元数据暴露出来。这样多个 agent 并发时不会互相污染缓存。

- 请求体里加 `"session_id": "agent-a"`，或请求头 `X-Session-Id: agent-a`；
  同一个 `session_id` 后续请求复用同一个槽位（顺序请求；同一 session 并发会 429 `session_busy`）。
- 不带 `session_id` 的请求拿一个临时槽位，用完即释放（一次性调用，如分类、探测）。
- 槽位会记录最近一次请求的 `n_prompt_tokens` / `n_completion_tokens`，可在 `/v1/slots` 查看。
- 槽位满了会按 LRU 淘汰空闲会话；全部在忙时返回 429 `server_busy`。
- 响应头回显 `x-session-id`，方便客户端绑定。

```powershell
curl.exe http://127.0.0.1:8000/v1/slots
```

```json
{
  "object": "list",
  "total_slots": 4,
  "busy_slots": 0,
  "sessions": 1,
  "slots": [
    {"id_slot": 0, "session_id": "agent-a", "busy": false, "n_ctx": 1048576,
     "n_prompt_tokens": 12, "n_completion_tokens": 8, "requests": 2},
    {"id_slot": 1, "session_id": null, "busy": false, "n_ctx": 1048576,
     "n_prompt_tokens": 0, "n_completion_tokens": 0, "requests": 0}
  ]
}
```

后端约定（`Backend` 协议，见 `server.py`）：

```python
class Backend(Protocol):
    model_id: str
    aliases: tuple[str, ...]
    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]: ...
    def count_prompt_tokens(self, request: GenerationRequest) -> int: ...
    def count_tokens(self, text: str) -> int: ...
    def render_prompt(self, request: GenerationRequest) -> str: ...
    def reset_slot(self, slot_id: int) -> None: ...
    def health(self) -> dict: ...
    def warmup(self) -> None: ...
    def close(self) -> None: ...
```

- `GenerationRequest.slot_id` 就是 KV/专家缓存的归属；`reset_slot` 在新会话占用该槽位时调用，
  后端应丢弃旧缓存（TinyModelBackend 目前只有一份 ring-buffer KV，所以直接 reset 并用锁串行化）。
- 每个 token 通过 `TokenEvent(text=...)` 流式返回；最后一个事件带 `finish_reason`（`"stop"`/`"length"`）
  和 `usage`；服务端在缺失时兜底计算 usage。
- 并发上限由 `max_concurrency` 控制；超过直接 429（带 `Retry-After: 1`），不会排队。

---

## 7. 错误格式

所有错误都是 OpenAI error object：

```json
{
  "error": {
    "message": "Invalid request: body.messages: List should have at least 1 item after validation, not 0",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "invalid_request"
  }
}
```

| HTTP | code（示例） | 场景 |
|---|---|---|
| 400 | `invalid_request` | 参数缺失/非法（温度、n、stop、消息角色、内容分片等） |
| 400 | `context_length_exceeded` | prompt + max_tokens 超出上下文；开启 `--fit-max-tokens` 时会自动压缩 |
| 400 | `invalid_session_id` | session_id 含非法字符 |
| 401 | `invalid_api_key` | 没带/带错 API key（响应含 `WWW-Authenticate: Bearer`） |
| 404 | `not_found` / `model_not_found` | 未知路径 / `/v1/models/{id}` 未知模型 |
| 405 | `method_not_allowed` | 方法不对 |
| 429 | `rate_limit_exceeded` / `server_busy` / `session_busy` | 并发超限 / 槽位全忙 / 同会话并发 |
| 500 | `server_error` / `model_crashed` | 后端内部错误 |
| 501 | `not_implemented_error` | `/v1/messages`（Anthropic 预留） |
| 503 | `model_not_loaded` / `tokenizer_load_failed` | 权重/分词器未就绪 |
| 504 | `timeout` | 超过 `--request-timeout` |

流式请求一旦开始（HTTP 200），中途出错会以 SSE 帧上报，然后 `[DONE]` 收尾：

```text
data: {"error":{"message":"...","type":"timeout_error","param":null,"code":"timeout"}}

data: [DONE]
```

---

## 8. 上下文、token 与限制

- 上下文默认 **1,048,576 token（1M）**，输出上限同样可配到 1M；都由启动参数控制。
- 请求进来先由后端 `count_prompt_tokens` 计数，再检查 `prompt + max_tokens <= max_context_tokens`：
  超了返回 400 `context_length_exceeded`；加 `--fit-max-tokens` 则把 `max_tokens` 压到剩余空间。
- usage 来自后端终态事件；MockBackend 按空白分词估算 prompt tokens，TinyModelBackend 用真实 tokenizer。
- `stop` 最多 4 个；命中后返回 `finish_reason="stop"`，stop 文本本身不返回。
- `n>1` 不支持（返回 400）；多 prompt 列表只支持单元素；`prompt` 为空字符串允许。
- 每个请求的 `temperature/top_p/top_k/seed` 会传给后端；MockBackend 只用 seed（结果确定、可测试）。

---

## 9. TinyModelBackend 接入状态

`TinyModelBackend` 是真正模型的接入点（Lead 后续集成）：

```powershell
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.server `
  --backend tiny --model-dir .\tinyinfer\cache\weights --layers 0,1 --port 8000
```

- 启动时 `warmup()` 加载 `TinyConfig` + `ShardStore` + `TinyModel`（按 `--layers` 组装）和 tokenizer；
  权重缺失/下载未完成时启动失败并打印原因；请求路径上的 TinyModelBackend 错误是 503 `model_not_loaded`。
- 生成走 `TinyModel.forward` 的 PyTorch 参考路径；chat 请求优先用 tokenizer 的 chat template。
- 当前限制：CPU 参考实现、单请求串行、全局一份 KV ring-buffer（`reset_slot` 会清空）。
  真正的 per-slot KV/专家缓存和 CUDA 路径由引擎集成时补上，接口（`slot_id`/`reset_slot`/`TokenEvent`）已经留好。

---

## 10. CLI generate（不启动 HTTP 时自测）

```powershell
# Mock 后端
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.cli generate "hello tinyinfer" --max-tokens 12 --seed 3 --json

# Tiny 后端（需要权重）
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.cli generate "hello" `
  --backend tiny --model-dir .\tinyinfer\cache\weights --layers 0,1 --stream
```

`--json` 输出 `{text, finish_reason, usage, elapsed_s}`；不加时文本打到 stdout、统计打到 stderr。

---

## 11. 测试

```powershell
.\tinyinfer\.venv\Scripts\python.exe -m pytest .\tinyinfer\tests\test_server.py -q
```

覆盖：`/health`、`/v1/models`、completions/chat 的流式与非流式、usage、stop、
上下文/输出上限、超时 504、并发 429、API key 401、session/slot、后端错误 500/503/SSE 错误帧、
keep-alive、Anthropic 占位。全部基于 MockBackend，不需要权重、GPU 或网络。

---

## 12. 兼容性矩阵

| 能力 | 状态 |
|---|---|
| OpenAI `/v1/models`、`/v1/completions`、`/v1/chat/completions` | ✅ 已实现 |
| 流式 SSE + `[DONE]` + keep-alive | ✅ |
| `usage`（prompt/completion/total） | ✅（后端提供，服务端兜底） |
| `temperature/top_p/top_k/seed/stop` | ✅ |
| `max_completion_tokens` 别名、`stream_options.include_usage` | ✅ |
| 多个 API key、request id、统一错误对象 | ✅ |
| session/slot 多会话隔离元数据 | ✅（引擎侧 per-slot 缓存待接入） |
| `tools` / `tool_choice` | ⚠️ 接受但忽略（占位） |
| `n>1`、多 prompt 列表、`logprobs`、`echo`、penalties、图片输入 | ❌ 400 或忽略（见第 8 节） |
| Anthropic `/v1/messages` | ⛔ 501 预留位 |
| OpenAI `/v1/responses` | ⛔ 未实现（预留，可复用同一 Backend） |

---

## 13. 常见问题

- **401 invalid_api_key**：服务端配置了 `--api-key`，客户端要带 `Authorization: Bearer <key>`；
  `/health` 不需要。
- **429 rate_limit_exceeded**：`--max-concurrency` 太小或上次请求未结束；加大它或等 `Retry-After`。
- **429 server_busy / session_busy**：槽位都忙 / 同一个 `session_id` 并发请求；加大 `--max-slots`
  或给不同 agent 用不同 `session_id`。
- **503 model_not_loaded**：`--backend tiny` 但 `--model-dir` 不对、权重没下完，或先启动再下载。
- **504 timeout**：预填/解码超过 `--request-timeout`；给大 prompt 或慢机器调大（默认 600s）。
- **流式没有输出但连接成功**：看 `/health` 的 `ready` 和后端日志；长 prompt 期间只会有
  `: keep-alive` 注释（属正常）。
