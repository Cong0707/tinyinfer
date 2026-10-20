# tinyinfer OpenAI 兼容 API

tinyinfer 的对外服务层：把后端（MockBackend / TinyModelBackend / 后续真正的引擎）包装成
**OpenAI 兼容的 HTTP API**，供一个或几个 agent 并发使用。后端只负责"生成 token"，
服务层负责协议、SSE、usage、错误格式、request id、API key、限流、超时和会话/槽位。

- 代码：`tinyinfer/src/tinyinfer/server.py`（FastAPI 应用）、`tinyinfer/src/tinyinfer/cli.py`（serve / generate）
- 测试：`tinyinfer/tests/test_server.py`（52 个用例；默认全 Mock、无需权重/GPU，另有一个 `TINYINFER_L0=1` 才跑的真实 L0 权重端到端用例）
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
| `--model-dir` | – | 默认模型的 HF safetensors 目录（`--backend tiny` 时与 `--extra-model` 至少给一个） |
| `--extra-model NAME=DIR` | 无 | 注册额外模型，可重复；多模型时按请求的 `model` 路由 |
| `--default-model` | 第一个 | 客户端传了未注册模型名时落到哪个模型 |
| `--layers` | 全部 40 层 | 逗号分隔的层号，做分段验证时用，如 `--layers 0`（L0 端到端） |
| `--device` | `auto` | 预留；当前 TinyModelBackend 是 CPU 参考路径 |
| `--adapter` | `auto` | `auto` / `hf_deepseek_v41` / `gguf`（GGUF 为预留，返回 501） |
| `--tier` | `conservative` | `conservative`（纯 PyTorch 参考）/ `balanced`（ExpertStore 分层专家缓存 + tilelang 可用时优先） |
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
| GET | `/v1/models` | 动态列出已配置模型：id、adapter、tier、layers、loaded、supported、context_length 等 |
| GET | `/v1/models/{id}` | 单个模型元数据；未知 404 |
| POST | `/v1/completions` | 文本补全（流式/非流式） |
| POST | `/v1/chat/completions` | 聊天补全（流式/非流式，最小 function calling 回显） |
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

### 5.4 最小 function calling（tools 回显）

`tools` / `tool_choice` 是**最小实现**：不执行工具，但能从模型输出里识别一次函数调用并回显成
OpenAI 的 `tool_calls`，让 agent 客户端拿到可解析的结果。识别格式（任一）：

- 纯 JSON：{"name": "lookup", "arguments": {"q": "x"}}
- 代码块：json 围栏包裹的同一 JSON
- 包装：<tool_call>{...}</tool_call>

规则：

- 只有请求里声明了 `tools` 且模型写出了合法 JSON、名字在 `tools[].function.name` 中时才识别；
  名字不匹配或 JSON 不合法时，原文按普通 `content` 返回，不会误吞。
- 非流式：`message.content` 为 null，`message.tool_calls=[{id,type:"function",function:{name,arguments}}]`，
  `finish_reason="tool_calls"`；arguments 是 JSON 字符串。
- 流式：声明了 tools 时服务端会**缓冲**完整输出（因为要检测完整 JSON），生成结束后一次性发
  tool_calls delta 与 finish 帧；未命中则把文本作为单个 content delta 发出（此场景不逐 token 流式）。
- `tool_choice: "none"` 会关闭识别；`parallel_tool_calls` 目前不处理（只回显一个调用）。

示例：

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"find tinyinfer"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"max_tokens":64}'
```

```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_ab12...",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{\"q\": \"tinyinfer\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20}
}
```

注：这是协议层回显，模型本身没有工具执行能力；真正的结构化输出/工具循环留给上层 agent。

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

- `GenerationRequest.slot_id` 就是 KV/专家缓存的归属。TinyModelBackend 为每个 slot 保存/恢复
  一份每层 MLA ring-buffer KV 快照（`_bind_slot` / `_stash_slot`），新会话占用槽位时 `reset_slot` 丢掉该槽
  的 KV 并清理有界专家缓存；不同槽位互不覆盖。OpenAI 语义是"每次请求带完整对话"，所以请求仍从 token 0
  重新 prefill；槽位快照是隔离边界与将来 resume-from-prefix 的挂点，不是跨请求复用。
- 专家缓存是模型级共享（只读权重），按 tier 决定：`conservative` 直接读 shard 并只保留最近 N 个
  dequantized 专家（`max_cached_experts`，默认 64），`balanced` 走 `tinyinfer.loading.ExpertStore`
  的分层 RAM 缓存。
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
| 501 | `adapter_not_implemented` / `adapter_not_supported` | GGUF adapter（预留）或其他未实现的 checkpoint 布局 |
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

`TinyModelBackend` 已接真实 `tinyinfer.model.TinyModel`，checkpoint 布局由
`tinyinfer.adapters.open_adapter()` 选择（当前实现了 HF deepseek_v41，GGUF 预留）。L0 单层真实权重可以完整跑通：

```powershell
# L0 单层（cache/weights 里已有 embed / L0 / head 分片，无需新下载）
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.server `
  --backend tiny --model-dir .\tinyinfer\cache\weights --layers 0 `
  --model-id deepseek-v41-l0 --tier conservative --port 8000

# balanced：ExpertStore 分层专家缓存
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.cli serve `
  --backend tiny --model-dir .\tinyinfer\cache\weights --layers 0 --tier balanced
```

- 启动 `warmup()`：`open_adapter` 检测布局 → `TinyConfig` → `ShardStore` → `TinyModel(layers)` → tokenizer；
  权重缺失/分片未下完是 503 `model_not_loaded`；GGUF 是 501 `adapter_not_implemented`。
- adapter：`hf_deepseek_v41` 已实现（adapter 名 `deepseek_v41_hf`，HF index + safetensors 直接加载）；
  GGUF 走 `tinyinfer.adapters.open_adapter` 的预留分支，server 层已把错误映射成 501 并在 `/v1/models` 标 `supported: false`。
- tiers（`--tier`）：`conservative` = 纯 PyTorch 参考路径；`balanced` = 额外挂
  `tinyinfer.loading.ExpertStore`（分层 RAM 专家缓存），attention 在 tilelang 可构建时优先、失败自动回退参考实现；
  `aggressive`/`extreme` 暂未接入参考路径，启动时明确报错而不是静默降级。
- slots：见第 6 节。每请求保存/恢复该槽位的 KV ring-buffer 快照，新会话 `reset_slot` 清空；专家缓存模型级共享。
- 当前限制：CPU 参考实现、同一模型内生成串行（`_gen_lock`）；per-slot KV 只是快照隔离，跨请求仍从
  token 0 重算（OpenAI 请求本来就带完整对话）。CUDA/真正的 prefix cache 留给引擎集成。
- L0 实测数字见第 15 节。

---

## 10. CLI generate（不启动 HTTP 时自测）

```powershell
# Mock 后端
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.cli generate "hello tinyinfer" --max-tokens 12 --seed 3 --json

# Tiny 后端：L0 单层真实权重（conservative / balanced）
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.cli generate "hello" `
  --backend tiny --model-dir .\tinyinfer\cache\weights --layers 0 --max-tokens 2 --json
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.cli generate "hello" `
  --backend tiny --model-dir .\tinyinfer\cache\weights --layers 0 --tier balanced --stream

# 多模型：默认模型 + 额外注册一个（请求里用 model 选择，未知名字落到 --default-model）
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.cli serve `
  --backend tiny --model-dir DIR_L0 --layers 0 --model-id l0 `
  --extra-model gguf-future=DIR_GGUF --default-model l0
```

`--json` 输出 `{text, finish_reason, usage, elapsed_s}`；不加时文本打到 stdout、统计打到 stderr。

---

## 11. 测试

```powershell
# 默认：52 个用例里 51 个 Mock 全链路 + 1 个真实 L0 用例自动 skip
.\tinyinfer\.venv\Scripts\python.exe -m pytest .\tinyinfer\tests\test_server.py -q

# 打开真实 L0 权重用例（需要 cache/weights 里已有 embed/L0/head 分片，约 13s）
$env:TINYINFER_L0='1'
.\tinyinfer\.venv\Scripts\python.exe -m pytest .\tinyinfer\tests\test_server.py -q -k l0
```

覆盖：`/health`、`/v1/models`（含 adapter/tier/loaded/supported）、completions/chat 的流式与非流式、
usage、stop、上下文/输出上限（per-model 1M）、超时 504、并发 429、API key 401、session/slot、
TinyModelBackend 的 slot KV 归属、multi-model 路由、GGUF 预留 501、最小 function calling、
后端错误 500/503/SSE 错误帧、keep-alive、Anthropic 占位；L0 用例额外验证真实权重加载、chat 非流式、
completions SSE 与 /v1/slots。

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
| session/slot：per-slot KV 快照 + reset_slot | ✅（TinyModelBackend；跨请求 prefix 复用待引擎） |
| 多模型路由、动态 /v1/models（adapter/tier/loaded/supported） | ✅ |
| HF deepseek_v41 adapter（open_adapter） | ✅；GGUF ⏳ 预留（501 adapter_not_implemented） |
| tiers：conservative / balanced（ExpertStore） | ✅；aggressive/extreme ⏳ 未接入参考路径 |
| `tools` / `tool_choice` 最小 function calling 回显 | ✅（缓冲检测 JSON；不执行工具、不流式逐字输出） |
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
- **501 adapter_not_implemented**：checkpoint 是 GGUF（预留）或布局无法识别；换成 HF safetensors
  （目录里需要 `config.json` + `model.safetensors.index.json`），或查看 `/v1/models` 里的 `supported/note`。
- **balanced tier 日志出现 ExpertStore 回退**：说明分层专家缓存建不起来（缺少 loading 需要的文件等），
  会自动退回直接 shard 读取，结果正确但慢；检查 `tinyinfer.loading.ShardIndex` 能否读到分片 header。
- **L0 启动偏慢**：第一次加载 torch + transformers tokenizer + 3GB 左右的 embed/head/attn 权重，约 10-15s；
  之后每个请求的 prefill/decode 取决于 prompt 长度和首次专家加载（见第 15 节）。
- **tools 场景流式不逐字**：声明 `tools` 时服务端需要完整文本才能判断 JSON 函数调用，因此缓冲到结束再发；
  不声明 tools 时仍逐 token 流式。

---

## 14. 多模型、适配器与 tiers

### 14.1 配置多个模型

没有新的路由端点：同一个 `create_app(backend)` 里，若后端是 `MultiModelBackend`，请求按 `model` 字段路由。

```powershell
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.server `
  --backend tiny `
  --model-dir .\tinyinfer\cache\weights --layers 0 --model-id deepseek-v41-l0 `
  --extra-model gguf-future=.\deepseek\future-gguf `
  --tier conservative --default-model deepseek-v41-l0
```

路由规则（对 agent 友好）：

- `model` 精确匹配注册 id 或其 `aliases` → 用该模型的 backend（按需懒加载）；
- 未知名字（如 agent 配置里的 `gpt-4o` / `deepseek-chat`）→ 落到 `--default-model`，响应里的 `model` 仍回显请求名；
- `GET /v1/models` 动态列出所有注册项：`id、adapter、tier、layers、loaded、supported、context_length` 等。

```json
{
  "object": "list",
  "data": [
    {
      "id": "deepseek-v41-l0",
      "object": "model",
      "created": 1792512960,
      "owned_by": "tinyinfer",
      "context_length": 1048576,
      "max_output_tokens": 1048576,
      "adapter": "deepseek_v41_hf",
      "tier": "conservative",
      "layers": [0],
      "model_dir": "E:\\Develop\\Deepseek\\tinyinfer\\cache\\weights",
      "loaded": true,
      "supported": true
    },
    {
      "id": "gguf-future",
      "adapter": "gguf",
      "loaded": false,
      "supported": false,
      "note": "GGUF adapter is reserved (see tinyinfer.adapters.open_adapter)"
    }
  ]
}
```

### 14.2 适配器

| adapter | 状态 | 说明 |
|---|---|---|
| `hf_deepseek_v41`（auto 检测到 `model.safetensors.index.json`） | ✅ | 走 `tinyinfer.adapters.HFDeepSeekV41Adapter`，adapter 名 `deepseek_v41_hf` |
| `gguf`（目录里出现 `*.gguf`） | ⏳ 预留 | `open_adapter` 抛 NotImplementedError → server 映射 501；`/v1/models` 标 `supported:false` + note |
| 其他 HF 架构 | ⏳ | 需要先在 `tinyinfer.adapters` 注册新 adapter |

### 14.3 tiers（`--tier conservative|balanced`）

| tier | 当前行为 | 用途 |
|---|---|---|
| `conservative` | 纯 PyTorch 参考路径；专家直接读 shard，每层只保留最近 `max_cached_experts`（默认 64）个 dequantized 专家 | 正确性基线、L0 验证 |
| `balanced` | 额外构造 `tinyinfer.loading.ExpertStore`（分层 RAM 专家缓存，默认按 `tiers.plan` 的 RAM 预算）；attention 在 tilelang 可构建时优先、失败自动回退 | 更接近最终内存/IO 策略 |
| `aggressive` / `extreme` | 未接入参考路径，构造时明确报错 | 留给 kernel/量化 KV/投机阶段 |

```python
from tinyinfer.server import TinyModelBackend

backend = TinyModelBackend(
    model_dir=r".\tinyinfer\cache\weights",
    layers=[0],
    tier="balanced",
    model_id="deepseek-v41-l0",
)
backend.warmup()
print(backend.health())  # adapter / tier / loaded / slot_caches / expert_store ...
```

---

## 15. L0 单层真实权重端到端（实测）

2026-10-21，RTX 5060 Laptop（8GB，sm_120）+ Ryzen 9 9955HX + 47GB RAM，纯 Windows，CPU 参考路径。
权重只用了 `tinyinfer/cache/weights` 里已有的分片（**没有下载新权重**）：

| 分片 | 内容 | 大小 |
|---|---|---|
| `model-00002-of-00048.safetensors` | embed | 1.32 GB |
| `model-00003-of-00048.safetensors` | L0（2334 个张量：attention/HC/shared expert/gate/384×fp4 experts） | 7.39 GB |
| `model-00043-of-00048.safetensors` | norm + head | 1.32 GB |

启动：`python -m tinyinfer.server --backend tiny --model-dir tinyinfer\cache\weights --layers 0 --model-id deepseek-v41-l0 --tier conservative`

实测（首请求会按 token 路由加载专家，故第一个非流式请求最慢）：

| 客户端 / 场景 | 结果 |
|---|---|
| OpenAI Python SDK `client.models.list()` | 0.07s，id=`deepseek-v41-l0`，context_length=1048576 |
| OpenAI SDK chat 非流式，max_tokens=3 | 5.91s（冷专家），finish=length，usage 10/3/13，text=`medi crowdrende` |
| OpenAI SDK chat SSE（include_usage） | 0.36s（专家已缓存），6 个 chunk，finish=length，usage 10/3/13 |
| httpx 非流式 chat，max_tokens=3 | 0.98s，200，finish=length，usage 10/3/13 |
| httpx 原始 SSE | 1.03s，6 个 chunk，收到 `[DONE]`，usage 10/3/13 |
| `GET /v1/slots` | 两个 session（`sdk-e2e`、`http-e2e`）各 2 个请求，slot 0/1 |

更早的单模型测量（同一台机器、prefill 4 token）：模型加载 <1s（mmap），tokenizer 首次 8.3s；
prefill 3.7s，decode 一步 0.74s；prefill 后专家缓存 20 个。balanced tier 预热 5.4s，
2 token 生成 1.29s，ExpertStore 生效（`expert_store: true`）。

复现（终端 1 启动，终端 2 跑客户端）：

```powershell
# 终端 1
.\tinyinfer\.venv\Scripts\python.exe -m tinyinfer.server `
  --backend tiny --model-dir .\tinyinfer\cache\weights --layers 0 `
  --model-id deepseek-v41-l0 --port 8125
```

```python
from openai import OpenAI
import httpx, json

client = OpenAI(base_url="http://127.0.0.1:8125/v1", api_key="dummy", timeout=300)
r = client.chat.completions.create(
    model="deepseek-v41-l0",
    messages=[{"role": "user", "content": "hello tinyinfer"}],
    max_tokens=3, temperature=0, seed=7,
)
print(r.choices[0].message.content, r.usage)

with httpx.Client(base_url="http://127.0.0.1:8125", timeout=300) as http:
    with http.stream("POST", "/v1/chat/completions", json={
        "model": "deepseek-v41-l0",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2, "stream": True,
    }) as stream:
        for line in stream.iter_lines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                chunk = json.loads(line[6:])
                print(chunk["choices"][0]["delta"].get("content", ""), end="")
```

注意：只加载 L0 时模型没有完整语义（输出是单层前向的 token，只为验证链路正确性）；
L0 也没有经过训练后的 40 层组合，不代表模型质量。真正语义需要全量分片（仍在下载/分层执行阶段）。
