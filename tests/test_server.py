"""Tests for the OpenAI-compatible HTTP layer (tinyinfer.server).

Everything runs against MockBackend through fastapi.testclient, so the suite
needs no model weights, GPU or network.  It covers: model list, health,
completions (stream and non-stream), chat (stream and non-stream), usage,
stop sequences, sessions/slots, limits, errors, API keys and keep-alives.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import model_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from tinyinfer.server import (
    APIError,
    BackendError,
    BaseBackend,
    MockBackend,
    ModelEntry,
    ModelSpec,
    MultiModelBackend,
    SlotManager,
    TinyModelBackend,
    build_backend,
    create_app,
    detect_adapter_kind,
    resolve_tier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client(**kwargs) -> TestClient:
    kwargs.setdefault("request_timeout", 10.0)
    return TestClient(create_app(MockBackend(), **kwargs))


@pytest.fixture()
def client() -> TestClient:
    return make_client()


def sse_payloads(text: str) -> list[dict]:
    payloads: list[dict] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    return payloads


def chat_text(chunks: list[dict]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        parts.append(delta.get("content") or "")
    return "".join(parts)


def completion_text(chunks: list[dict]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if choices:
            parts.append(choices[0].get("text") or "")
    return "".join(parts)


class FailingBackend(BaseBackend):
    model_id = "failing"
    aliases = ()

    def count_prompt_tokens(self, request) -> int:
        return 3

    def generate(self, request):
        raise BackendError("model exploded", status_code=500, code="model_crashed")
        yield  # pragma: no cover

    def health(self) -> dict:
        return {"ready": False, "backend": "FailingBackend"}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["model"] == "tinyinfer-mock"
    assert body["max_context_tokens"] == 1_048_576
    assert body["slots"]["total"] == 4
    assert body["backend"]["backend"] == "MockBackend"


def test_models(client: TestClient):
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    entries = {entry["id"]: entry for entry in body["data"]}
    assert "tinyinfer-mock" in entries
    assert entries["tinyinfer-mock"]["object"] == "model"
    assert entries["tinyinfer-mock"]["context_length"] == 1_048_576
    assert entries["tinyinfer-mock"]["owned_by"] == "tinyinfer"


def test_model_metadata_route(client: TestClient):
    assert client.get("/v1/models/tinyinfer-mock").status_code == 200
    missing = client.get("/v1/models/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"


def test_slots_endpoint(client: TestClient):
    response = client.get("/v1/slots")
    assert response.status_code == 200
    body = response.json()
    assert body["total_slots"] == 4
    assert len(body["slots"]) == 4
    assert client.get("/slots").status_code == 200


def test_openapi_docs(client: TestClient):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_anthropic_interface_reserved(client: TestClient):
    response = client.post("/v1/messages", json={"model": "x", "messages": []})
    assert response.status_code == 501
    assert response.json()["error"]["type"] == "not_implemented_error"


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------


def test_completions_non_streaming(client: TestClient):
    response = client.post(
        "/v1/completions",
        json={"prompt": "hello tiny world", "max_tokens": 5, "seed": 7, "temperature": 0},
    )
    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["text"].startswith("echo: hello")
    assert choice["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 5
    assert body["usage"]["prompt_tokens"] == 3
    assert body["usage"]["total_tokens"] == 8
    assert body["id"].startswith("cmpl-")
    assert body["object"] == "text_completion"
    assert response.headers["x-request-id"].startswith("req_")


def test_completions_seed_is_deterministic(client: TestClient):
    request = {"prompt": "same prompt", "max_tokens": 8, "seed": 42}
    first = client.post("/v1/completions", json=request).json()["choices"][0]["text"]
    second = client.post("/v1/completions", json=request).json()["choices"][0]["text"]
    assert first == second


def test_completions_accepts_unknown_model_name(client: TestClient):
    response = client.post(
        "/v1/completions",
        json={"model": "some-agent-model", "prompt": "hi", "max_tokens": 2},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "some-agent-model"


def test_completions_streaming_matches_non_streaming(client: TestClient):
    request = {"prompt": "stream me", "max_tokens": 6, "seed": 3}
    plain = client.post("/v1/completions", json=request).json()["choices"][0]["text"]

    response = client.post("/v1/completions", json={**request, "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.rstrip().endswith("data: [DONE]")
    chunks = sse_payloads(response.text)
    assert completion_text(chunks) == plain
    assert chunks[0]["object"] == "text_completion"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_completions_stream_includes_usage_when_asked(client: TestClient):
    response = client.post(
        "/v1/completions",
        json={
            "prompt": "usage please",
            "max_tokens": 3,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    chunks = sse_payloads(response.text)
    usage_chunks = [chunk for chunk in chunks if chunk.get("usage")]
    assert usage_chunks
    usage = usage_chunks[-1]["usage"]
    assert usage["completion_tokens"] == 3
    assert usage["total_tokens"] == usage["prompt_tokens"] + 3
    assert usage_chunks[-1]["choices"] == []


def test_completions_stream_has_no_usage_by_default(client: TestClient):
    response = client.post(
        "/v1/completions",
        json={"prompt": "no usage", "max_tokens": 2, "stream": True},
    )
    chunks = sse_payloads(response.text)
    assert all("usage" not in chunk for chunk in chunks)


def test_completion_stop_sequence(client: TestClient):
    app = create_app(MockBackend(script=["one", " two", " STOP", " three"]), request_timeout=10)
    local = TestClient(app)
    response = local.post(
        "/v1/completions",
        json={"prompt": "x", "max_tokens": 10, "stop": ["STOP"]},
    )
    body = response.json()
    assert body["choices"][0]["text"] == "one two "
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 3

    streamed = local.post(
        "/v1/completions",
        json={"prompt": "x", "max_tokens": 10, "stop": "STOP", "stream": True},
    )
    chunks = sse_payloads(streamed.text)
    assert completion_text(chunks) == "one two "
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_completion_script_is_truncated_by_max_tokens(client: TestClient):
    app = create_app(MockBackend(script=["a", " b", " c", " d"]), request_timeout=10)
    local = TestClient(app)
    body = local.post(
        "/v1/completions", json={"prompt": "x", "max_tokens": 2}
    ).json()
    assert body["choices"][0]["text"] == "a b"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 2


def test_prompt_list_is_rejected(client: TestClient):
    response = client.post(
        "/v1/completions", json={"prompt": ["one", "two"], "max_tokens": 2}
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "prompt"
    assert error["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def test_chat_non_streaming(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "write a haiku about GPUs"},
            ],
            "max_tokens": 4,
            "seed": 1,
            "temperature": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"].startswith("echo: system be brief")
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 4
    assert body["id"].startswith("chatcmpl-")


def test_chat_accepts_content_parts(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one "},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ],
            "max_tokens": 8,
            "seed": 2,
        },
    )
    assert response.status_code == 200
    text = response.json()["choices"][0]["message"]["content"]
    assert "part one part two" in text


def test_chat_tools_accepted_but_not_executed(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "search", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 2,
            "seed": 5,
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["role"] == "assistant"


def test_chat_max_completion_tokens_alias(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "aliases"}],
            "max_completion_tokens": 3,
            "seed": 9,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["usage"]["completion_tokens"] == 3
    assert body["choices"][0]["finish_reason"] == "length"


def test_chat_streaming_matches_non_streaming(client: TestClient):
    request = {
        "messages": [{"role": "user", "content": "stream the chat"}],
        "max_tokens": 5,
        "seed": 11,
    }
    plain = client.post("/v1/chat/completions", json=request).json()["choices"][0]["message"]["content"]

    response = client.post("/v1/chat/completions", json={**request, "stream": True})
    assert response.status_code == 200
    assert response.text.rstrip().endswith("data: [DONE]")
    chunks = sse_payloads(response.text)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chat_text(chunks) == plain
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_chat_streaming_usage(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "usage"}],
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    chunks = sse_payloads(response.text)
    usage_chunks = [chunk for chunk in chunks if chunk.get("usage")]
    assert len(usage_chunks) == 1
    usage = usage_chunks[0]["usage"]
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == usage["prompt_tokens"] + 2


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_validation_errors_are_openai_objects(client: TestClient):
    cases = [
        ({"messages": [], "max_tokens": 2}, 400, "messages"),
        ({"messages": [{"role": "robot", "content": "hi"}]}, 400, "role"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
            400,
            "content",
        ),
        ({"messages": [{"role": "user", "content": "hi"}], "temperature": 5}, 400, "temperature"),
        ({"messages": [{"role": "user", "content": "hi"}], "n": 2}, 400, "n"),
        ({"messages": [{"role": "user", "content": "hi"}], "stop": [""]}, 400, "stop"),
    ]
    for payload, status, param in cases:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == status, payload
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["message"]
        assert param in (error["param"] or "") or param in error["message"], (param, error)


def test_invalid_json_body(client: TestClient):
    response = client.post(
        "/v1/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_unknown_route_and_method(client: TestClient):
    missing = client.get("/v1/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"

    wrong_method = client.get("/v1/completions")
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "method_not_allowed"


def test_request_id_is_echoed(client: TestClient):
    response = client.post(
        "/v1/completions",
        json={"prompt": "id", "max_tokens": 1},
        headers={"x-request-id": "my-request-1"},
    )
    assert response.headers["x-request-id"] == "my-request-1"


# ---------------------------------------------------------------------------
# Limits, context and timeouts
# ---------------------------------------------------------------------------


def test_context_length_is_enforced():
    app = create_app(MockBackend(), request_timeout=10, max_context_tokens=8)
    local = TestClient(app)
    response = local.post(
        "/v1/completions", json={"prompt": "one two three four five", "max_tokens": 8}
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["param"] == "max_tokens"


def test_fit_max_tokens_shortens_instead_of_failing():
    app = create_app(MockBackend(), request_timeout=10, max_context_tokens=10, fit_max_tokens=True)
    local = TestClient(app)
    body = local.post(
        "/v1/completions", json={"prompt": "one two three", "max_tokens": 100}
    ).json()
    # 3 prompt tokens leave 7 for the completion.
    assert body["usage"]["completion_tokens"] == 7


def test_max_tokens_limit():
    app = create_app(MockBackend(), request_timeout=10, max_tokens_limit=4)
    local = TestClient(app)
    response = local.post(
        "/v1/completions", json={"prompt": "x", "max_tokens": 5}
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "max_tokens"


def test_request_timeout_returns_504():
    app = create_app(MockBackend(token_delay=0.2), request_timeout=0.05)
    local = TestClient(app)
    response = local.post(
        "/v1/completions", json={"prompt": "slow", "max_tokens": 4}
    )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "timeout"


def test_concurrency_limit_returns_429():
    app = create_app(MockBackend(), request_timeout=10, max_concurrency=1)
    client = TestClient(app)
    app.state.limiter.acquire()
    try:
        response = client.post(
            "/v1/completions", json={"prompt": "busy", "max_tokens": 1}
        )
    finally:
        app.state.limiter.release()
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
    assert response.headers["retry-after"] == "1"


def test_streaming_keepalive_comments():
    app = create_app(
        MockBackend(token_delay=0.12),
        request_timeout=10,
        sse_keepalive_s=0.02,
    )
    local = TestClient(app)
    response = local.post(
        "/v1/completions",
        json={"prompt": "keepalive", "max_tokens": 2, "stream": True},
    )
    assert response.status_code == 200
    assert ": keep-alive" in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


# ---------------------------------------------------------------------------
# Backends, API keys, sessions and slots
# ---------------------------------------------------------------------------


def test_backend_error_maps_to_openai_error():
    app = create_app(FailingBackend(), request_timeout=10)
    local = TestClient(app)
    response = local.post(
        "/v1/completions", json={"prompt": "x", "max_tokens": 1}
    )
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "model_crashed"
    assert "model exploded" in error["message"]


def test_streaming_backend_error_is_an_sse_event():
    app = create_app(FailingBackend(), request_timeout=10, sse_keepalive_s=0)
    local = TestClient(app)
    response = local.post(
        "/v1/completions",
        json={"prompt": "x", "max_tokens": 1, "stream": True},
    )
    assert response.status_code == 200
    chunks = sse_payloads(response.text)
    errors = [chunk for chunk in chunks if "error" in chunk]
    assert errors and errors[0]["error"]["code"] == "model_crashed"
    assert response.text.rstrip().endswith("data: [DONE]")


def test_tiny_backend_reports_missing_model(tmp_path):
    backend = TinyModelBackend(model_dir=tmp_path / "missing")
    app = create_app(backend, request_timeout=10)
    local = TestClient(app)
    response = local.post(
        "/v1/completions", json={"prompt": "x", "max_tokens": 1}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_loaded"


def test_api_key_is_enforced():
    app = create_app(MockBackend(), api_key="sk-test,sk-second", request_timeout=10)
    local = TestClient(app)

    missing = local.get("/v1/models")
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "invalid_api_key"
    assert missing.headers["www-authenticate"] == "Bearer"

    wrong = local.get("/v1/models", headers={"authorization": "Bearer nope"})
    assert wrong.status_code == 401

    for key in ("sk-test", "sk-second"):
        assert local.get("/v1/models", headers={"authorization": f"Bearer {key}"}).status_code == 200

    # Health stays public; /slots and POSTs need the key too.
    assert local.get("/health").status_code == 200
    assert local.get("/slots").status_code == 401
    post = local.post("/v1/completions", json={"prompt": "x", "max_tokens": 1})
    assert post.status_code == 401
    ok = local.post(
        "/v1/completions",
        json={"prompt": "x", "max_tokens": 1},
        headers={"authorization": "Bearer sk-test"},
    )
    assert ok.status_code == 200


def test_sessions_bind_slots(client: TestClient):
    first = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "session one"}],
            "max_tokens": 2,
            "session_id": "agent-a",
        },
    )
    assert first.status_code == 200
    assert first.headers["x-session-id"] == "agent-a"

    second = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "session one again"}],
            "max_tokens": 2,
            "session_id": "agent-a",
        },
    )
    assert second.headers["x-session-id"] == "agent-a"

    header_session = client.post(
        "/v1/completions",
        json={"prompt": "via header", "max_tokens": 1},
        headers={"x-session-id": "agent-b"},
    )
    assert header_session.headers["x-session-id"] == "agent-b"

    slots = client.get("/v1/slots").json()
    assert slots["sessions"] == 2
    by_session = {slot["session_id"]: slot for slot in slots["slots"] if slot["session_id"]}
    assert by_session["agent-a"]["n_prompt_tokens"] > 0
    assert by_session["agent-a"]["requests"] >= 2
    assert by_session["agent-b"]["requests"] == 1


def test_invalid_session_id(client: TestClient):
    response = client.post(
        "/v1/completions",
        json={"prompt": "x", "max_tokens": 1, "session_id": "bad id!"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_session_id"


def test_session_eviction_is_lru():
    app = create_app(MockBackend(), request_timeout=10, max_slots=1)
    local = TestClient(app)
    first = local.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": 1,
            "session_id": "agent-a",
        },
    )
    assert first.status_code == 200
    second = local.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "b"}],
            "max_tokens": 1,
            "session_id": "agent-b",
        },
    )
    assert second.status_code == 200
    slots = local.get("/v1/slots").json()
    assert slots["sessions"] == 1
    assert slots["slots"][0]["session_id"] == "agent-b"


def test_slot_manager_reuse_and_busy():
    manager = SlotManager(max_slots=1, max_context_tokens=128)
    slot, needs_reset = manager.acquire("s1")
    assert needs_reset is True
    with pytest.raises(APIError) as busy:
        manager.acquire("s1")
    assert busy.value.status_code == 429
    assert busy.value.code == "session_busy"
    manager.release(slot, prompt_tokens=5, completion_tokens=2)
    again, needs_reset = manager.acquire("s1")
    assert needs_reset is False
    assert again.slot_id == slot.slot_id
    manager.release(again)

    evicted, needs_reset = manager.acquire("s2")
    assert needs_reset is True
    assert evicted.session_id == "s2"
    manager.release(evicted)


class RecordingBackend(MockBackend):
    """MockBackend that records what the server hands the backend."""

    def __init__(self) -> None:
        super().__init__(script=["ok"])
        self.requests = []
        self.resets = []

    def reset_slot(self, slot_id: int) -> None:
        self.resets.append(slot_id)

    def generate(self, request):
        self.requests.append(request)
        yield from super().generate(request)


def test_backend_receives_session_and_slot():
    backend = RecordingBackend()
    app = create_app(backend, request_timeout=10, max_slots=2)
    local = TestClient(app)

    first = local.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "one"}],
            "max_tokens": 2,
            "session_id": "agent-a",
        },
    )
    assert first.status_code == 200
    assert backend.requests[-1].session_id == "agent-a"
    assert backend.requests[-1].slot_id == 0
    assert backend.resets == [0]

    local.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "two"}],
            "max_tokens": 2,
            "session_id": "agent-a",
        },
    )
    assert backend.requests[-1].slot_id == 0
    assert backend.resets == [0]

    local.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "three"}],
            "max_tokens": 2,
            "session_id": "agent-b",
        },
    )
    assert backend.requests[-1].slot_id == 1
    assert backend.resets == [0, 1]

    local.post("/v1/completions", json={"prompt": "transient", "max_tokens": 2})
    assert backend.requests[-1].session_id is None
    assert backend.requests[-1].slot_id is not None


# ---------------------------------------------------------------------------
# Tiers, adapters and multi-model routing
# ---------------------------------------------------------------------------


def test_tier_resolution():
    conservative = resolve_tier("conservative")
    balanced = resolve_tier("balanced")
    assert conservative.level.name == "CONSERVATIVE"
    assert balanced.level.name == "BALANCED"
    assert balanced.expert_cache_gb > 0
    # aggressive (level 2) is wired: the backend builds the same Engine the CLI does, so
    # flat streaming applies (analysis.md 27: 82.2s -> 63.9s on the same prompt)
    aggressive = resolve_tier("aggressive")
    assert aggressive.level.name == "AGGRESSIVE"
    with pytest.raises(ValueError):
        resolve_tier("warp")
    with pytest.raises(ValueError):
        TinyModelBackend(model_dir=None, tier="extreme")


def test_detect_adapter_kind(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert detect_adapter_kind(hf) == "hf_deepseek_v41"
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    (gguf / "model-Q4.gguf").write_bytes(b"GGUF")
    assert detect_adapter_kind(gguf) == "gguf"
    assert detect_adapter_kind(tmp_path / "empty") == "unknown"
    assert detect_adapter_kind(None) == "unknown"


def test_gguf_adapter_reserved(tmp_path):
    gguf = tmp_path / "gguf-model"
    gguf.mkdir()
    (gguf / "model.gguf").write_bytes(b"GGUF")
    backend = TinyModelBackend(model_dir=gguf, model_id="gguf-model")
    app = create_app(backend, request_timeout=10)
    local = TestClient(app)

    models = {entry["id"]: entry for entry in local.get("/v1/models").json()["data"]}
    assert models["gguf-model"]["adapter"] == "gguf"
    assert models["gguf-model"]["supported"] is False
    assert "GGUF" in models["gguf-model"]["note"]

    response = local.post("/v1/completions", json={"prompt": "x", "max_tokens": 1})
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "adapter_not_implemented"


def test_build_backend_single_and_multi(tmp_path):
    mock = build_backend("mock")
    assert isinstance(mock, MockBackend)

    single = build_backend(
        "tiny", model_dir=tmp_path / "one", layers=[0], tier="balanced", model_id="one"
    )
    assert isinstance(single, TinyModelBackend)
    assert single.tier_name == "balanced"
    entry = single.list_models()[0]
    assert entry["id"] == "one"
    assert entry["supported"] is False  # no checkpoint files under tmp_path/"one"

    multi = build_backend(
        "tiny",
        models=[
            {"model_id": "alpha", "model_dir": tmp_path / "a"},
            {"model_id": "beta", "model_dir": tmp_path / "b"},
        ],
        default_model="beta",
        tier="conservative",
    )
    assert isinstance(multi, MultiModelBackend)
    assert multi.default_model == "beta"
    assert [item["id"] for item in multi.list_models()[:2]] == ["alpha", "beta"]

    with pytest.raises(ValueError):
        build_backend("tiny", model_dir=tmp_path / "x", tier="extreme")


def test_multi_model_endpoint_and_routing():
    alpha = RecordingBackend()
    alpha.model_id = "alpha"
    beta = RecordingBackend()
    beta.model_id = "beta"
    backend = MultiModelBackend(
        [
            ModelEntry(
                model_id="alpha",
                factory=lambda: alpha,
                adapter="mock",
                tier="conservative",
                layers=(0,),
            ),
            ModelEntry(
                model_id="beta",
                factory=lambda: beta,
                adapter="gguf",
                tier="balanced",
                layers=(0, 1),
                supported=False,
                note="GGUF adapter is reserved",
            ),
        ],
        default_model="alpha",
    )
    app = create_app(backend, request_timeout=10)
    local = TestClient(app)

    models = {item["id"]: item for item in local.get("/v1/models").json()["data"]}
    assert set(models) >= {"alpha", "beta"}
    assert models["alpha"]["adapter"] == "mock"
    assert models["alpha"]["layers"] == [0]
    assert models["beta"]["supported"] is False
    assert models["beta"]["note"] == "GGUF adapter is reserved"

    response = local.post(
        "/v1/chat/completions",
        json={
            "model": "beta",
            "messages": [{"role": "user", "content": "route me"}],
            "max_tokens": 2,
        },
    )
    assert response.status_code == 200
    assert beta.requests and beta.requests[-1].model == "beta"
    assert not alpha.requests

    fallback = local.post(
        "/v1/chat/completions",
        json={
            "model": "not-registered",
            "messages": [{"role": "user", "content": "fallback"}],
            "max_tokens": 2,
        },
    )
    assert fallback.status_code == 200
    assert alpha.requests and alpha.requests[-1].model == "not-registered"


def test_tiny_slot_kv_ownership():
    class FakeEngine:
        def __init__(self):
            self.kv = {}
            self.resets = 0

        def reset(self):
            self.kv = {}
            self.resets += 1

        def snapshot(self):
            return dict(self.kv)

        def restore(self, state):
            self.kv = dict(state)

    backend = TinyModelBackend(model_dir=None, tier="conservative")
    engine = FakeEngine()
    backend._model = engine
    backend._cfg = SimpleNamespace(max_position_embeddings=128)
    sentinel_three = object()
    sentinel_four = object()

    backend._bind_slot(3)
    assert engine.kv == {} and engine.resets == 1
    engine.kv[0] = sentinel_three
    backend._stash_slot(3)
    assert backend._slot_kv[3][0] is sentinel_three
    assert 4 not in backend._slot_kv

    backend._bind_slot(4)
    assert engine.kv == {}
    engine.kv[0] = sentinel_four
    backend._stash_slot(4)

    backend._bind_slot(3)
    assert engine.kv[0] is sentinel_three

    backend.reset_slot(3)
    assert 3 not in backend._slot_kv
    assert engine.kv == {}
    assert backend._slot_kv[4][0] is sentinel_four


# ---------------------------------------------------------------------------
# Minimal function calling
# ---------------------------------------------------------------------------

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "look something up",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}
CALL_TEXT = '{"name": "lookup", "arguments": {"q": "tinyinfer"}}'


def _tools_client(script):
    return TestClient(create_app(MockBackend(script=script), request_timeout=10))


def test_tool_call_non_streaming():
    client = _tools_client([CALL_TEXT])
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "find tinyinfer"}],
            "tools": [LOOKUP_TOOL],
            "tool_choice": "auto",
            "max_tokens": 8,
            "seed": 1,
        },
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] is None
    call = message["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"q": "tinyinfer"}


def test_tool_call_streaming():
    client = _tools_client([CALL_TEXT])
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "find tinyinfer"}],
            "tools": [LOOKUP_TOOL],
            "max_tokens": 8,
            "stream": True,
        },
    )
    assert response.status_code == 200
    chunks = sse_payloads(response.text)
    tool_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(tool_chunks) == 1
    call = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["function"]["name"] == "lookup"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    content_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0]["delta"].get("content")
    ]
    assert content_chunks == []


def test_tool_call_unknown_name_keeps_text():
    client = _tools_client(['{"name": "unregistered", "arguments": {}}'])
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "tools": [LOOKUP_TOOL],
            "max_tokens": 8,
        },
    )
    choice = response.json()["choices"][0]
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"].startswith('{"name": "unregistered"')
    assert choice["finish_reason"] != "tool_calls"


def test_tools_streaming_plain_text_is_buffered():
    client = _tools_client(["hello", " world"])
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "tools": [LOOKUP_TOOL],
            "max_tokens": 8,
            "stream": True,
        },
    )
    chunks = sse_payloads(response.text)
    assert chat_text(chunks) == "hello world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_tool_choice_none_disables_extraction():
    client = _tools_client([CALL_TEXT])
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "tools": [LOOKUP_TOOL],
            "tool_choice": "none",
            "max_tokens": 8,
        },
    )
    choice = response.json()["choices"][0]
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == CALL_TEXT


# ---------------------------------------------------------------------------
# Optional: real L0 weights end-to-end (set TINYINFER_L0=1 to enable)
# ---------------------------------------------------------------------------

L0_DIR = model_dir()
L0_READY = (
    (L0_DIR / "model-00003-of-00048.safetensors").is_file()
    and (L0_DIR / "model-00002-of-00048.safetensors").is_file()
    and (L0_DIR / "model-00043-of-00048.safetensors").is_file()
)


@pytest.mark.skipif(
    not (os.environ.get("TINYINFER_L0") and L0_READY),
    reason="set TINYINFER_L0=1 with the L0/embed/head shards present",
)
def test_l0_real_weights_end_to_end():
    backend = TinyModelBackend(
        model_dir=L0_DIR, layers=[0], tier="conservative", model_id="deepseek-v41-l0"
    )
    app = create_app(backend, request_timeout=300, max_slots=2)
    with TestClient(app) as client:
        models = {item["id"]: item for item in client.get("/v1/models").json()["data"]}
        assert models["deepseek-v41-l0"]["loaded"] is True
        assert models["deepseek-v41-l0"]["adapter"] == "deepseek_v41_hf"
        assert models["deepseek-v41-l0"]["context_length"] == 1_048_576

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello tinyinfer"}],
                "max_tokens": 3,
                "seed": 7,
                "session_id": "l0-session",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["usage"]["completion_tokens"] == 3
        assert body["usage"]["prompt_tokens"] > 0
        assert isinstance(body["choices"][0]["message"]["content"], str)

        streamed = client.post(
            "/v1/completions",
            json={
                "prompt": "hello",
                "max_tokens": 2,
                "stream": True,
                "session_id": "l0-session",
            },
        )
        assert streamed.status_code == 200
        assert streamed.text.rstrip().endswith("data: [DONE]")

        slots = client.get("/v1/slots").json()
        by_session = {
            slot["session_id"]: slot for slot in slots["slots"] if slot["session_id"]
        }
        assert "l0-session" in by_session
        assert by_session["l0-session"]["n_prompt_tokens"] > 0
        assert backend._slot_kv
