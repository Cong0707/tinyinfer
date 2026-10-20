"""OpenAI-compatible HTTP server for tinyinfer.

This module implements the service layer of tinyinfer: an OpenAI-shaped API
(/v1/models, /v1/completions, /v1/chat/completions, /v1/slots, /health) with
SSE streaming, request ids, usage accounting, API keys and per-session slots.
It is deliberately backend-agnostic:

* Backend -- the protocol the model engine implements.
* MockBackend -- a deterministic, tiny in-process backend used by the tests
  and by "tinyinfer serve --backend mock".
* TinyModelBackend -- the integration point for tinyinfer.model.TinyModel
  (HF safetensors, fp8/fp4 experts).  It loads lazily and runs the reference
  PyTorch forward; per-slot KV ownership is left for the engine integration.

Design notes (see docs/api.md):

* One HTTP request takes exactly one *slot*.  A slot owns the KV cache and any
  expert cache the backend keeps for a conversation; the server only tracks
  slot/session metadata, so a real engine can reuse a slot across turns.
* session_id (body field or X-Session-Id header) binds a conversation to a
  slot, so parallel agents do not share state.  Requests without a session get
  a transient slot.
* Streaming uses a producer thread pumping the backend iterator into a queue;
  while the model reads a long prompt the HTTP stream sends ": keep-alive"
  comments, so agent clients do not time out.

Run it with::

    python -m tinyinfer.server --backend mock --port 8000

or through the CLI: python -m tinyinfer.cli serve
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import queue
import random
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol, Sequence, runtime_checkable

try:  # fastapi lives in the optional "server" extra
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field, field_validator
    from starlette.concurrency import run_in_threadpool
    from starlette.exceptions import HTTPException as StarletteHTTPException
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise SystemExit(
        "The tinyinfer server extra is not installed.\n"
        "Install it with: uv pip install -e 'tinyinfer[server]' "
        "(or: uv pip install fastapi uvicorn httpx)"
    ) from exc

logger = logging.getLogger("tinyinfer.server")

__all__ = [
    "APIError",
    "Backend",
    "BackendError",
    "BaseBackend",
    "GenerationRequest",
    "MockBackend",
    "SamplingParams",
    "ServerSettings",
    "SlotManager",
    "TinyModelBackend",
    "TokenEvent",
    "Usage",
    "build_backend",
    "create_app",
    "main",
    "parse_layers",
]

DEFAULT_CONTEXT_TOKENS = 1_048_576  # 1M, the DeepSeek-V4.1-Flash context window
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_KEEPALIVE_S = 15.0
MAX_STOP_SEQUENCES = 4
_SESSION_RE = re.compile(r"[A-Za-z0-9._:\-]{1,128}")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class APIError(Exception):
    """An error that maps onto an OpenAI error object and an HTTP status code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_type: str = "server_error",
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code
        self.headers = headers or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class BackendError(Exception):
    """Raised by a backend for a request it cannot serve.

    status_code/error_type/code let the server translate it into an OpenAI
    error object without knowing the backend internals.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_type: str = "server_error",
        code: str = "backend_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code

    def to_api_error(self) -> APIError:
        return APIError(
            self.message,
            status_code=self.status_code,
            error_type=self.error_type,
            code=self.code,
        )


# ---------------------------------------------------------------------------
# Generation protocol
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_counts(cls, prompt_tokens: int, completion_tokens: int) -> "Usage":
        prompt_tokens = max(0, int(prompt_tokens))
        completion_tokens = max(0, int(completion_tokens))
        return cls(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class SamplingParams:
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(slots=True)
class GenerationRequest:
    """Everything a backend needs for one request.

    slot_id identifies the KV-cache / expert-cache slot the backend must use
    for this request.  session_id is the client-facing conversation id bound
    to that slot (None for one-shot requests).
    """

    request_id: str
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    sampling: SamplingParams = field(default_factory=SamplingParams)
    stream: bool = False
    session_id: str | None = None
    slot_id: int | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    deadline: float | None = None


@dataclass(slots=True)
class TokenEvent:
    """One streamed token, or the terminal event of a generation.

    A backend yields zero or more token events (text carries the decoded
    text, token_id the sampled id, both optional) and exactly one terminal
    event with finish_reason set to "stop" or "length".  usage on the terminal
    event is authoritative; the server fills it in if missing.
    """

    text: str = ""
    token_id: int | None = None
    finish_reason: str | None = None
    usage: Usage | None = None


@runtime_checkable
class Backend(Protocol):
    """The contract between the HTTP layer and a model engine.

    The backend owns all model state.  Slots are the unit of cache ownership:
    reset_slot must drop whatever the backend keeps for a slot (KV cache,
    expert cache, sampler state) so it can be reused for another conversation.
    """

    model_id: str
    aliases: tuple[str, ...]

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        """Yield token events and a terminal event for one request."""

    def count_prompt_tokens(self, request: GenerationRequest) -> int:
        """Return the exact prompt token count (for context/usage checks)."""

    def count_tokens(self, text: str) -> int:
        """Approximate token count of a plain string."""

    def render_prompt(self, request: GenerationRequest) -> str:
        """Render the raw prompt text a backend would tokenize."""

    def reset_slot(self, slot_id: int) -> None:
        """Drop all cached state for slot_id."""

    def health(self) -> dict[str, Any]:
        """Small JSON-serialisable status dictionary."""

    def warmup(self) -> None:
        """Load weights/tokenizer; may raise BackendError."""

    def close(self) -> None:
        """Release resources."""


class BaseBackend:
    """Convenience base class for backends; every method has a safe default."""

    model_id: str = "tinyinfer"
    aliases: tuple[str, ...] = ()
    ready: bool = True

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:  # pragma: no cover
        raise NotImplementedError

    def render_prompt(self, request: GenerationRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        return self.format_messages(request.messages or [])

    @staticmethod
    def format_messages(messages: Sequence[dict[str, Any]]) -> str:
        """Plain "role: content" rendering used when a chat template is absent."""
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def count_tokens(self, text: str) -> int:
        """Rough estimate used when a backend has no tokenizer (~4 chars/token)."""
        return max(1, (len(text) + 3) // 4)

    def count_prompt_tokens(self, request: GenerationRequest) -> int:
        return self.count_tokens(self.render_prompt(request))

    def reset_slot(self, slot_id: int) -> None:
        return None

    def health(self) -> dict[str, Any]:
        return {"backend": type(self).__name__, "model": self.model_id, "ready": bool(self.ready)}

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


def _split_at_stop(acc: str, fragment: str, stops: Sequence[str]) -> tuple[str, bool]:
    """Return (text to append, whether a stop sequence was hit).

    acc is the text emitted so far and fragment the next piece; a stop
    sequence may straddle the boundary, so both are checked together.
    """
    if not stops:
        return fragment, False
    haystack = acc + fragment
    hit = None
    for stop in stops:
        if not stop:
            continue
        pos = haystack.find(stop)
        if pos >= 0 and (hit is None or pos < hit):
            hit = pos
    if hit is None:
        return fragment, False
    return haystack[:hit][len(acc):], True


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

MOCK_VOCAB: tuple[str, ...] = (
    "the",
    "edge",
    "of",
    "reason",
    "on",
    "tiny",
    "hardware",
    "memory",
    "expert",
    "streaming",
    "slot",
    "agent",
    "NVMe",
    "token",
    "window",
    "cache",
    "8GB",
    "GPU",
    "blackwell",
    "quantized",
)


def _fragments_for_mock(
    prompt_text: str, sampling: SamplingParams, vocab: Sequence[str]
) -> list[str]:
    seed = 0 if sampling.seed is None else int(sampling.seed)
    rng = random.Random(f"{seed}|{prompt_text}")
    words = re.findall(r"[A-Za-z0-9_']+", prompt_text.lower())[:6]
    fragments = ["echo:"] + [f" {word}" for word in words]
    while len(fragments) < sampling.max_tokens:
        fragments.append(f" {rng.choice(vocab)}")
    return fragments[: sampling.max_tokens]


class MockBackend(BaseBackend):
    """Deterministic in-process backend for tests and demos.

    The output starts with "echo:" plus the first words of the prompt, then
    fills up to max_tokens with words from vocab.  The same prompt and seed
    always produce the same text.  script replaces the generated fragments
    (used to exercise stop sequences and terminal paths), and token_delay
    simulates a slow engine (streaming keep-alives, timeouts).
    """

    def __init__(
        self,
        model_id: str = "tinyinfer-mock",
        *,
        vocab: Sequence[str] | None = None,
        script: Sequence[str] | None = None,
        token_delay: float = 0.0,
        aliases: Sequence[str] = ("tinyinfer", "mock"),
    ) -> None:
        self.model_id = model_id
        self.aliases = tuple(dict.fromkeys((model_id, *aliases)))
        self.vocab = tuple(vocab or MOCK_VOCAB)
        self.script = list(script) if script is not None else None
        self.token_delay = max(0.0, float(token_delay))

    def health(self) -> dict[str, Any]:
        return {
            "backend": "MockBackend",
            "model": self.model_id,
            "ready": True,
            "fragments": "script" if self.script is not None else "random",
            "token_delay_s": self.token_delay,
        }

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.split()))

    def _fragments(self, prompt_text: str, sampling: SamplingParams) -> list[str]:
        if self.script is not None:
            return list(self.script)
        return _fragments_for_mock(prompt_text, sampling, self.vocab)

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        prompt_text = self.render_prompt(request)
        prompt_tokens = request.prompt_tokens
        if prompt_tokens is None:
            prompt_tokens = self.count_prompt_tokens(request)
        sampling = request.sampling
        fragments = self._fragments(prompt_text, sampling)
        acc = ""
        emitted = 0
        finish = "length" if len(fragments) >= sampling.max_tokens else "stop"
        for fragment in fragments:
            if emitted >= sampling.max_tokens:
                finish = "length"
                break
            if self.token_delay:
                time.sleep(self.token_delay)
            delta, stopped = _split_at_stop(acc, fragment, sampling.stop)
            if stopped:
                if delta:
                    yield TokenEvent(text=delta, token_id=emitted)
                    emitted += 1
                finish = "stop"
                break
            acc += fragment
            yield TokenEvent(text=fragment, token_id=emitted)
            emitted += 1
        yield TokenEvent(
            finish_reason=finish,
            usage=Usage.from_counts(prompt_tokens, emitted),
        )


# ---------------------------------------------------------------------------
# tinyinfer backend (integration point)
# ---------------------------------------------------------------------------


class TinyModelBackend(BaseBackend):
    """Backend that drives the reference tinyinfer.model.TinyModel.

    Loading is lazy and explicit: warmup() (called by serve) or the first
    generate/count_prompt_tokens call builds the ShardStore, TinyConfig and
    TinyModel for the configured model directory and layer list.  It is
    intentionally conservative: it runs the PyTorch reference path on CPU, one
    request at a time.

    Per-slot cache ownership is reserved but not implemented yet: the reference
    attention layers keep a single KV ring buffer, so reset_slot resets it and
    concurrent requests are serialised with a lock.  When the engine gains
    per-slot KV/expert caches this is the class to wire them into.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        layers: Sequence[int] | None = None,
        device: str = "auto",
        model_id: str | None = None,
        tokenizer: Any | None = None,
        aliases: Sequence[str] = ("tinyinfer",),
        load_at_init: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.layers = list(layers) if layers is not None else None
        self.requested_device = device if device in ("auto", "cpu", "cuda") else "auto"
        self.model_id = model_id or (
            self.model_dir.name if self.model_dir else "tinyinfer-deepseek-v4.1-flash"
        )
        self.aliases = tuple(dict.fromkeys((self.model_id, *aliases)))
        self._tokenizer = tokenizer
        self._model = None
        self._cfg = None
        self._store = None
        self._torch = None
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        if load_at_init:
            self.load()

    # -- lifecycle ---------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._model is not None

    def health(self) -> dict[str, Any]:
        return {
            "backend": "TinyModelBackend",
            "model": self.model_id,
            "ready": self.ready,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "layers": self.layers,
            "device": "cpu (reference PyTorch path)",
            "tokenizer": type(self._tokenizer).__name__ if self._tokenizer is not None else None,
        }

    def close(self) -> None:
        self._model = None
        self._cfg = None
        store = self._store
        self._store = None
        if store is not None:
            try:
                store.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("closing shard store failed", exc_info=True)

    def _check_files(self) -> None:
        if self.model_dir is None:
            raise BackendError(
                "TinyModelBackend has no model_dir; start the server with --model-dir DIR",
                status_code=503,
                code="model_not_loaded",
            )
        if not (self.model_dir / "config.json").is_file():
            raise BackendError(
                f"config.json not found under {self.model_dir}",
                status_code=503,
                code="model_not_loaded",
            )
        if not (self.model_dir / "model.safetensors.index.json").is_file():
            raise BackendError(
                f"model.safetensors.index.json not found under {self.model_dir} "
                "(download not finished?)",
                status_code=503,
                code="model_not_loaded",
            )

    def load(self) -> None:
        """Load config, sharded weights and tokenizer (idempotent)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._check_files()
            try:
                import torch
                from .config import TinyConfig
                from .model import TinyModel
                from .weights import ShardStore
            except Exception as exc:  # pragma: no cover - torch is installed in the venv
                raise BackendError(
                    f"cannot import the tinyinfer reference implementation: {exc}",
                    status_code=503,
                    code="model_not_loaded",
                ) from exc
            cfg = TinyConfig.from_hf(self.model_dir)
            layer_ids = list(range(cfg.n_layers)) if self.layers is None else list(self.layers)
            store = ShardStore(self.model_dir)
            try:
                model = TinyModel(store, cfg, layer_ids)
            except Exception as exc:
                try:
                    store.close()
                except Exception:  # pragma: no cover
                    pass
                raise BackendError(
                    f"failed to load tinyinfer model from {self.model_dir}: {exc}",
                    status_code=500,
                    code="model_load_failed",
                ) from exc
            if self._tokenizer is None:
                self._tokenizer = self._load_tokenizer()
            self._torch = torch
            self._cfg = cfg
            self._store = store
            self._model = model

    def warmup(self) -> None:
        self.load()

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(
                str(self.model_dir), trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:
            raise BackendError(
                f"failed to load tokenizer from {self.model_dir}: {exc}",
                status_code=503,
                code="tokenizer_load_failed",
            ) from exc

    # -- prompt handling ---------------------------------------------------
    def render_prompt(self, request: GenerationRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        messages = request.messages or []
        tokenizer = self._tokenizer
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                logger.debug("chat template rendering failed; using plain roles", exc_info=True)
        return self.format_messages(messages)

    def count_prompt_tokens(self, request: GenerationRequest) -> int:
        self.load()
        return len(self._encode(self.render_prompt(request)))

    def _encode(self, text: str) -> list[int]:
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "encode"):
            return list(tokenizer.encode(text))
        encoded = tokenizer(text)["input_ids"]
        return list(encoded)

    def _decode(self, ids: Sequence[int]) -> str:
        tokenizer = self._tokenizer
        try:
            return tokenizer.decode(list(ids), skip_special_tokens=True)
        except TypeError:  # pragma: no cover - tokenizer without skip_special_tokens
            return tokenizer.decode(list(ids))

    def _eos_ids(self) -> set[int]:
        tokenizer = self._tokenizer
        ids: set[int] = set()
        for name in ("eos_token_id", "eos_id"):
            value = getattr(tokenizer, name, None)
            if isinstance(value, int):
                ids.add(value)
        generation_config = getattr(tokenizer, "generation_config", None)
        values = getattr(generation_config, "eos_token_id", None) if generation_config else None
        if isinstance(values, int):
            ids.add(values)
        elif isinstance(values, (list, tuple)):
            ids.update(int(value) for value in values if isinstance(value, int))
        return ids

    # -- generation --------------------------------------------------------
    def _sample_token(self, logits, sampling: SamplingParams, generator) -> int:
        torch = self._torch
        logits = logits.float().view(-1)
        if sampling.temperature <= 0:
            return int(torch.argmax(logits).item())
        logits = logits / max(float(sampling.temperature), 1e-5)
        if sampling.top_k and 0 < sampling.top_k < logits.numel():
            kth = torch.topk(logits, int(sampling.top_k)).values[-1]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        if sampling.top_p < 1.0:
            sorted_probs, sorted_index = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > float(sampling.top_p)
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_probs = torch.where(remove, torch.zeros_like(sorted_probs), sorted_probs)
            probs = torch.zeros_like(probs).scatter_(0, sorted_index, sorted_probs)
            total = probs.sum()
            if total <= 0:  # pragma: no cover - degenerate distribution
                return int(torch.argmax(logits).item())
            probs = probs / total
        return int(torch.multinomial(probs, 1, generator=generator).item())

    def _reset_kv(self) -> None:
        model = self._model
        if model is None:
            return
        for block in getattr(model, "blocks", []):
            attention = getattr(block, "attn", None)
            reset = getattr(attention, "reset_cache", None)
            if callable(reset):
                reset()

    def reset_slot(self, slot_id: int) -> None:
        # Per-slot caches are a later engine feature; the reference model keeps
        # a single ring buffer, so resetting it is the safe approximation.
        self._reset_kv()

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        with self._gen_lock:
            yield from self._generate_locked(request)

    def _generate_locked(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        self.load()
        torch = self._torch
        cfg = self._cfg
        sampling = request.sampling
        prompt_text = self.render_prompt(request)
        prompt_ids = self._encode(prompt_text)
        if not prompt_ids:
            fallback = getattr(self._tokenizer, "bos_token_id", None)
            prompt_ids = [int(fallback)] if isinstance(fallback, int) else [0]
        prompt_tokens = request.prompt_tokens if request.prompt_tokens is not None else len(prompt_ids)
        if len(prompt_ids) + sampling.max_tokens > cfg.max_position_embeddings:
            raise BackendError(
                f"prompt ({len(prompt_ids)} tokens) plus max_tokens ({sampling.max_tokens}) "
                f"exceeds the model context ({cfg.max_position_embeddings})",
                status_code=400,
                error_type="invalid_request_error",
                code="context_length_exceeded",
            )

        self._reset_kv()
        eos_ids = self._eos_ids()
        generator = None
        if sampling.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(sampling.seed) & 0xFFFFFFFF)
        generated: list[int] = []
        emitted_text = ""
        token_count = 0
        finish = "length"
        with torch.inference_mode():
            ids = torch.tensor([prompt_ids], dtype=torch.long)
            logits = self._model.forward(ids, 0)
            position = len(prompt_ids)
            for step in range(sampling.max_tokens):
                if request.deadline is not None and time.monotonic() > request.deadline:
                    raise BackendError(
                        "generation timed out",
                        status_code=504,
                        error_type="timeout_error",
                        code="timeout",
                    )
                next_id = self._sample_token(logits[:, -1, :], sampling, generator)
                if next_id in eos_ids:
                    finish = "stop"
                    break
                generated.append(next_id)
                token_count += 1
                full_text = self._decode(generated)
                delta = full_text[len(emitted_text):] if full_text.startswith(emitted_text) else ""
                if delta:
                    emit, stopped = _split_at_stop(emitted_text, delta, sampling.stop)
                    emitted_text += emit
                    if emit:
                        yield TokenEvent(text=emit, token_id=next_id)
                    if stopped:
                        finish = "stop"
                        break
                if step + 1 >= sampling.max_tokens:
                    finish = "length"
                    break
                position += 1
                logits = self._model.forward(torch.tensor([[next_id]], dtype=torch.long), position)
        yield TokenEvent(
            finish_reason=finish,
            usage=Usage.from_counts(prompt_tokens, token_count),
        )


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Slot:
    slot_id: int
    session_id: str | None = None
    busy: bool = False
    n_ctx: int = DEFAULT_CONTEXT_TOKENS
    n_prompt_tokens: int = 0
    n_completion_tokens: int = 0
    requests: int = 0
    created_at: float = 0.0
    last_used: float = 0.0


class SlotManager:
    """Maps sessions to slots and records per-slot usage.

    The backend owns the actual KV/expert caches; this class only decides which
    slot (and therefore which conversation) a request belongs to.  Idle session
    slots are evicted least-recently-used when a new session arrives.
    """

    def __init__(self, max_slots: int = 4, max_context_tokens: int = DEFAULT_CONTEXT_TOKENS) -> None:
        if max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        self.max_slots = int(max_slots)
        self.max_context_tokens = int(max_context_tokens)
        now = time.time()
        self._slots: list[Slot] = [
            Slot(slot_id=i, n_ctx=self.max_context_tokens, created_at=now, last_used=now)
            for i in range(self.max_slots)
        ]
        self._sessions: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()

    def _take_slot_locked(self) -> tuple[Slot, bool]:
        for slot in self._slots:
            if not slot.busy and slot.session_id is None:
                return slot, bool(slot.requests) or slot.n_prompt_tokens > 0
        for session_id in list(self._sessions):
            slot = self._slots[self._sessions[session_id]]
            if not slot.busy:
                del self._sessions[session_id]
                slot.session_id = None
                return slot, True
        raise APIError(
            f"all {self.max_slots} slots are busy; retry shortly",
            status_code=429,
            error_type="rate_limit_error",
            code="server_busy",
            headers={"Retry-After": "1"},
        )

    def acquire(self, session_id: str | None) -> tuple[Slot, bool]:
        """Return (slot, needs_reset) and mark the slot busy.

        needs_reset is true when the backend must drop old cache state for the
        slot before serving this request (new session or transient slot).
        """
        now = time.time()
        with self._lock:
            if session_id is not None:
                bound = self._sessions.get(session_id)
                if bound is not None:
                    slot = self._slots[bound]
                    if slot.busy:
                        raise APIError(
                            f"session {session_id!r} already has a request in flight",
                            status_code=429,
                            error_type="rate_limit_error",
                            code="session_busy",
                            headers={"Retry-After": "1"},
                        )
                    slot.busy = True
                    slot.last_used = now
                    self._sessions.move_to_end(session_id)
                    return slot, False
                slot, reused = self._take_slot_locked()
                slot.session_id = session_id
                slot.busy = True
                slot.last_used = now
                slot.n_prompt_tokens = 0
                slot.n_completion_tokens = 0
                self._sessions[session_id] = slot.slot_id
                return slot, True
            slot, reused = self._take_slot_locked()
            slot.busy = True
            slot.last_used = now
            slot.n_prompt_tokens = 0
            slot.n_completion_tokens = 0
            return slot, True

    def release(self, slot: Slot, *, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._lock:
            slot.busy = False
            slot.last_used = time.time()
            slot.requests += 1
            slot.n_prompt_tokens = max(0, int(prompt_tokens))
            slot.n_completion_tokens = max(0, int(completion_tokens))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            slots = [
                {
                    "id_slot": slot.slot_id,
                    "session_id": slot.session_id,
                    "busy": slot.busy,
                    "n_ctx": slot.n_ctx,
                    "n_prompt_tokens": slot.n_prompt_tokens,
                    "n_completion_tokens": slot.n_completion_tokens,
                    "requests": slot.requests,
                }
                for slot in self._slots
            ]
            sessions = len(self._sessions)
        return {
            "total_slots": self.max_slots,
            "busy_slots": sum(1 for entry in slots if entry["busy"]),
            "sessions": sessions,
            "slots": slots,
        }

    @property
    def busy(self) -> int:
        with self._lock:
            return sum(1 for slot in self._slots if slot.busy)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


class ConcurrencyLimiter:
    """Bounded semaphore returning an OpenAI-style 429 when full."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.limit = int(limit)
        self._semaphore = threading.BoundedSemaphore(self.limit)
        self._active = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if not self._semaphore.acquire(blocking=False):
            raise APIError(
                f"too many concurrent requests (limit {self.limit})",
                status_code=429,
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
                headers={"Retry-After": "1"},
            )
        with self._lock:
            self._active += 1

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1
        self._semaphore.release()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active


# ---------------------------------------------------------------------------
# Settings + request schemas
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ServerSettings:
    api_keys: tuple[str, ...] = ()
    max_concurrency: int = 4
    request_timeout: float = DEFAULT_TIMEOUT_S
    max_slots: int = 4
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS
    max_tokens_limit: int = DEFAULT_CONTEXT_TOKENS
    default_max_tokens: int = DEFAULT_MAX_TOKENS
    fit_max_tokens: bool = False
    sse_keepalive_s: float = DEFAULT_KEEPALIVE_S
    docs_url: str | None = "/docs"

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        if self.max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be >= 1")
        if self.max_tokens_limit < 1:
            raise ValueError("max_tokens_limit must be >= 1")
        if not 1 <= self.default_max_tokens <= self.max_tokens_limit:
            raise ValueError("default_max_tokens must be within [1, max_tokens_limit]")
        if self.sse_keepalive_s < 0:
            raise ValueError("sse_keepalive_s must be >= 0")


class StreamOptions(BaseModel):
    include_usage: bool = False


class SamplingRequest(BaseModel):
    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1, le=1, description="only n=1 is supported")
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    user: str | None = None
    session_id: str | None = Field(
        default=None, description="tinyinfer extension: conversation/slot id"
    )

    @field_validator("stop")
    @classmethod
    def _check_stop(cls, value):
        if value is None:
            return value
        stops = [value] if isinstance(value, str) else list(value)
        if len(stops) > MAX_STOP_SEQUENCES:
            raise ValueError(f"at most {MAX_STOP_SEQUENCES} stop sequences are supported")
        if any(not isinstance(item, str) or item == "" for item in stops):
            raise ValueError("stop sequences must be non-empty strings")
        return value


class CompletionRequest(SamplingRequest):
    prompt: str | list[str]
    echo: bool = False
    suffix: str | None = None
    logprobs: int | None = None


def _flatten_content(value):
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if not isinstance(part, dict):
                raise ValueError("content parts must be objects")
            part_type = part.get("type")
            if part_type not in ("text", "input_text"):
                raise ValueError("only text content parts are supported")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("text content parts need a string 'text'")
            parts.append(text)
        return "".join(parts)
    return value


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer", "function"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _content_parts(cls, value):
        return _flatten_content(value)


class ChatCompletionRequest(SamplingRequest):
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None

    @field_validator("messages")
    @classmethod
    def _check_messages(cls, value):
        if not value:
            raise ValueError("messages must contain at least one message")
        total = sum(len(message.content or "") for message in value)
        if total == 0:
            raise ValueError("messages must contain at least one non-empty message")
        return value


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _error_response(request: Request, error: APIError) -> JSONResponse:
    headers = dict(error.headers)
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        headers.setdefault("x-request-id", request_id)
    return JSONResponse(error.to_dict(), status_code=error.status_code, headers=headers)


def parse_layers(text: str | None) -> list[int] | None:
    """Parse "0,1,2" into [0, 1, 2] (None keeps every layer)."""
    if text is None or str(text).strip() == "":
        return None
    layers: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    return layers or None


def build_backend(
    kind: str = "mock",
    *,
    model_dir: str | Path | None = None,
    layers: Sequence[int] | None = None,
    device: str = "auto",
    model_id: str | None = None,
    script: Sequence[str] | None = None,
    token_delay: float = 0.0,
) -> Backend:
    """Create one of the built-in backends."""
    if kind == "mock":
        return MockBackend(
            model_id=model_id or "tinyinfer-mock", script=script, token_delay=token_delay
        )
    if kind == "tiny":
        return TinyModelBackend(
            model_dir=model_dir, layers=layers, device=device, model_id=model_id
        )
    raise ValueError(f"unknown backend {kind!r} (choose from 'mock', 'tiny')")


def create_app(
    backend: Backend,
    *,
    api_key: str | Sequence[str] | None = None,
    max_concurrency: int = 4,
    request_timeout: float = DEFAULT_TIMEOUT_S,
    max_slots: int = 4,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    max_tokens_limit: int = DEFAULT_CONTEXT_TOKENS,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
    fit_max_tokens: bool = False,
    sse_keepalive_s: float = DEFAULT_KEEPALIVE_S,
    docs_url: str | None = "/docs",
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    """Build the FastAPI application around backend."""
    if isinstance(api_key, str):
        api_keys = tuple(key.strip() for key in api_key.split(",") if key.strip())
    elif api_key is None:
        api_keys = ()
    else:
        api_keys = tuple(str(key) for key in api_key)
    # Defaults must fit inside limits; clamp instead of failing at startup.
    default_max_tokens = min(int(default_max_tokens), int(max_tokens_limit))
    settings = ServerSettings(
        api_keys=api_keys,
        max_concurrency=max_concurrency,
        request_timeout=request_timeout,
        max_slots=max_slots,
        max_context_tokens=max_context_tokens,
        max_tokens_limit=max_tokens_limit,
        default_max_tokens=default_max_tokens,
        fit_max_tokens=fit_max_tokens,
        sse_keepalive_s=sse_keepalive_s,
        docs_url=docs_url,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at = time.time()
        logger.info(
            "tinyinfer server starting: %s (backend=%s)",
            backend.model_id,
            type(backend).__name__,
        )
        try:
            await run_in_threadpool(backend.warmup)
        except Exception:
            logger.exception("backend warmup failed; /health will report ready=false")
        try:
            yield
        finally:
            try:
                await run_in_threadpool(backend.close)
            except Exception:  # pragma: no cover - defensive
                logger.exception("backend close failed")

    app = FastAPI(
        title="tinyinfer API",
        version=_package_version(),
        description="OpenAI-compatible API for tinyinfer "
        "(DeepSeek-V4.1-Flash on consumer hardware).",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url="/openapi.json" if docs_url else None,
    )
    app.state.backend = backend
    app.state.settings = settings
    app.state.limiter = ConcurrencyLimiter(settings.max_concurrency)
    app.state.slots = SlotManager(settings.max_slots, settings.max_context_tokens)
    app.state.started_at = time.time()

    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["x-request-id", "x-session-id"],
        )

    # -- middleware: request id + API key ---------------------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = (request.headers.get("x-request-id") or "").strip()[:128]
        request.state.request_id = request_id or f"req_{uuid.uuid4().hex}"
        configured = request.app.state.settings.api_keys
        path = request.url.path
        if configured and (path.startswith("/v1/") or path == "/slots"):
            authorization = request.headers.get("authorization", "")
            scheme, _, token = authorization.partition(" ")
            valid = (
                scheme.lower() == "bearer"
                and bool(token)
                and any(hmac.compare_digest(token, key) for key in configured)
            )
            if not valid:
                return _error_response(
                    request,
                    APIError(
                        "Incorrect API key provided. "
                        "Set it as 'Authorization: Bearer <key>'.",
                        status_code=401,
                        error_type="invalid_request_error",
                        param=None,
                        code="invalid_api_key",
                        headers={"WWW-Authenticate": "Bearer"},
                    ),
                )
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    # -- exception handlers: every error is an OpenAI error object --------
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return _error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        details: list[str] = []
        param: str | None = None
        for item in errors[:4]:
            location = [str(part) for part in item.get("loc", ()) if part != "body"]
            label = ".".join(location) or "request"
            details.append(f"{label}: {item.get('msg', 'invalid value')}")
            if param is None and location:
                param = location[-1]
        return _error_response(
            request,
            APIError(
                "Invalid request: " + "; ".join(details),
                status_code=400,
                error_type="invalid_request_error",
                param=param,
                code="invalid_request",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            message = f"Unknown request URL: {request.method} {request.url.path}"
            code, error_type = "not_found", "invalid_request_error"
        elif exc.status_code == 405:
            message = f"Method {request.method} is not allowed on {request.url.path}"
            code, error_type = "method_not_allowed", "invalid_request_error"
        else:
            message = str(exc.detail)
            code = "http_error"
            error_type = "server_error" if exc.status_code >= 500 else "invalid_request_error"
        return _error_response(
            request,
            APIError(
                message,
                status_code=exc.status_code,
                error_type=error_type,
                code=code,
                headers=getattr(exc, "headers", None),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("unhandled error for %s %s", request.method, request.url.path)
        return _error_response(
            request,
            APIError(
                f"Internal server error: {type(exc).__name__}: {exc}",
                status_code=500,
                error_type="server_error",
                code="internal_error",
            ),
        )

    _register_routes(app)
    return app


def _package_version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0.1.0"


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", f"req_{uuid.uuid4().hex}")


def _resolve_session(app: FastAPI, request: Request, body_session: str | None) -> str | None:
    session_id = (body_session or request.headers.get("x-session-id") or "").strip()
    if not session_id:
        return None
    if not _SESSION_RE.fullmatch(session_id):
        raise APIError(
            "session_id must be 1-128 characters of [A-Za-z0-9._:-]",
            status_code=400,
            error_type="invalid_request_error",
            param="session_id",
            code="invalid_session_id",
        )
    return session_id


def _sampling_from(
    body: SamplingRequest,
    settings: ServerSettings,
    *,
    max_tokens: int | None = None,
) -> SamplingParams:
    value = body.max_tokens if max_tokens is None else max_tokens
    value = settings.default_max_tokens if value is None else int(value)
    if value < 1:
        raise APIError(
            "max_tokens must be >= 1",
            status_code=400,
            error_type="invalid_request_error",
            param="max_tokens",
            code="invalid_request",
        )
    if value > settings.max_tokens_limit:
        raise APIError(
            f"max_tokens must be <= {settings.max_tokens_limit}",
            status_code=400,
            error_type="invalid_request_error",
            param="max_tokens",
            code="invalid_request",
        )
    raw_stop = body.stop
    if raw_stop is None:
        stops: tuple[str, ...] = ()
    elif isinstance(raw_stop, str):
        stops = (raw_stop,)
    else:
        stops = tuple(str(item) for item in raw_stop)
    return SamplingParams(
        max_tokens=value,
        temperature=float(body.temperature),
        top_p=float(body.top_p),
        top_k=int(body.top_k),
        stop=stops,
        seed=None if body.seed is None else int(body.seed),
    )


def _check_context(
    app: FastAPI,
    request: GenerationRequest,
    *,
    param: str,
) -> None:
    settings: ServerSettings = app.state.settings
    backend: Backend = app.state.backend
    try:
        prompt_tokens = int(backend.count_prompt_tokens(request))
    except BackendError as exc:
        raise exc.to_api_error() from exc
    request.prompt_tokens = prompt_tokens
    if prompt_tokens >= settings.max_context_tokens:
        raise APIError(
            f"prompt is too long: {prompt_tokens} tokens, "
            f"the context is {settings.max_context_tokens}",
            status_code=400,
            error_type="invalid_request_error",
            param=param,
            code="context_length_exceeded",
        )
    room = settings.max_context_tokens - prompt_tokens
    if request.sampling.max_tokens > room:
        if settings.fit_max_tokens and room >= 1:
            request.sampling.max_tokens = room
        else:
            raise APIError(
                f"max_tokens ({request.sampling.max_tokens}) plus prompt tokens "
                f"({prompt_tokens}) exceeds the context "
                f"({settings.max_context_tokens}); only {room} tokens are left",
                status_code=400,
                error_type="invalid_request_error",
                param="max_tokens",
                code="context_length_exceeded",
            )
    request.deadline = time.monotonic() + settings.request_timeout


def _acquire_resources(app: FastAPI, session_id: str | None) -> Slot:
    limiter: ConcurrencyLimiter = app.state.limiter
    slots: SlotManager = app.state.slots
    limiter.acquire()
    try:
        slot, needs_reset = slots.acquire(session_id)
    except BaseException:
        limiter.release()
        raise
    if needs_reset:
        try:
            app.state.backend.reset_slot(slot.slot_id)
        except Exception:  # pragma: no cover - backend specific
            logger.exception("backend.reset_slot(%s) failed", slot.slot_id)
    return slot


def _release_resources(
    app: FastAPI,
    slot: Slot,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    app.state.slots.release(
        slot, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    app.state.limiter.release()


def _close_generator(generator) -> None:
    if generator is None:
        return
    close = getattr(generator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - defensive
            logger.debug("closing generator failed", exc_info=True)


def _run_generation(app: FastAPI, request: GenerationRequest) -> tuple[str, str, Usage]:
    """Consume a backend generation to completion (non-streaming path)."""
    backend: Backend = app.state.backend
    settings: ServerSettings = app.state.settings
    slot = _acquire_resources(app, request.session_id)
    request.slot_id = slot.slot_id
    generator = None
    text_parts: list[str] = []
    finish_reason: str | None = None
    usage: Usage | None = None
    completion_tokens = 0
    try:
        try:
            generator = backend.generate(request)
        except BackendError as exc:
            raise exc.to_api_error() from exc
        while True:
            if request.deadline is not None and time.monotonic() > request.deadline:
                raise APIError(
                    f"request timed out after {settings.request_timeout:g}s",
                    status_code=504,
                    error_type="timeout_error",
                    code="timeout",
                )
            try:
                event = next(generator)
            except StopIteration:
                break
            except BackendError as exc:
                raise exc.to_api_error() from exc
            if event.finish_reason is not None:
                finish_reason = event.finish_reason
                usage = event.usage
                break
            completion_tokens += 1
            if event.text:
                text_parts.append(event.text)
    finally:
        _close_generator(generator)
        _release_resources(
            app,
            slot,
            prompt_tokens=request.prompt_tokens or 0,
            completion_tokens=completion_tokens,
        )
    text = "".join(text_parts)
    finish_reason = finish_reason or "stop"
    if usage is None:
        usage = Usage.from_counts(request.prompt_tokens or 0, completion_tokens)
    return text, finish_reason, usage


class _StreamPump:
    """Runs a backend iterator on a helper thread and feeds a queue."""

    def __init__(self, backend: Backend, request: GenerationRequest) -> None:
        self._backend = backend
        self._request = request
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=256)
        self._cancel = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"tinyinfer-stream-{request.request_id}",
            daemon=True,
        )
        self._thread.start()

    def _put(self, item: tuple[str, Any]) -> bool:
        while not self._cancel.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        generator = None
        try:
            generator = self._backend.generate(self._request)
            for event in generator:
                if self._cancel.is_set():
                    break
                if not self._put(("event", event)):
                    break
        except BackendError as exc:
            self._put(("backend_error", exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("backend streaming failed")
            self._put(("error", exc))
        finally:
            _close_generator(generator)
            self._put(("done", None))

    def get(self, timeout: float | None) -> tuple[str, Any] | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def cancel(self) -> None:
        self._cancel.set()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _model_created(app: FastAPI) -> int:
    return int(getattr(app.state, "started_at", time.time()))


def _model_objects(app: FastAPI) -> list[dict[str, Any]]:
    backend: Backend = app.state.backend
    settings: ServerSettings = app.state.settings
    created = _model_created(app)
    ids = [backend.model_id, *getattr(backend, "aliases", ())]
    seen: list[str] = []
    for model_id in ids:
        if model_id and model_id not in seen:
            seen.append(model_id)
    return [
        {
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": "tinyinfer",
            "context_length": settings.max_context_tokens,
            "max_output_tokens": settings.max_tokens_limit,
        }
        for model_id in seen
    ]


def _completion_chunk(
    completion_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}
        ],
    }


def _chat_chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
    *,
    usage: Usage | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage.to_dict()
    return payload


def _usage_chunk(
    completion_id: str,
    created: int,
    model: str,
    usage: Usage,
    object_name: str,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": object_name,
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage.to_dict(),
    }


def _normalise_prompt(prompt: str | list[str]) -> str:
    if isinstance(prompt, str):
        return prompt
    if len(prompt) == 1:
        return prompt[0]
    raise APIError(
        "only a single string prompt is supported (got a list of "
        f"{len(prompt)}; send one request per prompt)",
        status_code=400,
        error_type="invalid_request_error",
        param="prompt",
        code="invalid_request",
    )


def _messages_to_dicts(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {
            "role": message.role,
            "content": message.content or "",
        }
        if message.name:
            entry["name"] = message.name
        if message.tool_call_id:
            entry["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            entry["tool_calls"] = message.tool_calls
        payload.append(entry)
    return payload


def _streaming_response(
    app: FastAPI,
    request: GenerationRequest,
    *,
    completion_id: str,
    created: int,
    model: str,
    kind: Literal["completion", "chat"],
    include_usage: bool,
    session_id: str | None,
):
    settings: ServerSettings = app.state.settings
    slot = _acquire_resources(app, request.session_id)
    request.slot_id = slot.slot_id
    try:
        pump = _StreamPump(app.state.backend, request)
    except BaseException:
        _release_resources(app, slot)
        raise

    def frame(payload: dict[str, Any]) -> str:
        return (
            "data: "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n\n"
        )

    def body():
        completion_tokens = 0
        finish_reason: str | None = None
        usage: Usage | None = None
        failed = False
        wait = settings.sse_keepalive_s if settings.sse_keepalive_s > 0 else 1.0
        keepalive = settings.sse_keepalive_s > 0
        try:
            if kind == "chat":
                yield frame(
                    _chat_chunk(
                        completion_id,
                        created,
                        model,
                        {"role": "assistant", "content": ""},
                        None,
                    )
                )
            else:
                yield frame(_completion_chunk(completion_id, created, model, "", None))
            while True:
                item = pump.get(timeout=wait)
                if item is None:
                    if (
                        request.deadline is not None
                        and time.monotonic() > request.deadline
                    ):
                        failed = True
                        yield frame(
                            APIError(
                                f"request timed out after {settings.request_timeout:g}s",
                                status_code=504,
                                error_type="timeout_error",
                                code="timeout",
                            ).to_dict()
                        )
                        break
                    if keepalive:
                        yield ": keep-alive\n\n"
                    continue
                item_kind, payload = item
                if item_kind == "event":
                    event: TokenEvent = payload
                    if event.finish_reason is not None:
                        finish_reason = event.finish_reason
                        usage = event.usage
                        break
                    completion_tokens += 1
                    if event.text:
                        if kind == "chat":
                            yield frame(
                                _chat_chunk(
                                    completion_id,
                                    created,
                                    model,
                                    {"content": event.text},
                                    None,
                                )
                            )
                        else:
                            yield frame(
                                _completion_chunk(
                                    completion_id, created, model, event.text, None
                                )
                            )
                elif item_kind == "backend_error":
                    failed = True
                    yield frame(payload.to_api_error().to_dict())
                    break
                elif item_kind == "error":
                    failed = True
                    yield frame(
                        {
                            "error": {
                                "message": f"{type(payload).__name__}: {payload}",
                                "type": "server_error",
                                "param": None,
                                "code": "stream_error",
                            }
                        }
                    )
                    break
                else:  # done without a terminal event
                    break
            if not failed:
                finish_reason = finish_reason or "stop"
                if usage is None:
                    usage = Usage.from_counts(
                        request.prompt_tokens or 0, completion_tokens
                    )
                if kind == "chat":
                    yield frame(
                        _chat_chunk(completion_id, created, model, {}, finish_reason)
                    )
                else:
                    yield frame(
                        _completion_chunk(
                            completion_id, created, model, "", finish_reason
                        )
                    )
                if include_usage:
                    object_name = (
                        "chat.completion.chunk" if kind == "chat" else "text_completion"
                    )
                    yield frame(
                        _usage_chunk(completion_id, created, model, usage, object_name)
                    )
            yield "data: [DONE]\n\n"
        finally:
            pump.cancel()
            _release_resources(
                app,
                slot,
                prompt_tokens=request.prompt_tokens or 0,
                completion_tokens=completion_tokens,
            )

    headers = dict(_SSE_HEADERS)
    if session_id:
        headers["x-session-id"] = session_id
    return StreamingResponse(body(), media_type="text/event-stream", headers=headers)


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health():
        backend: Backend = app.state.backend
        settings: ServerSettings = app.state.settings
        snapshot = app.state.slots.snapshot()
        try:
            backend_health = backend.health()
        except Exception as exc:  # pragma: no cover - defensive
            backend_health = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "status": "ok",
            "model": backend.model_id,
            "backend": backend_health,
            "ready": bool(getattr(backend, "ready", True)),
            "uptime_s": round(
                time.time() - getattr(app.state, "started_at", time.time()), 3
            ),
            "active_requests": app.state.limiter.active,
            "max_concurrency": settings.max_concurrency,
            "request_timeout_s": settings.request_timeout,
            "max_context_tokens": settings.max_context_tokens,
            "slots": {
                "total": snapshot["total_slots"],
                "busy": snapshot["busy_slots"],
                "sessions": snapshot["sessions"],
            },
        }

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": _model_objects(app)}

    @app.get("/v1/models/{model_id:path}")
    def get_model(model_id: str):
        for entry in _model_objects(app):
            if entry["id"] == model_id:
                return entry
        raise APIError(
            f"The model {model_id!r} does not exist",
            status_code=404,
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )

    @app.get("/v1/slots")
    @app.get("/slots")
    def list_slots():
        return {"object": "list", **app.state.slots.snapshot()}

    @app.post("/v1/completions")
    def create_completion(body: CompletionRequest, request: Request):
        app_ = request.app
        settings: ServerSettings = app_.state.settings
        backend: Backend = app_.state.backend
        prompt = _normalise_prompt(body.prompt)
        session_id = _resolve_session(app_, request, body.session_id)
        model = body.model or backend.model_id
        sampling = _sampling_from(body, settings)
        generation_request = GenerationRequest(
            request_id=_request_id(request),
            prompt=prompt,
            sampling=sampling,
            stream=bool(body.stream),
            session_id=session_id,
            model=model,
        )
        _check_context(app_, generation_request, param="prompt")
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if body.stream:
            include_usage = bool(
                body.stream_options and body.stream_options.include_usage
            )
            return _streaming_response(
                app_,
                generation_request,
                completion_id=completion_id,
                created=created,
                model=model,
                kind="completion",
                include_usage=include_usage,
                session_id=session_id,
            )
        text, finish_reason, usage = _run_generation(app_, generation_request)
        payload = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage.to_dict(),
        }
        response = JSONResponse(payload)
        if session_id:
            response.headers["x-session-id"] = session_id
        return response

    @app.post("/v1/chat/completions")
    def create_chat_completion(body: ChatCompletionRequest, request: Request):
        app_ = request.app
        settings: ServerSettings = app_.state.settings
        backend: Backend = app_.state.backend
        messages = _messages_to_dicts(body.messages)
        session_id = _resolve_session(app_, request, body.session_id)
        model = body.model or backend.model_id
        max_tokens = (
            body.max_tokens
            if body.max_tokens is not None
            else body.max_completion_tokens
        )
        sampling = _sampling_from(body, settings, max_tokens=max_tokens)
        if body.tools:
            logger.debug(
                "tools are accepted but not executed by %s (%d definitions)",
                type(backend).__name__,
                len(body.tools),
            )
        generation_request = GenerationRequest(
            request_id=_request_id(request),
            messages=messages,
            sampling=sampling,
            stream=bool(body.stream),
            session_id=session_id,
            model=model,
        )
        _check_context(app_, generation_request, param="messages")
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if body.stream:
            include_usage = bool(
                body.stream_options and body.stream_options.include_usage
            )
            return _streaming_response(
                app_,
                generation_request,
                completion_id=completion_id,
                created=created,
                model=model,
                kind="chat",
                include_usage=include_usage,
                session_id=session_id,
            )
        text, finish_reason, usage = _run_generation(app_, generation_request)
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage.to_dict(),
        }
        response = JSONResponse(payload)
        if session_id:
            response.headers["x-session-id"] = session_id
        return response

    # Anthropic Messages: reserved interface, not part of the OpenAI path yet.
    @app.post("/v1/messages")
    @app.post("/v1/messages/count_tokens")
    def anthropic_not_implemented():
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "not_implemented_error",
                    "message": "Anthropic Messages compatibility is reserved but "
                    "not implemented; use the OpenAI-compatible "
                    "/v1/chat/completions endpoint.",
                },
            },
            status_code=501,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tinyinfer.server",
        description="Serve tinyinfer through an OpenAI-compatible HTTP API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--backend", choices=("mock", "tiny"), default="mock", help="model backend"
    )
    parser.add_argument(
        "--model-dir", default=None, help="HF safetensors directory (backend=tiny)"
    )
    parser.add_argument(
        "--layers", default=None, help="comma separated layer ids for staged loading"
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--model-id", default=None, help="model id reported by /v1/models")
    parser.add_argument(
        "--api-key", default=None, help="comma separated API keys (or TINYINFER_API_KEY)"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=4, help="parallel HTTP requests"
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="per-request seconds",
    )
    parser.add_argument(
        "--max-slots", type=int, default=4, help="session slots (KV/cache owners)"
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_CONTEXT_TOKENS,
        help="context window",
    )
    parser.add_argument(
        "--max-tokens-limit",
        type=int,
        default=DEFAULT_CONTEXT_TOKENS,
        help="max output tokens",
    )
    parser.add_argument("--default-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--fit-max-tokens",
        action="store_true",
        help="shorten max_tokens to the context room",
    )
    parser.add_argument(
        "--sse-keepalive",
        type=float,
        default=DEFAULT_KEEPALIVE_S,
        help="SSE keep-alive interval (0=off)",
    )
    parser.add_argument(
        "--log-level", default="info", choices=("debug", "info", "warning", "error")
    )
    return parser


def _backend_from_args(args: argparse.Namespace) -> Backend:
    return build_backend(
        args.backend,
        model_dir=args.model_dir,
        layers=parse_layers(args.layers),
        device=args.device,
        model_id=args.model_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.backend == "tiny" and not args.model_dir:
        parser.error("--model-dir is required with --backend tiny")
    api_key = args.api_key or os.environ.get("TINYINFER_API_KEY")
    backend = _backend_from_args(args)
    try:
        backend.warmup()
    except BackendError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
    app = create_app(
        backend,
        api_key=api_key,
        max_concurrency=args.max_concurrency,
        request_timeout=args.request_timeout,
        max_slots=args.max_slots,
        max_context_tokens=args.max_context_tokens,
        max_tokens_limit=args.max_tokens_limit,
        default_max_tokens=args.default_max_tokens,
        fit_max_tokens=args.fit_max_tokens,
        sse_keepalive_s=args.sse_keepalive,
    )
    logger.info(
        "OpenAI-compatible API at http://%s:%s/v1  (backend=%s, model=%s)",
        args.host,
        args.port,
        args.backend,
        backend.model_id,
    )
    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn is not installed (install the server extra)", file=sys.stderr)
        return 2
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
