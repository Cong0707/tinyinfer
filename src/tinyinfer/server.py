"""OpenAI-compatible HTTP server for tinyinfer.

Reference implementations (consult when fixing this file):
  * vLLM       vllm/entrypoints/openai/api_server.py + serving_* (OpenAI schema,
        SSE streaming, usage accounting, request ids) -- mirrored here without engine deps
  * HF TGI     text-generation-inference (same OpenAI-compatible surface)
  * fastapi    starlette/FastAPI streaming responses and pydantic validation

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
from pathlib import Path
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
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol, Sequence, runtime_checkable

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


def _thinking_mode() -> str:
    # "chat" (answer directly) or "thinking" (reason first); the checkpoint encodes the
    # two differently, so this must reach the prompt renderer (see tinyinfer.chat)
    mode = os.environ.get("TI_THINKING_MODE", "chat").strip().lower()
    return mode if mode in ("chat", "thinking") else "chat"

__all__ = [
    "APIError",
    "Backend",
    "ModelEntry",
    "ModelSpec",
    "MultiModelBackend",
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
    "detect_adapter_kind",
    "main",
    "models_from_args",
    "parse_layers",
    "resolve_tier",
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
    repetition_penalty: float = 1.0  # vLLM-style: >1 discourages already generated ids
    # OpenAI-style: subtracted from the logit of every token that has already appeared,
    # once per distinct token however often it appeared. Qwen's model card specifies 2.0
    # for text tasks, which is what keeps a small model from collapsing into a looping
    # count.
    presence_penalty: float = 0.0
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


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------

_TIER_KEYS = ("conservative", "balanced", "aggressive", "extreme")


def resolve_tier(tier: str):
    """Return a tinyinfer.tiers.TierConfig for a supported reference tier.

    Only conservative and balanced are wired into the current PyTorch path;
    aggressive/extreme are accepted by tinyinfer.tiers but need kernels,
    quantized KV and speculation that the reference driver does not use yet.
    """
    from .tiers import OptLevel, plan

    key = str(tier or "conservative").strip().lower()
    levels = {
        "conservative": OptLevel.CONSERVATIVE,
        "balanced": OptLevel.BALANCED,
        "aggressive": OptLevel.AGGRESSIVE,
        "extreme": OptLevel.EXTREME,
    }
    if key not in levels:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(levels)}")
    level = levels[key]
    if level == OptLevel.EXTREME:
        raise ValueError(
            f"tier {key!r} is not implemented (no extreme kernels yet); "
            "use 'conservative', 'balanced' or 'aggressive'"
        )
    # aggressive (level 2) IS wired: the backend constructs the same Engine the CLI does,
    # so flat streaming / the resident-vs-streamed split apply. It used to be refused here
    # from when the server only drove the reference driver; that made the HTTP path ~50%
    # slower than in-process for no reason (analysis.md 27).
    return plan(level)


def detect_adapter_kind(model_dir: str | Path | None) -> str:
    """Best-effort checkpoint layout detection, without loading any weights.

    Both the layout and the architecture are decided here, from config.json, so the
    runtime path is the engine's choice -- a client only ever names a model id.
    """
    if model_dir is None:
        return "unknown"
    directory = Path(model_dir)
    index = directory / "model.safetensors.index.json"
    cfg_file = directory / "config.json"
    if index.is_file():
        if cfg_file.is_file():
            try:
                cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            mt = str(cfg.get("model_type")
                     or (cfg.get("text_config") or {}).get("model_type") or "")
            if mt.startswith("qwen3_5") or mt.startswith("qwen3.5"):
                return "hf_qwen3_5"
        return "hf_deepseek_v41"
    if any(directory.glob("*.gguf")):
        return "gguf"
    return "unknown"


def _adapter_support(adapter: str, model_dir: str | Path | None) -> tuple[bool, str | None]:
    kind = detect_adapter_kind(model_dir) if adapter in ("auto", "", None) else str(adapter)
    if kind == "hf_deepseek_v41":
        return True, None
    if kind == "hf_qwen3_5":
        return True, None
    if kind == "gguf":
        return False, "GGUF adapter is reserved (see tinyinfer.adapters.open_adapter)"
    if kind == "unknown":
        return False, "unrecognized checkpoint layout (need model.safetensors.index.json)"
    return False, f"adapter {kind!r} is not implemented"


# ---------------------------------------------------------------------------
# tinyinfer backend (adapter-driven reference path)
# ---------------------------------------------------------------------------


class _Sampler:
    """Greedy / temperature / top-p / top-k sampling over one logits row.

    Kept device-side apart from the final choice: the row is 248320 wide for Qwen, so the
    sort and the cumulative sum stay on the card and only the chosen id round-trips.
    """

    def __init__(self, params: SamplingParams, torch_module) -> None:
        self.torch = torch_module
        self.temperature = float(params.temperature or 0.0)
        self.top_p = float(params.top_p or 1.0)
        self.top_k = int(params.top_k or 0)
        self.repetition_penalty = float(params.repetition_penalty or 1.0)
        self.presence_penalty = float(params.presence_penalty or 0.0)
        self.generator = None
        if params.seed is not None:
            self.generator = torch_module.Generator(device="cuda")
            self.generator.manual_seed(int(params.seed))
        self._seen: list[int] = []

    def pick(self, logits: Any, generated: int) -> int:
        torch = self.torch
        row = logits[0]
        if self.presence_penalty != 0.0 and self._seen:
            # one penalty per distinct token already emitted; the original row is left
            # untouched (it belongs to the graph buffer, not to us)
            ids = torch.tensor(self._seen, dtype=torch.long, device=row.device).unique()
            row = row.clone()
            row[ids] = row[ids] - self.presence_penalty
        if self.repetition_penalty != 1.0 and self._seen:
            ids = torch.tensor(self._seen, dtype=torch.long, device=row.device).unique()
            picked = row[ids]
            row = row.clone()
            row[ids] = torch.where(picked > 0, picked / self.repetition_penalty,
                                   picked * self.repetition_penalty)
        if self.temperature <= 0.0:
            tok = int(row.argmax())
            self._seen.append(tok)
            return tok
        scores = row.float() / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, scores.numel())
            thresh = torch.topk(scores, k).values[-1]
            scores = torch.where(scores < thresh, float("-inf"), scores)
        if 0.0 < self.top_p < 1.0:
            ordered, order = torch.sort(scores, descending=True)
            probs = torch.softmax(ordered, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            cut = cumulative - probs > self.top_p
            ordered = torch.where(cut, float("-inf"), ordered)
            scores = torch.full_like(scores, float("-inf")).scatter(0, order, ordered)
        probs = torch.softmax(scores, dim=-1)
        tok = int(torch.multinomial(probs, 1, generator=self.generator))
        self._seen.append(tok)
        return tok


class QwenBackend(BaseBackend):
    """Qwen3.5 through the engine: hybrid linear/full attention, driven by a CUDA graph.

    The architecture is nothing like DeepSeek V4.1 (no MLA, no compressor, no engram, no
    hyper-connections, no MoE), so it gets its own backend rather than a branch inside
    TinyModelBackend. What it shares is the contract: the HTTP layer only sees
    GenerationRequest in and TokenEvent out.

    Layers 0..23 alternate GatedDeltaNet (18) with ordinary GQA attention (6). The
    hand-written kernels in tinyinfer.gdn/native cover the delta rule, the depthwise conv
    and the gated norm, because the reference kernels (causal_conv1d, flash-linear-
    attention) cannot be built in this environment. Decode runs from a captured CUDA
    graph over a static KV cache -- DynamicCache concatenates, and inside a capture that
    allocation is frozen, which silently pins the KV length and makes the model repeat
    one token forever.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        model_id: str | None = None,
        max_context_tokens: int = 8192,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_id = model_id or self.model_dir.name
        self.aliases: tuple[str, ...] = (self.model_id,)
        self.max_context_tokens = int(max_context_tokens)
        self._tok = None
        self._model = None
        self._torch = None
        self._slots: dict[int, Any] = {}
        self._graphs: dict[int, Any] = {}
        self._prep: dict[int, list[int]] = {}
        self._gen_lock = threading.Lock()
        self._lock = threading.Lock()
        self.ready = False

    # -- lifecycle -----------------------------------------------------
    def warmup(self) -> None:
        import torch
        import tinyinfer.torch_compat  # noqa: F401  (must precede transformers import)
        from tinyinfer import gdn_patch, qwen_elt, qwen_norm
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gdn_patch.install()
        qwen_norm.install()
        self._tok = AutoTokenizer.from_pretrained(str(self.model_dir),
                                                  trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir), dtype=torch.bfloat16,
            trust_remote_code=True).eval().to("cuda")
        qwen_elt.install(self._model)
        self._torch = torch
        self.ready = True

    def close(self) -> None:
        self._slots.clear()
        self._model = None
        self._tok = None
        self.ready = False

    def reset_slot(self, slot_id: int) -> None:
        with self._lock:
            self._slots.pop(slot_id, None)

    # -- prompt handling -----------------------------------------------
    def render_prompt(self, request: GenerationRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        return self._render_messages(list(request.messages or []))

    def _render_messages(self, messages: Sequence[dict[str, Any]]) -> str:
        if self._tok is None:
            return BaseBackend.format_messages(messages)
        return self._tok.apply_chat_template(list(messages), tokenize=False,
                                             add_generation_prompt=True)

    def _encode(self, request: GenerationRequest):
        if request.prompt is not None:
            return self._tok(request.prompt, return_tensors="pt").input_ids
        text = self._render_messages(list(request.messages or []))
        return self._tok(text, return_tensors="pt").input_ids

    def count_prompt_tokens(self, request: GenerationRequest) -> int:
        if request.prompt_tokens is not None:
            return int(request.prompt_tokens)
        return int(self._encode(request).shape[1])

    def count_tokens(self, text: str) -> int:
        return len(self._tok(text).input_ids)

    def context_limit(self, request: GenerationRequest | None = None) -> int:
        return self.max_context_tokens

    # -- generation -----------------------------------------------------
    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        # One captured graph owns one KV buffer, and replay mutates it in place, so two
        # generations cannot interleave: doing so corrupts the cache and trips a
        # device-side assert (measured: seven concurrent requests from a REPL left open
        # plus two probes). The engine is single-model single-card here, so serializing
        # is correct rather than a shortcut.
        with self._gen_lock:
            with self._torch.inference_mode():
                yield from self._generate(request)

    def _generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        from tinyinfer.hf_graph import GraphDecoder

        torch = self._torch
        params = request.sampling
        ids = self._encode(request).to("cuda")
        prompt_tokens = int(ids.shape[1])
        budget = max(1, min(int(params.max_tokens),
                            self.max_context_tokens - prompt_tokens))
        eos = self._tok.eos_token_id
        dec, fresh = self._decoder_for(request, ids)
        sampler = _Sampler(params, torch)
        produced = 0
        text_buf: list[str] = []
        stops = tuple(s for s in (params.stop or ()) if s)
        with torch.inference_mode():
            # tokens the decoder already produced while preparing the graph
            # the prep tokens (emitted between prefill and the captured step) are the same
            # for every request on this graph shape: _decoder_for recorded them once
            for tok_id in self._prep.get(int(ids.shape[1]), ()):
                piece = self._tok.decode([tok_id], skip_special_tokens=False)
                text_buf.append(piece)
                produced += 1
                yield TokenEvent(text=piece, token_id=tok_id)
                if any("".join(text_buf).endswith(s) for s in stops):
                    yield TokenEvent(
                        finish_reason="stop",
                        usage=Usage.from_counts(prompt_tokens, produced))
                    return
                if produced >= budget:
                    yield TokenEvent(
                        finish_reason="length",
                        usage=Usage.from_counts(prompt_tokens, produced))
                    return
            while produced < budget:
                dec.replay_step()
                tok_id = sampler.pick(dec.logits_buf, produced)
                dec.set_token(tok_id)
                piece = self._tok.decode([tok_id], skip_special_tokens=False)
                text_buf.append(piece)
                produced += 1
                yield TokenEvent(text=piece, token_id=tok_id)
                if eos is not None and tok_id == eos:
                    yield TokenEvent(
                        finish_reason="stop",
                        usage=Usage.from_counts(prompt_tokens, produced))
                    return
                if any("".join(text_buf).endswith(s) for s in stops):
                    yield TokenEvent(
                        finish_reason="stop",
                        usage=Usage.from_counts(prompt_tokens, produced))
                    return
        yield TokenEvent(
            finish_reason="length",
            usage=Usage.from_counts(prompt_tokens, produced))

    # -- capture reuse -----------------------------------------------------
    def _decoder_for(self, request, ids):
        """A captured decoder for this prompt length.

        A capture costs ~137ms (it runs the step several times to prove the replay matches
        eager). Building one per request puts that on top of every completion, which is
        where the 174 tok/s of the raw decode loop collapses to tens. The graph depends on
        the cache *shape*, not its contents, so it is cached by prompt length; GraphDecoder
        .rewind() puts the state back to what replay expects before each use.
        """
        from tinyinfer.hf_graph import GraphDecoder

        length = int(ids.shape[1])
        cached = self._graphs.get(length)
        if cached is not None:
            # the prep tokens belong to the prompt, not to whichever request happened to
            # trigger the capture: they are what the model emits right after prefill
            self._prep.setdefault(length, getattr(cached, "prep_tokens", []))
            # rewind to the exact state replay expects: the capture advanced the cache by
            # one step, and this decoder has been replayed since
            cached.rewind()
            return cached, False
        dec = GraphDecoder.from_prompt(self._model, ids).capture()
        if dec.using_graph:
            dec.prep_tokens = dec.drain()
            self._graphs[length] = dec
            self._prep[length] = dec.prep_tokens
            return dec, True
        return dec, True

    def health(self) -> dict[str, Any]:
        return {
            "backend": "QwenBackend",
            "model": self.model_id,
            "model_dir": str(self.model_dir),
            "ready": self.ready,
            "context_limit": self.max_context_tokens,
        }

class TinyModelBackend(BaseBackend):
    """Backend that drives tinyinfer.model.TinyModel through an adapter.

    The checkpoint layout and weights come from
    tinyinfer.adapters.open_adapter(): HF safetensors for deepseek_v41 is the
    implemented adapter, GGUF is reserved (open_adapter reports it as not
    implemented and the server surfaces a clear 501).

    Tiers: "conservative" is the plain PyTorch reference path; "balanced"
    additionally streams routed experts through tinyinfer.loading.ExpertStore
    (bounded RAM expert cache) and lets the attention dispatcher use tilelang
    when the environment can build it.  Both tiers run on CPU here; device
    placement and per-slot KV reuse at CUDA speed come with the engine work.

    Slots: every request is generated with the KV ring buffers restored from
    that slot's snapshot, and the resulting caches are stored back when the
    request ends.  The OpenAI API is stateless (the prompt always carries the
    whole conversation), so a request still prefills from token 0; the slot
    snapshot is the ownership/isolation boundary and the hook for future
    resume-from-prefix support.  reset_slot() drops a slot's KV and (bounded)
    expert cache.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        adapter: str = "auto",
        layers: Sequence[int] | None = None,
        tier: str = "conservative",
        device: str = "auto",
        model_id: str | None = None,
        tokenizer: Any | None = None,
        aliases: Sequence[str] = ("tinyinfer",),
        load_at_init: bool = False,
        max_cached_experts: int = 64,
        expert_cache_gb: float | None = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.layers = list(layers) if layers is not None else None
        self.requested_device = device if device in ("auto", "cpu", "cuda") else "auto"
        self.requested_adapter = adapter or "auto"
        self.tier_name = str(tier or "conservative").strip().lower()
        self._tier = resolve_tier(self.tier_name)
        self.model_id = model_id or (
            self.model_dir.name if self.model_dir else "tinyinfer-deepseek-v4.1-flash"
        )
        self.aliases = tuple(dict.fromkeys((self.model_id, *aliases)))
        self.max_cached_experts = max(1, int(max_cached_experts))
        self.expert_cache_gb = None if expert_cache_gb is None else float(expert_cache_gb)
        self._tokenizer = tokenizer
        self._model = None
        self._cfg = None
        self._store = None
        self._torch = None
        self._adapter = None
        self.adapter_name = (
            detect_adapter_kind(self.model_dir)
            if self.requested_adapter in ("auto", "", None)
            else self.requested_adapter
        )
        self._expert_store = None
        self._slot_kv: dict[int, dict[int, Any]] = {}
        # slot -> (token ids of the conversation so far, their count); lets a
        # request resume from the prefix instead of re-prefilling it (TI_SERVER_PREFIX=1)
        self._slot_prefix: dict = {}
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        if load_at_init:
            self.load()

    # -- lifecycle ---------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._model is not None

    def health(self) -> dict[str, Any]:
        model = self._model
        return {
            "backend": "TinyModelBackend",
            "model": self.model_id,
            "adapter": self.adapter_name,
            "tier": self.tier_name,
            "layers": self.layers,
            "ready": self.ready,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "device": getattr(model, "device", "cpu") if model is not None else "cpu",
            "gpu_layers_budget": getattr(model, "gpu_layers", None) if model is not None else None,
            "engine_stats": model.stats() if model is not None and hasattr(model, "stats") else None,
            "tokenizer": type(self._tokenizer).__name__ if self._tokenizer is not None else None,
            "slot_caches": len(self._slot_kv),
            "expert_store": self._expert_store is not None,
        }

    def list_models(self) -> list[dict[str, Any]]:
        if self._adapter is not None:
            kind = getattr(self._adapter, "name", self.adapter_name)
            supported, note = True, None
            context = (
                int(self._cfg.max_position_embeddings)
                if self._cfg is not None
                else None
            )
        else:
            kind = self.adapter_name
            supported, note = _adapter_support(kind, self.model_dir)
            context = None
        entry = {
            "id": self.model_id,
            "adapter": kind,
            "tier": self.tier_name,
            "layers": list(self.layers) if self.layers is not None else None,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "loaded": self._model is not None,
            "supported": supported,
            "context_length": context,
        }
        if note:
            entry["note"] = note
        models = [entry]
        for alias in self.aliases:
            if alias != self.model_id:
                models.append({**entry, "id": alias, "alias_of": self.model_id})
        return models

    def close(self) -> None:
        self._model = None
        self._cfg = None
        self._slot_kv.clear()
        expert_store, self._expert_store = self._expert_store, None
        if expert_store is not None:
            try:
                expert_store.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("closing expert store failed", exc_info=True)
        store, self._store = self._store, None
        if store is not None:
            try:
                store.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("closing shard store failed", exc_info=True)
        self._adapter = None

    def _check_files(self) -> None:
        if self.model_dir is None:
            raise BackendError(
                "TinyModelBackend has no model_dir; start the server with --model-dir DIR",
                status_code=503,
                code="model_not_loaded",
            )
        if not self.model_dir.is_dir():
            raise BackendError(
                f"model directory not found: {self.model_dir}",
                status_code=503,
                code="model_not_loaded",
            )
        kind = detect_adapter_kind(self.model_dir)
        if self.requested_adapter == "gguf" or kind == "gguf":
            raise BackendError(
                "the GGUF adapter is reserved but not implemented yet "
                "(tinyinfer.adapters.open_adapter reports GGUF as planned); "
                "use an HF safetensors checkpoint",
                status_code=501,
                code="adapter_not_implemented",
            )
        if self.requested_adapter not in ("auto", "hf", "hf_deepseek_v41", "deepseek_v41"):
            raise BackendError(
                f"adapter {self.requested_adapter!r} is not implemented",
                status_code=501,
                code="adapter_not_implemented",
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
        """Detect the adapter, load the selected layers and the tokenizer."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._check_files()
            try:
                import torch
                from .adapters import open_adapter
                from .model import TinyModel
            except Exception as exc:  # pragma: no cover - torch is installed in the venv
                raise BackendError(
                    f"cannot import the tinyinfer reference implementation: {exc}",
                    status_code=503,
                    code="model_not_loaded",
                ) from exc
            try:
                adapter = open_adapter(self.model_dir)
            except NotImplementedError as exc:
                raise BackendError(
                    str(exc), status_code=501, code="adapter_not_implemented"
                ) from exc
            except (FileNotFoundError, ValueError) as exc:
                raise BackendError(
                    f"cannot open checkpoint {self.model_dir}: {exc}",
                    status_code=503,
                    code="model_not_loaded",
                ) from exc
            cfg = adapter.load_config()
            store = getattr(adapter, "store", None)
            if store is None:
                raise BackendError(
                    f"adapter {getattr(adapter, 'name', type(adapter).__name__)!r} cannot "
                    "drive tinyinfer.model.TinyModel yet (no ShardStore)",
                    status_code=501,
                    code="adapter_not_supported",
                )
            # None means "let the engine decide": the planner then keeps its resident
            # set and streams the rest. Passing every layer id here told the engine that
            # nothing needed streaming, so all 40 layers were treated as resident and the
            # 8GB card paged on every step -- measured as `streamed=False` on all 40 layers
            # and 73-97s for a 12-token prompt against ~51s in-process (analysis.md 27).
            layer_ids = None if self.layers is None else list(self.layers)
            if self._tokenizer is None:
                self._tokenizer = self._load_tokenizer()
            expert_store = self._make_expert_store(store)
            from .engine import Engine

            device = self.requested_device
            if device == "auto":
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            model = Engine(
                adapter,
                tier=self._tier,
                expert_store=expert_store,
                device=device,
                tokenizer=self._tokenizer,
                layers=layer_ids,
            )
            self._adapter = adapter
            self.adapter_name = getattr(adapter, "name", type(adapter).__name__)
            self._cfg = cfg
            self._store = store
            self._model = model
            self._torch = torch

    def _make_expert_store(self, store):
        from .tiers import OptLevel

        if self._tier.level == OptLevel.CONSERVATIVE:
            return None
        try:
            from .loading import ExpertStore

            cache_gb = (
                self.expert_cache_gb
                if self.expert_cache_gb is not None
                else float(self._tier.expert_cache_gb)
            )
            # Let the Engine own the expert store: it builds the cudaHostAlloc pool, the
            # pinned layer slab and the VRAM slot plan around the same tier. A bare
            # ExpertStore here (no pool, no slab) sent every layer through the per-tensor
            # copy path and every expert through pageable reads -- measured 3.3s/token
            # over HTTP against 0.85s in-process.
            logger.debug("deferring the expert store to the engine (tier %s)", self.tier_name)
            return None
        except Exception:
            logger.warning(
                "balanced tier: ExpertStore unavailable; falling back to direct shard reads",
                exc_info=True,
            )
            return None
        self._expert_store = expert_store
        return expert_store

    def warmup(self) -> None:
        self.load()
        # One short forward primes the allocator, the streaming ring and the expert cache.
        # Without it the very first HTTP request paid for all of that inside its own
        # budget: one cold request returned a single visible token where the identical
        # code path in-process returned the full sentence, and every later request was
        # exact. A ~15s start-up cost beats a wrong answer.
        model = self._model
        if model is None or self.requested_device == "cpu":
            return
        torch = self._torch
        bos = getattr(self._tokenizer, "bos_token_id", None)
        ids = torch.tensor([[int(bos) if isinstance(bos, int) else 0]], dtype=torch.long)
        try:
            with torch.inference_mode():
                model.forward(ids, 0)
            model.reset()
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("warmup forward failed; first request will pay for it", exc_info=exc)

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

    def context_limit(self, request: GenerationRequest | None = None) -> int | None:
        if self._cfg is None:
            self.load()
        return int(self._cfg.max_position_embeddings) if self._cfg is not None else None

    # -- prompt handling ---------------------------------------------------
    def render_prompt(self, request: GenerationRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        messages = request.messages or []
        tokenizer = self._tokenizer
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                logger.debug("chat template rendering failed; using the checkpoint encoder", exc_info=True)
        # this tokenizer ships without a chat template, and the model only answers in its
        # own turn format: <|User|>...<|Assistant|></think>... (see tinyinfer.chat)
        from .chat import encode_messages

        return encode_messages(messages, thinking_mode=_thinking_mode())

    def count_prompt_tokens(self, request: GenerationRequest) -> int:
        self.load()
        return len(self._encode(self.render_prompt(request)))

    def _encode(self, text: str) -> list[int]:
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "encode"):
            ids = list(tokenizer.encode(text))
        else:
            ids = list(tokenizer(text)["input_ids"])
        # the checkpoint is a base model: without the sentence-start token it continues
        # whatever pattern the raw text suggests instead of answering (measured: JSON
        # fragments like `"role": "user"` for a plain question). The in-process path has
        # always prepended it, which is why the same prompt answered correctly there.
        bos = getattr(tokenizer, "bos_token_id", None)
        if isinstance(bos, int) and (not ids or ids[0] != bos):
            ids = [int(bos)] + ids
        return ids

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

    # -- slot KV ownership -------------------------------------------------
    def _kv_path(self, key: str):
        # tinyinfer/src/tinyinfer/server.py -> tinyinfer/.kvcache
        root = Path(__file__).resolve().parent.parent.parent / ".kvcache"
        root.mkdir(exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(key))[:80]
        return root / f"{safe}.pt"

    def _load_slot_disk(self, slot_id: int, session_id: str | None = None):
        """Restore a slot's KV + turn prefix from .kvcache (survives a server restart)."""
        # one file per conversation: a slot is a reusable execution lane, a session is the
        # conversation, and two different chats must never share a KV file
        key = session_id or f"slot{slot_id}"
        path = self._kv_path(key)
        if not path.exists():
            return None
        try:
            blob = self._torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning("kvcache %s unreadable (%r); ignoring", path, exc)
            return None
        if blob.get("prefix") is not None:
            self._slot_prefix[key] = blob["prefix"]
        return blob.get("kv")

    @staticmethod
    def _slot_key(slot_id: int, session_id: str | None = None) -> str:
        """The one place that decides how a slot is keyed.

        _bind_slot and _stash_slot share a slot through this key, and they used to build
        it differently -- an int in one, the string "slotN" in the other -- so a slot
        never found the state it had just written. The string form is also the .kvcache
        file name for requests that carry no session.
        """
        return session_id if session_id else f"slot{slot_id}"

    def _bind_slot(self, slot_id: int, session_id: str | None = None) -> None:
        if self._model is None:
            return
        key = self._slot_key(slot_id, session_id)
        state = self._slot_kv.get(key)
        if state is None:
            state = self._load_slot_disk(slot_id, session_id)
        if state is None:
            self._model.reset()
        else:
            # the stash keeps tensors on the host (a device-side copy doubled the resident
            # KV and OOM'd the next prefill), so restore has to place them back -- inside
            # tuples/dicts too, the compressor keeps its pending group as a pair
            t = self._torch
            dev = self._model.device

            def to_dev(v):
                if t.is_tensor(v):
                    return v.to(dev)
                if isinstance(v, tuple):
                    return tuple(to_dev(x) for x in v)
                if isinstance(v, dict):
                    return {k2: to_dev(x) for k2, x in v.items()}
                return v

            state = {k: to_dev(v) for k, v in state.items()}
            self._slot_kv[key] = state
            self._model.restore(state)

    def _stash_slot(self, slot_id: int, session_id: str | None = None) -> None:
        if self._model is None:
            return
        key = self._slot_key(slot_id, session_id)
        t = self._torch

        def to_host(v):
            if t.is_tensor(v):
                return v.detach().cpu()
            if isinstance(v, tuple):
                return tuple(to_host(x) for x in v)
            if isinstance(v, dict):
                return {k2: to_host(x) for k2, x in v.items()}
            return v

        # host-side copy: keeping the snapshot on the device doubled the resident KV and
        # the next turn's prefill then OOM'd (measured: free 1205MB at layer 2)
        state = {k: to_host(v) for k, v in self._model.snapshot().items()}
        self._slot_kv[key] = state
        # persist the session (KV + rendered turn prefix) after every request so the
        # server can be stopped at any time without losing the conversation
        try:
            t.save({"kv": state, "prefix": self._slot_prefix.get(key)},
                   self._kv_path(key))
        except Exception as exc:
            logger.warning("kvcache save failed: %r", exc)

    def _trim_expert_cache(self) -> None:
        limit = self.max_cached_experts
        blocks = getattr(self._model, "_blocks", None)
        if not blocks:
            return
        for block in list(blocks.values()):
            cache = block.ffn._expert_cache
            if not cache:
                continue
            while len(cache) > limit:
                cache.pop(next(iter(cache)))

    def reset_slot(self, slot_id: int) -> None:
        """Drop the KV snapshot and bounded expert cache for one slot."""
        with self._gen_lock:
            # same key as _bind_slot/_stash_slot, otherwise the entry survives the reset
            self._slot_kv.pop(self._slot_key(slot_id), None)
            if self._model is not None:
                self._model.reset()
                self._trim_expert_cache()

    # -- generation --------------------------------------------------------
    def _sample_token(self, logits, sampling: SamplingParams, generator, recent=None) -> int:
        torch = self._torch
        logits = logits.float().view(-1)
        if sampling.repetition_penalty and float(sampling.repetition_penalty) != 1.0 and recent:
            # vLLM semantics: divide positive logits and multiply negative logits of
            # already-generated tokens by the penalty (>1 => less likely to repeat)
            penalty = float(sampling.repetition_penalty)
            idx = torch.tensor(sorted(set(recent)), dtype=torch.long, device=logits.device)
            sel = logits.index_select(0, idx)
            sel = torch.where(sel > 0, sel / penalty, sel * penalty)
            logits = logits.index_copy(0, idx, sel)
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

        slot_id = 0 if request.slot_id is None else int(request.slot_id)
        self._bind_slot(slot_id, request.session_id)
        eos_ids = self._eos_ids()
        generator = None
        if sampling.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(sampling.seed) & 0xFFFFFFFF)
        generated: list[int] = []
        emitted_text = ""
        token_count = 0
        finish = "length"
        try:
            with torch.inference_mode():
                ids = torch.tensor([prompt_ids], dtype=torch.long)
                prefix_len = 0
                # Multi-turn prefix reuse is on by default: an OpenAI chat request
                # carries the whole conversation, so every turn after the first would
                # otherwise re-prefill the entire history (measured 9 -> 25 tokens of
                # prompt turning 10s into 26s). TI_SERVER_PREFIX=0 disables it.
                if __import__("os").environ.get("TI_SERVER_PREFIX") != "0":
                    prev = self._slot_prefix.get(request.session_id or slot_id)
                    if prev is not None:
                        prev_ids, prev_len = prev
                        # >= not >: asking the same one-shot question again has the same
                        # length, and that is exactly the case the KV file is for
                        if (len(prompt_ids) >= prev_len
                                and tuple(prompt_ids[:prev_len]) == prev_ids):
                            prefix_len = prev_len
                try:
                    if torch.cuda.is_available() and torch.cuda.is_initialized():
                        # hand back the previous turn's blocks BEFORE this prefill: a
                        # resumed session otherwise starts with ~1.2GB free and OOMs
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if prefix_len >= len(prompt_ids):
                    # identical prompt to last turn (same one-shot question): keep one token
                    # to compute so we get logits, the rest is already in the KV cache
                    prefix_len = max(0, len(prompt_ids) - 1)
                if prefix_len:
                    logits = self._model.forward(ids[:, prefix_len:], prefix_len)
                else:
                    logits = self._model.forward(ids, 0)
                # the prefill kernels (tilelang workspace for the prompt-length shape)
                # leave gigabytes of cached blocks behind; hand them back before decoding
                # so the decode steps do not page.
                try:
                    if torch.cuda.is_available() and torch.cuda.is_initialized():
                        # hand back the previous turn's blocks BEFORE this prefill: a
                        # resumed session needs the workspace and the first prefill
                        # otherwise ran into 1.2GB of free VRAM and OOM'd
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                # the next sampled token occupies position len(prompt_ids)-1+1 == len(prompt_ids)
                position = len(prompt_ids) - 1
                for step in range(sampling.max_tokens):
                    if request.deadline is not None and time.monotonic() > request.deadline:
                        raise BackendError(
                            "generation timed out",
                            status_code=504,
                            error_type="timeout_error",
                            code="timeout",
                        )
                    next_id = self._sample_token(logits[:, -1, :], sampling, generator, generated)
                    if next_id in eos_ids:
                        finish = "stop"
                        break
                    generated.append(next_id)
                    token_count += 1
                    full_text = self._decode(generated)
                    delta = (
                        full_text[len(emitted_text):]
                        if full_text.startswith(emitted_text)
                        else ""
                    )
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
                    logits = self._model.forward(
                        torch.tensor([[next_id]], dtype=torch.long), position
                    )
        finally:
            self._stash_slot(slot_id, request.session_id)
            if __import__("os").environ.get("TI_SERVER_PREFIX") != "0":
                # Record the *rendered* prompt, not prompt+generated: the next request
                # re-renders the assistant turn from the template, so appending the ids we
                # generated makes the two sequences diverge before the new question and
                # the prefix never matches (measured: turn2 was slower than turn1).
                full = tuple(prompt_ids)
                self._slot_prefix[request.session_id or slot_id] = (full, len(full))
            self._trim_expert_cache()
        yield TokenEvent(
            finish_reason=finish,
            usage=Usage.from_counts(prompt_tokens, token_count),
        )


@dataclass(slots=True)
class ModelEntry:
    """One model advertised by /v1/models, with a lazy backend factory."""

    model_id: str
    factory: Callable[[], Backend]
    adapter: str = "auto"
    tier: str = "conservative"
    layers: tuple[int, ...] | None = None
    model_dir: str | None = None
    aliases: tuple[str, ...] = ()
    supported: bool = True
    note: str | None = None

    def metadata(self, loaded: bool) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.model_id,
            "adapter": self.adapter,
            "tier": self.tier,
            "layers": list(self.layers) if self.layers is not None else None,
            "model_dir": self.model_dir,
            "loaded": loaded,
            "supported": self.supported,
        }
        if self.note:
            data["note"] = self.note
        if self.aliases:
            data["aliases"] = list(self.aliases)
        return data


class MultiModelBackend(BaseBackend):
    """Routes requests to one lazily-loaded backend per model id.

    Used when several checkpoints/adapters are registered (for example an HF
    deepseek_v41 model plus a future GGUF model).  Model names are matched
    against ids and aliases; unknown names fall back to the default model so
    agent clients that send arbitrary model names keep working.
    """

    def __init__(
        self,
        entries: Sequence[ModelEntry],
        *,
        default_model: str | None = None,
    ) -> None:
        if not entries:
            raise ValueError("MultiModelBackend needs at least one model")
        self._entries: OrderedDict[str, ModelEntry] = OrderedDict()
        for entry in entries:
            if entry.model_id in self._entries:
                raise ValueError(f"duplicate model id {entry.model_id!r}")
            self._entries[entry.model_id] = entry
        ids = list(self._entries)
        self.default_model = default_model or ids[0]
        if self.default_model not in self._entries:
            raise ValueError(f"default model {self.default_model!r} is not registered")
        self.model_id = self.default_model
        aliases: list[str] = []
        for entry in entries:
            aliases.append(entry.model_id)
            aliases.extend(entry.aliases)
        self.aliases = tuple(dict.fromkeys(aliases))
        self._backends: dict[str, Backend] = {}
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return bool(self._backends)

    def _entry_for(self, model: str | None) -> ModelEntry:
        if model in self._entries:
            return self._entries[model]
        if model:
            for entry in self._entries.values():
                if model in entry.aliases:
                    return entry
        return self._entries[self.default_model]

    def backend_for(self, model: str | None = None) -> Backend:
        entry = self._entry_for(model)
        with self._lock:
            backend = self._backends.get(entry.model_id)
            if backend is None:
                backend = entry.factory()
                self._backends[entry.model_id] = backend
            return backend

    def list_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for entry in self._entries.values():
            loaded = entry.model_id in self._backends
            models.append(entry.metadata(loaded))
            for alias in entry.aliases:
                models.append({**entry.metadata(loaded), "id": alias, "alias_of": entry.model_id})
        return models

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        yield from self.backend_for(request.model).generate(request)

    def count_prompt_tokens(self, request: GenerationRequest) -> int:
        return self.backend_for(request.model).count_prompt_tokens(request)

    def render_prompt(self, request: GenerationRequest) -> str:
        return self.backend_for(request.model).render_prompt(request)

    def context_limit(self, request: GenerationRequest | None = None) -> int | None:
        getter = getattr(self.backend_for(request.model if request else None), "context_limit", None)
        return getter(request) if callable(getter) else None

    def reset_slot(self, slot_id: int) -> None:
        for backend in list(self._backends.values()):
            backend.reset_slot(slot_id)

    def warmup(self) -> None:
        self.backend_for(self.default_model).warmup()

    def close(self) -> None:
        for backend in list(self._backends.values()):
            backend.close()
        self._backends.clear()

    def health(self) -> dict[str, Any]:
        models = self.list_models()
        return {
            "backend": "MultiModelBackend",
            "model": self.default_model,
            "default_model": self.default_model,
            "ready": any(model.get("loaded") for model in models),
            "models": models,
        }

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
    # both are OpenAI API fields; presence_penalty is the one Qwen's model card names for
    # text tasks (2.0), and it is what keeps a small model from looping over a count
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
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


@dataclass(slots=True)
class ModelSpec:
    """One model the server can route to (CLI --model-dir / --extra-model)."""

    model_id: str
    model_dir: str | Path | None = None
    adapter: str = "auto"
    tier: str = "conservative"
    layers: tuple[int, ...] | None = None
    aliases: tuple[str, ...] = ()


def _coerce_model_specs(
    models: Sequence[ModelSpec | Mapping[str, Any]] | None,
    *,
    model_dir: str | Path | None,
    layers: Sequence[int] | None,
    tier: str,
    adapter: str,
    model_id: str | None = None,
) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if models:
        for item in models:
            if isinstance(item, ModelSpec):
                spec = item
            elif isinstance(item, Mapping):
                raw_layers = item.get("layers")
                spec = ModelSpec(
                    model_id=str(item.get("model_id") or item.get("id") or ""),
                    model_dir=item.get("model_dir"),
                    adapter=str(item.get("adapter", adapter)),
                    tier=str(item.get("tier", tier)),
                    layers=(
                        tuple(int(value) for value in raw_layers)
                        if raw_layers is not None
                        else (tuple(layers) if layers else None)
                    ),
                    aliases=tuple(item.get("aliases") or ()),
                )
            else:
                raise TypeError(f"model specs must be ModelSpec or mapping, got {type(item)!r}")
            if spec.model_dir is None:
                raise ValueError(f"model {spec.model_id!r} has no model_dir")
            if not spec.model_id:
                spec = ModelSpec(
                    model_id=Path(spec.model_dir).name,
                    model_dir=spec.model_dir,
                    adapter=spec.adapter,
                    tier=spec.tier,
                    layers=spec.layers,
                    aliases=spec.aliases,
                )
            specs.append(spec)
    else:
        if model_dir is None:
            raise ValueError("the tiny backend needs model_dir or a models list")
        specs.append(
            ModelSpec(
                model_id=model_id or Path(model_dir).name,
                model_dir=model_dir,
                adapter=adapter,
                tier=tier,
                layers=tuple(layers) if layers is not None else None,
            )
        )
    seen: set[str] = set()
    for spec in specs:
        if spec.model_id in seen:
            raise ValueError(f"duplicate model id {spec.model_id!r}")
        seen.add(spec.model_id)
        resolve_tier(spec.tier)  # fail at startup, not on the first request
    return specs


def build_backend(
    kind: str = "mock",
    *,
    model_dir: str | Path | None = None,
    layers: Sequence[int] | None = None,
    device: str = "auto",
    model_id: str | None = None,
    script: Sequence[str] | None = None,
    token_delay: float = 0.0,
    tier: str = "conservative",
    adapter: str = "auto",
    models: Sequence[ModelSpec | Mapping[str, Any]] | None = None,
    default_model: str | None = None,
) -> Backend:
    """Create one of the built-in backends.

    kind="mock" -> MockBackend; kind="tiny" -> TinyModelBackend for one
    checkpoint, or a MultiModelBackend when several specs are supplied.
    """
    if kind == "mock":
        return MockBackend(
            model_id=model_id or "tinyinfer-mock", script=script, token_delay=token_delay
        )
    if kind != "tiny":
        raise ValueError(f"unknown backend {kind!r} (choose from 'mock', 'tiny')")
    specs = _coerce_model_specs(
        models,
        model_dir=model_dir,
        layers=layers,
        tier=tier,
        adapter=adapter,
        model_id=model_id,
    )
    entries: list[ModelEntry] = []
    for spec in specs:
        hint = (
            detect_adapter_kind(spec.model_dir)
            if spec.adapter in ("auto", "", None)
            else spec.adapter
        )
        if hint == "hf_qwen3_5":
            # Decided here, from the checkpoint: a client names a model id, the engine
            # picks the runtime path.
            return QwenBackend(
                spec.model_dir,
                model_id=spec.model_id or Path(spec.model_dir).name,
            )
        supported, note = _adapter_support(hint, spec.model_dir)

        def factory(spec: ModelSpec = spec, device: str = device):
            return TinyModelBackend(
                model_dir=spec.model_dir,
                adapter=spec.adapter,
                layers=spec.layers,
                tier=spec.tier,
                device=device,
                model_id=spec.model_id,
                aliases=spec.aliases,
            )

        entries.append(
            ModelEntry(
                model_id=spec.model_id,
                factory=factory,
                adapter=hint,
                tier=spec.tier,
                layers=spec.layers,
                model_dir=str(spec.model_dir) if spec.model_dir else None,
                aliases=spec.aliases,
                supported=supported,
                note=note,
            )
        )
    if len(entries) == 1:
        return entries[0].factory()
    return MultiModelBackend(entries, default_model=default_model or entries[0].model_id)


def models_from_args(args: argparse.Namespace) -> list[ModelSpec]:
    """Build model specs from server/CLI arguments (--model-dir / --extra-model)."""
    specs: list[ModelSpec] = []
    if getattr(args, "model_dir", None):
        specs.append(
            ModelSpec(
                model_id=getattr(args, "model_id", None) or Path(args.model_dir).name,
                model_dir=args.model_dir,
                adapter=getattr(args, "adapter", "auto"),
                tier=getattr(args, "tier", "conservative"),
                layers=parse_layers(getattr(args, "layers", None)),
            )
        )
    for item in getattr(args, "extra_model", None) or []:
        name, separator, directory = str(item).partition("=")
        if not separator or not name.strip() or not directory.strip():
            raise ValueError(f"--extra-model expects NAME=DIR, got {item!r}")
        specs.append(
            ModelSpec(
                model_id=name.strip(),
                model_dir=Path(directory.strip()),
                adapter=getattr(args, "adapter", "auto"),
                tier=getattr(args, "tier", "conservative"),
                layers=parse_layers(getattr(args, "layers", None)),
            )
        )
    if not specs:
        raise ValueError("no model configured: pass --model-dir or --extra-model NAME=DIR")
    return specs




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
        presence_penalty=float(body.presence_penalty),
        repetition_penalty=float(body.repetition_penalty),
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
    limit = settings.max_context_tokens
    resolver = getattr(backend, "context_limit", None)
    if callable(resolver):
        try:
            resolved = resolver(request)
        except BackendError as exc:
            raise exc.to_api_error() from exc
        if resolved:
            limit = min(limit, int(resolved))
    try:
        prompt_tokens = int(backend.count_prompt_tokens(request))
    except BackendError as exc:
        raise exc.to_api_error() from exc
    request.prompt_tokens = prompt_tokens
    if prompt_tokens >= limit:
        raise APIError(
            f"prompt is too long: {prompt_tokens} tokens, "
            f"the context is {limit}",
            status_code=400,
            error_type="invalid_request_error",
            param=param,
            code="context_length_exceeded",
        )
    room = limit - prompt_tokens
    if request.sampling.max_tokens > room:
        if settings.fit_max_tokens and room >= 1:
            request.sampling.max_tokens = room
        else:
            raise APIError(
                f"max_tokens ({request.sampling.max_tokens}) plus prompt tokens "
                f"({prompt_tokens}) exceeds the context "
                f"({limit}); only {room} tokens are left",
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
    entries: list[Any] | None = None
    lister = getattr(backend, "list_models", None)
    if callable(lister):
        try:
            entries = list(lister())
        except Exception:  # pragma: no cover - defensive
            logger.exception("backend list_models failed")
    if not entries:
        entries = [{"id": backend.model_id}]
        for alias in getattr(backend, "aliases", ()):
            if alias and alias != backend.model_id:
                entries.append({"id": alias, "alias_of": backend.model_id})
    objects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in entries:
        if isinstance(raw, str):
            raw = {"id": raw}
        if not isinstance(raw, Mapping):
            continue
        model_id = raw.get("id")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            continue
        seen.add(model_id)
        context = raw.get("context_length")
        try:
            context_length = int(context) if context is not None else settings.max_context_tokens
        except (TypeError, ValueError):
            context_length = settings.max_context_tokens
        entry: dict[str, Any] = {
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": "tinyinfer",
            "context_length": context_length,
            "max_output_tokens": settings.max_tokens_limit,
        }
        for key in (
            "adapter",
            "tier",
            "layers",
            "model_dir",
            "loaded",
            "supported",
            "note",
            "alias_of",
            "aliases",
        ):
            value = raw.get(key)
            if value is None:
                continue
            if isinstance(value, tuple):
                value = list(value)
            entry[key] = value
        objects.append(entry)
    return objects




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


def _tool_function_names(tools: Sequence[Mapping[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _extract_tool_call(text: str, tool_names: set[str]) -> dict[str, str] | None:
    """Recognise a minimal function call written as JSON by the model.

    Supports a plain JSON object, a fenced JSON block and the
    <tool_call>{...}</tool_call> wrapper.  The name must be declared in the
    request (when tools are given), otherwise the text is left untouched.
    """
    if not text:
        return None
    candidate = text.strip()
    wrapper = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", candidate, re.S)
    if wrapper:
        candidate = wrapper.group(1)
    else:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S)
        if fenced:
            candidate = fenced.group(1)
    data = None
    try:
        data = json.loads(candidate)
    except Exception:
        for match in re.finditer(r"\{", candidate):
            start = match.start()
            for end in range(len(candidate), start, -1):
                try:
                    data = json.loads(candidate[start:end])
                    break
                except Exception:
                    continue
            if isinstance(data, dict):
                break
            data = None
    if not isinstance(data, dict):
        return None
    inner = data.get("tool_call")
    if isinstance(inner, Mapping):
        data = dict(inner)
    function = data.get("function")
    name = data.get("name")
    if name is None and isinstance(function, Mapping):
        name = function.get("name")
    arguments = data.get("arguments")
    if arguments is None and isinstance(function, Mapping):
        arguments = function.get("arguments")
    if arguments is None:
        arguments = data.get("parameters", {})
    if not isinstance(name, str) or not name:
        return None
    if tool_names and name not in tool_names:
        return None
    if isinstance(arguments, str):
        arguments_text = arguments
    else:
        try:
            arguments_text = json.dumps(arguments, ensure_ascii=False)
        except TypeError:  # pragma: no cover - unusual argument objects
            arguments_text = json.dumps(str(arguments))
    return {"name": name, "arguments": arguments_text}


def _tool_call_payload(call: Mapping[str, str]) -> dict[str, Any]:
    return {
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": call["name"], "arguments": call["arguments"]},
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
    tool_names: Sequence[str] = (),
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
        buffering = kind == "chat" and bool(tool_names)
        buffered: list[str] = []
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
                        if buffering:
                            buffered.append(event.text)
                        elif kind == "chat":
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
                if buffering:
                    text = "".join(buffered)
                    call = _extract_tool_call(text, set(tool_names))
                    if call:
                        finish_reason = "tool_calls"
                        payload = _tool_call_payload(call)
                        yield frame(
                            _chat_chunk(
                                completion_id,
                                created,
                                model,
                                {"tool_calls": [{"index": 0, **payload}]},
                                None,
                            )
                        )
                    elif text:
                        yield frame(
                            _chat_chunk(
                                completion_id,
                                created,
                                model,
                                {"content": text},
                                None,
                            )
                        )
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
        models = _model_objects(app)
        return {
            "status": "ok",
            "model": backend.model_id,
            "models": [entry["id"] for entry in models],
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
        tool_names: set[str] = set()
        if body.tool_choice != "none":
            tool_names = _tool_function_names(body.tools)
        if body.tools:
            logger.debug(
                "tools declared for %s: %s (tool_choice=%r)",
                model,
                sorted(tool_names),
                body.tool_choice,
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
                tool_names=sorted(tool_names),
            )
        text, finish_reason, usage = _run_generation(app_, generation_request)
        tool_call = _extract_tool_call(text, tool_names) if tool_names else None
        if tool_call is not None:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call_payload(tool_call)],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": text}
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
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
        "--model-dir", default=None, help="HF safetensors directory of the default model"
    )
    parser.add_argument(
        "--extra-model",
        action="append",
        default=None,
        metavar="NAME=DIR",
        help="register an additional model (repeatable)",
    )
    parser.add_argument(
        "--default-model", default=None, help="model id used for unknown model names"
    )
    parser.add_argument(
        "--layers", default=None, help="comma separated layer ids for staged loading"
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--model-id", default=None, help="model id of the default model")
    parser.add_argument(
        "--adapter",
        default="auto",
        choices=("auto", "hf_deepseek_v41", "gguf"),
        help="checkpoint adapter (GGUF is reserved)",
    )
    parser.add_argument(
        "--tier",
        default="aggressive",
        choices=("conservative", "balanced", "aggressive"),
        help="optimization tier for the reference path (default: balanced, i.e. the first",
    )
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
        help="context window cap",
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
    if args.backend == "mock":
        return build_backend("mock", model_id=args.model_id)
    specs = models_from_args(args)
    return build_backend(
        "tiny",
        models=specs,
        tier=args.tier,
        adapter=args.adapter,
        default_model=args.default_model,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.backend == "tiny" and not args.model_dir and not args.extra_model:
        parser.error("--model-dir or --extra-model is required with --backend tiny")
    api_key = args.api_key or os.environ.get("TINYINFER_API_KEY")
    try:
        backend = _backend_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
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
        "OpenAI-compatible API at http://%s:%s/v1  (backend=%s, tier=%s, model=%s)",
        args.host,
        args.port,
        args.backend,
        args.tier,
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
