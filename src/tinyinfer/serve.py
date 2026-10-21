"""Minimal HTTP server for tinyinfer (stdlib only, no extra dependencies).

One model per process; requests are serialised because the engine keeps a single
session (KV state) and the card cannot hold a second one. Endpoints:

    GET  /health          -> {"ok": true, "tier": ..., "device": ...}
    POST /generate        -> {"prompt": str | "prompt_ids": [int], "max_new_tokens": int,
                              "temperature": float, "stop": [str]}
                             {"text": str, "tokens": [...], "rate": tok/s}

Usage:
    python -m tinyinfer.serve --model E:\\Models\\DeepSeek-V4.1-Flash-1017 --port 8000
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_LOCK = threading.Lock()
_ENGINE = None
_TOKENIZER = None
_CONFIG = {"model": None, "tier": None}


def _build(engine_cls, model: str, tier: int | None, device: str):
    global _ENGINE, _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    kwargs = {"device": device, "tokenizer": _TOKENIZER, "allow_missing_layers": True}
    if tier is not None:
        kwargs["tier"] = tier
    _ENGINE = engine_cls.from_hf(model, **kwargs)
    _CONFIG["model"] = model
    _CONFIG["tier"] = str(getattr(_ENGINE, "tier", None))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ok": _ENGINE is not None,
                             "model": _CONFIG["model"],
                             "tier": _CONFIG["tier"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._send(400, {"error": f"bad request: {exc}"})
            return
        if _ENGINE is None:
            self._send(503, {"error": "model still loading"})
            return
        prompt = req.get("prompt")
        prompt_ids = req.get("prompt_ids")
        if prompt is None and prompt_ids is None:
            self._send(400, {"error": "prompt or prompt_ids required"})
            return
        max_new = int(req.get("max_new_tokens", 16))
        stop = req.get("stop") or []
        with _LOCK:  # one session at a time: the KV state is not shareable
            try:
                if prompt_ids is None:
                    prompt_ids = _TOKENIZER.encode(prompt, add_special_tokens=False)
                t0 = time.perf_counter()
                out = _ENGINE.generate(prompt_ids, max_new_tokens=max_new)
                dt = time.perf_counter() - t0
                text = _TOKENIZER.decode(out, skip_special_tokens=False)
                for marker in stop:
                    if marker and marker in text:
                        text = text.split(marker)[0]
            except Exception as exc:
                self._send(500, {"error": repr(exc)})
                return
        n = max(1, len(out))
        self._send(200, {"text": text, "tokens": out, "rate": n / dt if dt else None})


def main() -> None:
    ap = argparse.ArgumentParser(description="tinyinfer HTTP server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from .engine import Engine

    # build in the background so /health answers while the weights stream in
    th = threading.Thread(target=_build, args=(Engine, args.model, args.tier, args.device), daemon=True)
    th.start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[tinyinfer] serving on http://{args.host}:{args.port} (model loading in background)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
