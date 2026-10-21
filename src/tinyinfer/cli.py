"""Command line interface for tinyinfer: serve the API, generate locally.

Usage::

    python -m tinyinfer.cli serve --backend mock --port 8000
    python -m tinyinfer.cli generate "hello tinyinfer" --max-tokens 24
    python -m tinyinfer.cli generate --backend tiny --model-dir DIR --layers 0,1

The CLI is a thin wrapper around tinyinfer.server: the same backends, slots,
sampling and streaming code paths are used, so it is a useful sanity check
before pointing an agent at the HTTP API.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

__all__ = ["build_parser", "main"]


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("backend")
    group.add_argument("--backend", choices=("mock", "tiny"), default="mock", help="model backend")
    group.add_argument("--model-dir", default=None, help="HF safetensors directory (backend=tiny)")
    group.add_argument(
        "--extra-model",
        action="append",
        default=None,
        metavar="NAME=DIR",
        help="register an additional model (repeatable)",
    )
    group.add_argument("--default-model", default=None, help="model id for unknown model names")
    group.add_argument("--layers", default=None, help="comma separated layer ids (staged loading)")
    group.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    group.add_argument("--model-id", default=None, help="model id reported by /v1/models")
    group.add_argument(
        "--tier",
        default="conservative",
        choices=("conservative", "balanced"),
        help="optimization tier (conservative=reference, balanced=ExpertStore experts)",
    )
    group.add_argument(
        "--adapter",
        default="auto",
        choices=("auto", "hf_deepseek_v41", "gguf"),
        help="checkpoint adapter (GGUF is reserved)",
    )


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("sampling")
    group.add_argument("--max-tokens", type=int, default=64, help="output tokens")
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--stop", action="append", default=None, help="stop sequence (repeatable)")
    group.add_argument("--seed", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyinfer",
        description="tinyinfer command line: serve an OpenAI-compatible API or generate locally.",
    )
    try:
        from . import __version__

        version = __version__
    except Exception:  # pragma: no cover
        version = "0.1.0"
    parser.add_argument("--version", action="version", version=f"tinyinfer {version}")
    subparsers = parser.add_subparsers(dest="command", metavar="{serve,generate}")

    serve = subparsers.add_parser(
        "serve",
        help="start the OpenAI-compatible HTTP API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--api-key", default=None, help="comma separated keys (or TINYINFER_API_KEY)")
    serve.add_argument("--max-concurrency", type=int, default=4)
    serve.add_argument("--request-timeout", type=float, default=600.0)
    serve.add_argument("--max-slots", type=int, default=4)
    serve.add_argument("--max-context-tokens", type=int, default=1_048_576)
    serve.add_argument("--max-tokens-limit", type=int, default=1_048_576)
    serve.add_argument("--default-max-tokens", type=int, default=256)
    serve.add_argument("--fit-max-tokens", action="store_true")
    serve.add_argument("--sse-keepalive", type=float, default=15.0)
    serve.add_argument(
        "--log-level", default="info", choices=("debug", "info", "warning", "error")
    )
    _add_backend_args(serve)

    generate = subparsers.add_parser(
        "generate",
        help="generate text locally through a backend (no HTTP server)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate.add_argument("prompt", nargs="?", default=None, help="prompt text (default: read stdin)")
    generate.add_argument("--system", default=None, help="system message (chat mode)")
    generate.add_argument("--model", default=None, help="registered model id to route to")
    generate.add_argument("--stream", action="store_true", help="print tokens as they arrive")
    generate.add_argument("--json", action="store_true", help="print a JSON result with usage")
    _add_sampling_args(generate)
    _add_backend_args(generate)
    return parser


def _build_backend(args: argparse.Namespace):
    from .server import build_backend, models_from_args

    if args.backend == "mock":
        return build_backend("mock", model_id=args.model_id)
    return build_backend(
        "tiny",
        models=models_from_args(args),
        tier=args.tier,
        adapter=args.adapter,
        default_model=args.default_model,
        device=args.device,
    )


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import BackendError, create_app

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.backend == "tiny" and not args.model_dir and not args.extra_model:
        print(
            "error: --model-dir or --extra-model NAME=DIR is required with --backend tiny",
            file=sys.stderr,
        )
        return 2
    backend = _build_backend(args)
    try:
        backend.warmup()
    except BackendError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
    app = create_app(
        backend,
        api_key=args.api_key,
        max_concurrency=args.max_concurrency,
        request_timeout=args.request_timeout,
        max_slots=args.max_slots,
        max_context_tokens=args.max_context_tokens,
        max_tokens_limit=args.max_tokens_limit,
        default_max_tokens=args.default_max_tokens,
        fit_max_tokens=args.fit_max_tokens,
        sse_keepalive_s=args.sse_keepalive,
    )
    print(
        f"tinyinfer: OpenAI-compatible API on http://{args.host}:{args.port}/v1 "
        f"(backend={args.backend}, tier={args.tier}, model={backend.model_id})"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    import time

    from .server import (
        BackendError,
        GenerationRequest,
        SamplingParams,
        Usage,
    )

    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read()
    if not prompt:
        print("error: empty prompt (pass it as an argument or on stdin)", file=sys.stderr)
        return 2
    backend = _build_backend(args)
    try:
        backend.warmup()
    except BackendError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=tuple(args.stop or ()),
        seed=args.seed,
    )
    if args.system:
        request = GenerationRequest(
            request_id="cli",
            messages=[
                {"role": "system", "content": args.system},
                {"role": "user", "content": prompt},
            ],
            sampling=sampling,
            stream=args.stream,
            model=args.model,
        )
    else:
        request = GenerationRequest(
            request_id="cli",
            prompt=prompt,
            sampling=sampling,
            stream=args.stream,
            model=args.model,
        )

    started = time.perf_counter()
    parts: list[str] = []
    finish_reason: str | None = None
    usage: Usage | None = None
    try:
        for event in backend.generate(request):
            if event.finish_reason is not None:
                finish_reason = event.finish_reason
                usage = event.usage
                break
            if event.text:
                parts.append(event.text)
                if args.stream:
                    sys.stdout.write(event.text)
                    sys.stdout.flush()
    except BackendError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
    finally:
        backend.close()
    elapsed = time.perf_counter() - started
    text = "".join(parts)
    usage_dict = usage.to_dict() if usage is not None else None
    if args.json:
        print(
            json.dumps(
                {
                    "text": text,
                    "finish_reason": finish_reason,
                    "usage": usage_dict,
                    "elapsed_s": round(elapsed, 3),
                },
                ensure_ascii=False,
            )
        )
    else:
        if args.stream:
            print()
        else:
            print(text)
        print(
            f"[finish_reason={finish_reason} usage={usage_dict} elapsed={elapsed:.2f}s]",
            file=sys.stderr,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # reduce streaming-layer allocator fragmentation (measured +14% decode on 8GB)
    import os
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "generate":
        return _cmd_generate(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
