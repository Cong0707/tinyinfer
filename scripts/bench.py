"""Production benchmark: prefill/decode throughput, TTFT and memory for tinyinfer.

Usage:
    python scripts/bench.py --model-dir cache/weights --layers 0-5
    python scripts/bench.py --model-dir cache/weights --max-layers 6 --device cuda --tier balanced
    python scripts/bench.py --prompt-tokens 128 --decode 16 --gpu-layers 33
"""
from __future__ import annotations

import argparse
import faulthandler
import statistics
import sys
import time
from pathlib import Path

faulthandler.enable()  # print the Python stack on a native crash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel


def parse_layers(text: str | None) -> list[int] | None:
    if not text:
        return None
    layers: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return layers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "cache" / "weights"))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--tier", default="conservative", choices=[l.name.lower() for l in OptLevel])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpu-layers", type=int, default=None)
    ap.add_argument("--head-cpu", action="store_true")
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = parse_layers(args.layers)
    engine = Engine(
        args.model_dir,
        tier=OptLevel[args.tier.upper()],
        device=device,
        layers=layers,
        max_layers=args.max_layers,
        gpu_layers=args.gpu_layers,
        head_on_cpu=args.head_cpu,
        allow_missing_layers=True,
    )
    n = len(layers) if layers else (args.max_layers or engine.cfg.n_layers)
    print(f"tier={args.tier} device={device} layers={n} available={engine.available_layers()}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    # a synthetic prompt of roughly the requested length
    base = tok("The quick brown fox jumps over the lazy dog. " * 8).input_ids
    prompt = (base * ((args.prompt_tokens + len(base) - 1) // len(base)))[: args.prompt_tokens]

    # warmup: prefill + one decode step (compiles both the M=T and M=1 kernel variants)
    with torch.inference_mode():
        engine.reset()
        warm_logits = engine.forward(torch.tensor([prompt]), 0)
        warm_nid = int(warm_logits[0, -1].argmax())
        engine.forward(torch.tensor([[warm_nid]]), len(prompt))

    # prefill
    ttfts, decodes = [], []
    for _ in range(args.repeat):
        engine.reset()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = engine.forward(torch.tensor([prompt]), 0)
        torch.cuda.synchronize() if device == "cuda" else None
        ttfts.append((time.perf_counter() - t0) * 1e3)
        if device == "cuda":
            torch.cuda.empty_cache()  # drop prefill allocator slabs before decode

        # decode
        pos = len(prompt)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _step in range(args.decode):
                nid = int(logits[0, -1].argmax())
                logits = engine.forward(torch.tensor([[nid]]), pos)
                pos += 1
        torch.cuda.synchronize() if device == "cuda" else None
        decodes.append((time.perf_counter() - t0) * 1e3)

    ttft = statistics.median(ttfts)
    dec = statistics.median(decodes)
    print(f"prefill: {args.prompt_tokens} tok in {ttft:.0f} ms "
          f"({args.prompt_tokens / (ttft / 1e3):.1f} tok/s)")
    print(f"decode:  {args.decode} tok in {dec:.0f} ms "
          f"({args.decode / (dec / 1e3):.2f} tok/s)")
    print("stats:", engine.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
