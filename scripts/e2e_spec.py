"""Speculative decoding smoke test: 40 layers + MTP draft/verify, report accept rate.

Usage: python scripts/e2e_spec.py [model_dir] [n_tokens]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/Models/DeepSeek-V4.1-Flash-1017")
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    if not (model_dir / "model.safetensors.index.json").exists():
        model_dir = Path("E:/Develop/Deepseek/tinyinfer/cache/weights")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyinfer.engine import Engine

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as e:
        print("tokenizer unavailable:", e)

    eng = Engine.from_hf(model_dir, device="cuda", tokenizer=tok, max_layers=40, allow_missing_layers=True)
    print("blocks:", eng.max_layers, "gpu_layers:", eng.gpu_layers, flush=True)
    print("load_mtp:", eng.load_mtp(), flush=True)
    prompt = tok.encode("Hello") if tok is not None else [1, 2, 3]
    t0 = time.time()
    out, rate = eng.speculative_generate(prompt, max_new_tokens=n_tokens, draft_len=2, temperature=0.0)
    torch.cuda.synchronize()
    dt = time.time() - t0
    text = tok.decode(out) if tok is not None else str(out)
    print(f"generated: {text!r}", flush=True)
    print(f"tokens={len(out)} time={dt:.2f}s = {len(out)/dt:.3f} tok/s  accept_rate={rate:.2%}", flush=True)
    vram = torch.cuda.max_memory_allocated() / (1 << 20)
    print(f"peak vram: {vram:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
