"""End-to-end check against the HF repo layout under E:\\Models.

Usage:  python scripts/e2e_models.py [model_dir] [n_tokens]

Loads the full 40-layer stack through Engine.from_hf (repo/snapshot/flat dir all
work), reports engram/MTP availability, and generates a few tokens.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/Models/DeepSeek-V4.1-Flash-1017")
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyinfer.engine import Engine
    from tinyinfer.nn import softmax_sample

    print(f"model_dir={model_dir} exists={model_dir.exists()}", flush=True)
    if not (model_dir / "model.safetensors.index.json").exists():
        print("index.json missing; flat HF repo layout expected", flush=True)
        return

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        print(f"tokenizer ok: vocab={len(tok)}", flush=True)
    except Exception as e:
        print(f"tokenizer unavailable: {e}", flush=True)

    t0 = time.time()
    eng = Engine.from_hf(model_dir, device="cuda", tokenizer=tok, allow_missing_layers=True)
    print(f"engine built in {time.time()-t0:.1f}s layers={eng.max_layers} gpu_layers={eng.gpu_layers}", flush=True)
    print(f"engram hashes: {'on' if eng.hashes is not None else 'off'} engram_layers={eng.cfg.engram_layer_ids}", flush=True)

    ids = tok.encode("Hello") if tok is not None else [1, 2, 3]
    print(f"prompt ids={ids}", flush=True)
    t0 = time.time()
    out_ids = eng.generate(ids, max_new_tokens=n_tokens, temperature=0.0)
    torch.cuda.synchronize()
    dt = time.time() - t0
    text = tok.decode(out_ids) if tok is not None else str(out_ids)
    print(f"generated: {text!r}", flush=True)
    print(f"generate {len(out_ids)-len(ids)} tok in {dt:.2f}s = {(len(out_ids)-len(ids))/dt:.3f} tok/s", flush=True)
    vram = torch.cuda.max_memory_allocated() / (1 << 20) if torch.cuda.is_available() else 0
    print(f"peak vram: {vram:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
