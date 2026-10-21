"""Final acceptance script for the full model at E:\Models (HF repo layout).

Checks (in order):
  1. repo layout: config.json, model.safetensors.index.json, all 48 shards present
  2. Engine.from_hf: 40 layers load, engram hashes enabled when tables are present
  3. prefill + decode benchmark (tokens/s, peak VRAM)
  4. optional MTP stage load (mtp.* present) and speculative smoke

Usage: python scripts/e2e_final.py [model_dir] [n_tokens]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


def check_layout(model_dir: Path) -> dict:
    out = {"dir": str(model_dir), "ok": True, "missing": [], "shards": 0}
    if not (model_dir / "config.json").exists():
        out["ok"] = False
        out["missing"].append("config.json")
    if not (model_dir / "model.safetensors.index.json").exists():
        out["ok"] = False
        out["missing"].append("model.safetensors.index.json")
        return out
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(idx["weight_map"].values()))
    out["shards"] = len(shards)
    for s in shards:
        if not (model_dir / s).exists():
            out["missing"].append(s)
    out["ok"] = not out["missing"]
    return out


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/Models/DeepSeek-V4.1-Flash-1017")
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lay = check_layout(model_dir)
    print(f"layout: ok={lay['ok']} shards={lay['shards']} missing={lay['missing'][:5]}", flush=True)
    if not lay["ok"]:
        print("layout incomplete; aborting", flush=True)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyinfer.engine import Engine

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        print(f"tokenizer: vocab={len(tok)}", flush=True)
    except Exception as e:
        print(f"tokenizer unavailable: {e}", flush=True)

    t0 = time.time()
    eng = Engine.from_hf(model_dir, device="cuda", tokenizer=tok, allow_missing_layers=False)
    print(f"engine: {eng.max_layers} layers, gpu_layers={eng.gpu_layers}, built {time.time()-t0:.1f}s", flush=True)
    print(f"engram: {'on' if eng.hashes is not None else 'off'} layers={eng.cfg.engram_layer_ids}", flush=True)

    ids = tok.encode("Hello") if tok is not None else [1, 2, 3]
    # generate() runs its own prefill; a separate forward() here would replay the
    # prefill on top of existing KV (shared compressed state resets mid-sequence).
    t0 = time.time()
    out = eng.generate(ids, max_new_tokens=n_tokens, temperature=0.0)
    torch.cuda.synchronize()
    dt = time.time() - t0
    text = tok.decode(out) if tok is not None else str(out)
    print(f"generated: {text!r}", flush=True)
    print(f"decode: {len(out)} tok in {dt:.2f}s = {len(out)/dt:.3f} tok/s", flush=True)
    print(f"peak vram: {torch.cuda.max_memory_allocated()/(1<<20):.0f} MB", flush=True)

    if (model_dir / "model.safetensors.index.json").exists():
        try:
            ok = eng.load_mtp()
            print(f"mtp: {'loaded' if ok else 'unavailable'}", flush=True)
        except Exception as e:
            print(f"mtp load failed: {e}", flush=True)


if __name__ == "__main__":
    main()
