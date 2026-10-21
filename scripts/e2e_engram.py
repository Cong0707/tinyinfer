"""Engram L1 row-read smoke: hash lookup + table row decode + forward on the L1 table.

L14 (00048) is still downloading; this test uses layers.1.engram.* (00047) to prove
the row-read path and the gated residual write end to end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/Develop/Deepseek/tinyinfer/cache/weights")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from transformers import AutoTokenizer
    from tinyinfer.config import TinyConfig
    from tinyinfer.engram import Engram, EngramHashes
    from tinyinfer.weights import ShardStore

    cfg = TinyConfig.from_hf(model_dir)
    store = ShardStore(model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    print("tokenizer vocab:", len(tok), flush=True)
    hashes = EngramHashes(cfg, tok)
    print("hashes ok, n_cols:", hashes.n_cols, "layers:", hashes.layer_ids, flush=True)

    ids = tok.encode("Hello") if hasattr(tok, "encode") else [19923]
    h = hashes(ids, 0)
    print("hash out:", h.shape, "layer1 sample:", h[0].reshape(-1)[:4].tolist(), flush=True)

    layer_id = cfg.engram_layer_ids[0]  # L1 = 00047
    t0 = __import__("time").time()
    eng = Engram(store, layer_id, cfg, hashes)
    eng.to("cuda")
    print(f"Engram L{layer_id} built in {__import__('time').time()-t0:.1f}s", flush=True)
    x = torch.zeros(1, len(ids), cfg.hc_mult, cfg.dim, dtype=torch.bfloat16, device="cuda")
    t0 = __import__("time").time()
    out = eng.forward(x, h[eng.hash_index])
    torch.cuda.synchronize()
    print(f"forward ok: {tuple(out.shape)} in {__import__('time').time()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
