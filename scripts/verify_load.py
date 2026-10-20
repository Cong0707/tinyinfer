"""Verify direct HF loading: embed -> RMSNorm -> lm_head (zero-layer forward)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "cache" / "weights")

cfg = TinyConfig.from_hf(MODEL)
print("config:", cfg.summary())
print("engram rows:", cfg.engram_num_embeddings, "kv_src:", cfg.kv_source_layer_ids,
      "compress:", cfg.compress_ratios[:6], "...")

with ShardStore(MODEL) as st:
    emb = st.get("embed.weight")
    head = st.get("head.weight")
    norm_w = st.get("norm.weight")
    print("embed:", tuple(emb.shape), emb.dtype)
    print("head :", tuple(head.shape), head.dtype)
    print("norm :", tuple(norm_w.shape), norm_w.dtype)
    print("shards available:", len(st.shard_files()))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    text = "Hello, I am DeepSeek."
    ids = tok(text).input_ids
    print("text:", text)
    print("tokens:", ids)
    x = emb[torch.tensor(ids)].float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    x = x * norm_w.float()
    logits = x @ head.float().T
    top = torch.topk(logits[-1], 12)
    out = [(tok.decode([int(i)]), round(float(v), 2)) for i, v in zip(top.indices, top.values)]
    print("next-token top12:", out)
