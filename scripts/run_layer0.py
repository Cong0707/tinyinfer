"""P2: real-weights layer-0 forward -> logits, with finite/shape checks."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.config import TinyConfig
from tinyinfer.model import TinyModel
from tinyinfer.weights import ShardStore

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "cache" / "weights")
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 8

cfg = TinyConfig.from_hf(MODEL)
print("config:", cfg.summary())

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
text = "Hello, I am DeepSeek. How are you?"
ids = tok(text).input_ids[:MAXTOK]
print("input:", ids)

torch.manual_seed(0)
with ShardStore(MODEL) as st:
    model = TinyModel(st, cfg, layers=[0])
    logits = model.forward(torch.tensor([ids], dtype=torch.long))
    print("logits:", tuple(logits.shape), logits.dtype)
    print("finite:", torch.isfinite(logits).all().item(),
          "| mean/std:", f"{logits.mean():.4f}/{logits.std():.4f}")
    top = torch.topk(logits[0, -1], 10)
    print("top10:", [(tok.decode([int(i)]), round(float(v), 2)) for i, v in zip(top.indices, top.values)])
    print("last input token:", tok.decode([ids[-1]]))
