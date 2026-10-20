"""P2: run layer-0 attention with real HF weights on a tiny random input."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.attention import MLAWindowAttention
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "cache" / "weights")
cfg = TinyConfig.from_hf(MODEL)
print("config:", cfg.summary())

torch.manual_seed(0)
with ShardStore(MODEL) as st:
    attn = MLAWindowAttention(st, 0, cfg)
    print("weights loaded: wq_a", tuple(attn.wq_a.shape), attn.wq_a.dtype,
          "| wq_b", tuple(attn.wq_b.shape), "| wo_a", tuple(attn.wo_a.shape))
    x = torch.randn(1, 6, cfg.dim, dtype=torch.bfloat16) * 0.02
    y = attn.forward(x)
    print("out:", tuple(y.shape), y.dtype)
    print(f"stats: mean={y.float().mean():.5f} std={y.float().std():.5f} "
          f"finite={torch.isfinite(y.float()).all().item()}")
    # sanity: a zero input through a linear layer is not necessarily zero, but must be finite
    y2 = attn.forward(torch.zeros(1, 3, cfg.dim, dtype=torch.bfloat16))
    print("zero-input out finite:", torch.isfinite(y2.float()).all().item(),
          "std:", f"{y2.float().std():.5f}")
