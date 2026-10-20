"""P2: greedy generation from a stack of real HF layers (layer 0 first, more as shards land)."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.config import TinyConfig
from tinyinfer.model import TinyModel
from tinyinfer.nn import softmax_sample
from tinyinfer.weights import ShardStore

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "cache" / "weights")
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "Hello, I am DeepSeek."
N_GEN = int(sys.argv[3]) if len(sys.argv) > 3 else 8
LAYERS = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["0"])]

cfg = TinyConfig.from_hf(MODEL)
print("config:", cfg.summary())
print("layers:", LAYERS)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
ids = tok(PROMPT).input_ids
print("prompt:", repr(PROMPT), "| tokens:", ids)

torch.manual_seed(0)
t0 = time.time()
with ShardStore(MODEL) as st:
    model = TinyModel(st, cfg, layers=LAYERS)
    print(f"model built in {time.time()-t0:.1f}s")
    with torch.inference_mode():
        logits = model.forward(torch.tensor([ids]), 0)
        print(f"prefill done: logits {tuple(logits.shape)} in {time.time()-t0:.1f}s")
        generated = []
        for step in range(N_GEN):
            nid = int(softmax_sample(logits[0, -1], temperature=0.0))
            generated.append(nid)
            print(f"  step {step}: id={nid} text={tok.decode([nid])!r}")
            if nid in (tok.eos_token_id,) if tok.eos_token_id is not None else ():
                break
            logits = model.forward(torch.tensor([[nid]]), len(ids) + step)
        print("OUTPUT:", repr(tok.decode(ids + generated)))
