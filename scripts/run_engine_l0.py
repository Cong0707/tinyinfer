"""End-to-end engine smoke test on the single downloaded layer (L0)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

MODEL = ROOT / "cache" / "weights"


def main():
    engine = Engine(str(MODEL), tier=OptLevel.CONSERVATIVE, allow_missing_layers=True)
    print("cfg:", engine.cfg.summary())
    print("available layers:", engine.available_layers())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    prompt = "Hello, I am DeepSeek."
    ids = tok(prompt).input_ids
    print("prompt:", repr(prompt), "tokens:", len(ids))
    gen = engine.generate(ids, max_new_tokens=4, temperature=0.0)
    print("generated ids:", gen)
    print("text:", repr(tok.decode(ids + gen)))
    print("stats:", engine.stats())


if __name__ == "__main__":
    main()
