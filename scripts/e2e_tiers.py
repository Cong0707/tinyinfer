"""Tier comparison: same prompt under BALANCED / AGGRESSIVE(top-5) / EXTREME(top-4).

Prints the generated text so we can judge whether the aggressive tiers stay coherent
(they must not degenerate into garbage even though they read fewer experts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def run(tier: str, model_dir: Path, tok) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyinfer.engine import Engine
    from tinyinfer.tiers import OptLevel

    lvl = OptLevel[tier]
    eng = Engine.from_hf(model_dir, device="cuda", tokenizer=tok, allow_missing_layers=True, tier=lvl)
    ids = tok.encode("The quick brown fox") if tok is not None else [1, 2, 3, 4, 5]
    out = eng.generate(ids, max_new_tokens=8, temperature=0.0)
    text = tok.decode(out) if tok is not None else str(out)
    print(f"[{tier}] topk={eng.tier.expert_topk or 'default'} -> {text!r}", flush=True)
    del eng
    torch.cuda.empty_cache()


def main() -> None:
    model_dir = Path("E:/Develop/Deepseek/tinyinfer/cache/weights")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    for tier in ("BALANCED", "AGGRESSIVE", "EXTREME"):
        try:
            run(tier, model_dir, tok)
        except Exception as e:
            print(f"[{tier}] failed: {e}", flush=True)


if __name__ == "__main__":
    main()
