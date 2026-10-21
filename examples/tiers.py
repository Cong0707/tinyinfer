"""Tier comparison: run the same prompt at two optimisation levels.

    PYTHONPATH=src python examples/tiers.py --model <checkpoint dir>
"""
import argparse
import time

from transformers import AutoTokenizer

from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok.encode(args.prompt, add_special_tokens=False)
    for level in (OptLevel.BALANCED, OptLevel.AGGRESSIVE):
        eng = Engine.from_hf(args.model, device="cuda", tokenizer=tok, tier=level,
                             allow_missing_layers=True)
        t0 = time.perf_counter()
        out = eng.generate(ids, max_new_tokens=8)
        dt = time.perf_counter() - t0
        print(f"tier {level.name}: {tok.decode(out, skip_special_tokens=False)!r} "
              f"({8 / dt:.3f} tok/s)")


if __name__ == "__main__":
    main()
