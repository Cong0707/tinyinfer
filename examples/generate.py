"""Minimal end-to-end example: load the checkpoint and generate text.

    PYTHONPATH=src python examples/generate.py --model <checkpoint dir>
"""
import argparse

import torch
from transformers import AutoTokenizer

from tinyinfer.engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--new-tokens", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eng = Engine.from_hf(args.model, device="cuda", tokenizer=tok, allow_missing_layers=True)
    ids = tok.encode(args.prompt, add_special_tokens=False)
    out = eng.generate(ids, max_new_tokens=args.new_tokens)
    print(tok.decode(out, skip_special_tokens=False))


if __name__ == "__main__":
    main()
