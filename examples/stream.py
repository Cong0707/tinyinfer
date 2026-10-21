"""Streaming example: print tokens as they are produced.

    PYTHONPATH=src python examples/stream.py --model <checkpoint dir> --prompt "Once upon a time"
"""
import argparse

from transformers import AutoTokenizer

from tinyinfer.engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--new-tokens", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eng = Engine.from_hf(args.model, device="cuda", tokenizer=tok, allow_missing_layers=True)
    ids = tok.encode(args.prompt, add_special_tokens=False)
    for nid in eng.stream_generate(ids, max_new_tokens=args.new_tokens):
        print(tok.decode([nid], skip_special_tokens=False), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
