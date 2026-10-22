"""Multi-turn chat that does not re-prefill the conversation.

A prefill of a 23-token conversation costs ~43s on the 8GB box and produces a 16.9MB
checkpoint that restores in ~0.01s. So each turn only prefills the new tokens: save after
the first turn, restore at the start of the next.

    PYTHONPATH=src python examples/multi_turn.py --model E:\Models\DeepSeek-V4.1-Flash-1017

Tier 2 (the default) owns the checkpoint mechanism; tier 1 refuses with a clear message.
"""
import argparse
import os
import tempfile

from transformers import AutoTokenizer

from tinyinfer.engine import Engine

QUESTIONS = [
    "What is the capital of France?",
    "What is its population?",
    "Who wrote Hamlet?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eng = Engine.from_hf(args.model, device="cuda", tokenizer=tok)
    bos = "<｜begin▁of▁sentence｜>"
    ckpt = os.path.join(tempfile.gettempdir(), "tinyinfer_prefix.pt")

    # turn 1: prefill the opening question and answer, then checkpoint the prefix
    text = bos + "Question: " + QUESTIONS[0] + "\nAnswer:"
    ids = tok.encode(text, add_special_tokens=False)
    out = eng.generate(ids, max_new_tokens=args.max_new_tokens)
    answer = tok.decode(out, skip_special_tokens=True)
    print("Q:", QUESTIONS[0])
    print("A:", answer.strip() or answer)
    text += answer
    eng.save_checkpoint(ckpt)
    print("saved prefix checkpoint:", ckpt, flush=True)

    # later turns: restore the prefix, prefill only the new text
    for q in QUESTIONS[1:]:
        eng.load_checkpoint(ckpt)
        new = "\nQuestion: " + q + "\nAnswer:"
        new_ids = tok.encode(new, add_special_tokens=False)
        start = len(tok.encode(text, add_special_tokens=False))
        out = eng.generate(new_ids, max_new_tokens=args.max_new_tokens,
                           start_pos=start, keep_session=True)
        ans = tok.decode(out, skip_special_tokens=True)
        print("Q:", q)
        print("A:", ans.strip() or ans)
        text += new + ans

    os.remove(ckpt)


if __name__ == "__main__":
    main()
