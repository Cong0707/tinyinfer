"""Interactive chat against Qwen3.5 with the CUDA-graph decode path.

    python qwen_repl.py                 # interactive
    python qwen_repl.py "问题"           # one-shot

The model runs through tinyinfer's hand-written linear-attention kernels and a captured
CUDA graph: 141 tok/s on this box against 32 tok/s for plain transformers generation.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402

import tinyinfer.torch_compat  # noqa: F401,E402  (must precede anything touching dynamo)
from tinyinfer import (attn_fused, gdn_fused, gdn_patch, mlp_fused,  # noqa: E402
                       qwen_norm)
from tinyinfer.hf_graph import GraphDecoder  # noqa: E402

MODEL = r"E:\Models\Qwen3.5-0.8B"


def load(model_dir=MODEL, fuse=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gdn_patch.install()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")
    if fuse:
        # Only the fusions that measured faster inside the captured step. gdn_fused and
        # mlp_fused reproduce the output bit-for-bit but do not move the step time (the
        # weights they join are not the bottleneck), and attn_fused measured *slower*
        # (6.57ms vs 6.41ms) because the Python it adds costs more than the two kernels
        # it removes. qwen_norm is the one that pays: 7.07 -> 6.41ms, 141 -> 156 tok/s.
        qwen_norm.install()
    return tok, model


def stream(tok, model, prompt, max_new=512, history=None):
    """Yield generated text pieces as they are produced."""
    msgs = list(history or []) + [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    dec = GraphDecoder.from_prompt(model, ids).capture()
    eos = tok.eos_token_id
    gen, dt = dec.generate(max_new=max_new, eos=eos)
    # GraphDecoder already accounts for the tokens produced by its prefill/pre-roll, so
    # the list is exactly the generated stream.
    for tid in gen:
        yield tok.decode([tid])
    n = len(gen)
    yield f"\n[{n} tok, {dt:.2f}s, {n / dt:.1f} tok/s]"


def main() -> None:
    one_shot = " ".join(sys.argv[1:]).strip()
    print("loading Qwen3.5-0.8B (graph path)...", flush=True)
    tok, model = load()
    print("ready.  /reset to clear history, Ctrl+C or Ctrl+D to quit.\n", flush=True)
    history = []
    if one_shot:
        for piece in stream(tok, model, one_shot):
            print(piece, end="", flush=True)
        print()
        return
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q in ("/reset", "/clear"):
            history = []
            print("(history cleared)")
            continue
        print("ai > ", end="", flush=True)
        answer = []
        for piece in stream(tok, model, q, history=history):
            print(piece, end="", flush=True)
            answer.append(piece)
        print()
        body = "".join(answer).rsplit("\n[", 1)[0]
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": body}]


if __name__ == "__main__":
    main()
