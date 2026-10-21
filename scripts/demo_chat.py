"""Demo: a real chat turn with a system prompt on the local engine.

DeepSeek chat format (the V4 tokenizer ships <|System|>/<|User|>/<|Assistant|> but no
chat_template, so we build the prompt explicitly):

    <|begin_of_sentence|><|System|>{system}<|User|>{user}<|Assistant|>

Usage: python scripts/demo_chat.py "你好"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

SYSTEM = "你是 DeepSeek，一个由深度求索公司创造的 AI 助手。"


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from transformers import AutoTokenizer

    from tinyinfer.engine import Engine

    md = "E:/Models/DeepSeek-V4.1-Flash-1017"
    user = sys.argv[1] if len(sys.argv) > 1 else "你好"
    tok = AutoTokenizer.from_pretrained(md, trust_remote_code=True)
    eng = Engine.from_hf(md, device="cuda", tokenizer=tok, allow_missing_layers=True)

    prompt = (f"<｜begin▁of▁sentence｜><｜System｜>{SYSTEM}"
              f"<｜User｜>{user}<｜Assistant｜>")
    ids = tok.encode(prompt, add_special_tokens=False)
    out: list[int] = []
    t0 = time.time()
    step_times: list[float] = []
    with torch.inference_mode():
        logits = eng.forward(torch.tensor([ids], device="cuda"), 0)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # drop the prefill kernel workspace before decoding
        t_pre = time.time() - t0
        for step in range(32):
            nid = int(logits[0, -1].argmax())
            out.append(nid)
            if nid == tok.eos_token_id:
                break
            t = time.time()
            logits = eng.forward(torch.tensor([[nid]], device="cuda"),
                                 start_pos=len(ids) + step)
            torch.cuda.synchronize()
            step_times.append(time.time() - t)
            print(f"  tok {step+1:2d} {step_times[-1]*1000:5.0f}ms  "
                  f"alloc={torch.cuda.memory_allocated()/2**20:5.0f}MB  "
                  f"reserved={torch.cuda.memory_reserved()/2**20:5.0f}MB", flush=True)
    dt = time.time() - t0
    print("USER:", user, flush=True)
    print("ASSISTANT:", tok.decode(out), flush=True)
    steady = sum(step_times[1:]) / max(1, len(step_times) - 1)
    print(f"\nprefill {t_pre:.1f}s | {len(out)} tokens in {dt:.1f}s = {len(out)/dt:.2f} tok/s"
          f" | steady {steady*1000:.0f}ms/token = {1/steady:.2f} tok/s"
          f" | vram {torch.cuda.max_memory_allocated()/2**20:.0f}MB", flush=True)


if __name__ == "__main__":
    main()