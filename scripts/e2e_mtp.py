"""MTP stage smoke test: load mtp.0..2 and run the stage-0 draft chain.

Official DFlash contract:
  golden token -> embed (noise-filled block)        -> draft hiddens x0
  main_proj(concat device target hiddens) + norm    -> main_x (seeds draft KV)
  forward(x0, pos, pre, main_x)                     -> draft hiddens (attn+MoE)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/Models/DeepSeek-V4.1-Flash-1017")
    if not (model_dir / "model.safetensors.index.json").exists():
        model_dir = Path("E:/Develop/Deepseek/tinyinfer/cache/weights")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyinfer.engine import Engine

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as e:
        print("tokenizer unavailable:", e)

    eng = Engine.from_hf(model_dir, device="cuda", tokenizer=tok, max_layers=40, allow_missing_layers=True)
    print("blocks:", eng.max_layers, "gpu_layers:", eng.gpu_layers, flush=True)
    print("load_mtp:", eng.load_mtp(), flush=True)
    s0 = eng.mtp[0]

    golden = torch.tensor([[19923]], dtype=torch.long, device="cuda")
    eng.forward(golden, start_pos=0)
    th = eng._target_hidden
    print("target layers:", sorted(th.keys()), flush=True)
    target_hiddens = [th[i] for i in sorted(th.keys())]
    s0.to("cuda")
    main_x = s0.main_hiddens(target_hiddens)          # [1,1,5120]
    print("main_x:", tuple(main_x.shape), flush=True)
    x0 = s0.forward_embed(golden)                     # [1,5,hc,dim]
    print("x0:", tuple(x0.shape), flush=True)
    pre = torch.zeros(1, x0.size(1), eng.cfg.hc_mult, device="cuda")
    pre[:, :, 0] = 1.0
    # emulate the first decode step after prefill: draft positions 1..block_size
    out, pre2 = s0.forward(x0, 1, pre, main_x)
    torch.cuda.synchronize()
    print("stage0 forward ok:", tuple(out.shape), "pre_mix:", tuple(pre2.shape), flush=True)


if __name__ == "__main__":
    main()
