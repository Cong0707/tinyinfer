"""MTP smoke test: load mtp.0..2, run the full draft chain and the last-stage head.

Official DFlash contract:
  golden token -> embed (noise-filled block)        -> draft hiddens x0
  main_proj(concat target hiddens) + norm           -> main_x (seeds draft KV)
  stage0..2 forward(x, pos, pre, main_x)            -> draft hiddens
  stage2.forward_head(hidden, pre_mix, draft_ids)   -> logits + confidence
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
    stages = eng.mtp
    s0 = stages[0]

    golden = torch.tensor([[19923]], dtype=torch.long, device="cuda")
    eng.forward(golden, start_pos=0)          # prefill: captures target hiddens + KV
    th = eng._target_hidden
    print("target layers:", sorted(th.keys()), flush=True)
    main_x = s0.main_hiddens([th[i] for i in sorted(th.keys())])
    print("main_x:", tuple(main_x.shape), flush=True)
    x = s0.forward_embed(golden)
    print("x0:", tuple(x.shape), flush=True)
    pos = 1                                    # first decode step after prefill
    pre = torch.zeros(1, x.size(1), eng.cfg.hc_mult, device="cuda")
    pre[:, :, 0] = 1.0
    for st in stages:
        st.to("cuda")
        x, pre = st.forward(x, pos, pre, main_x)
    torch.cuda.synchronize()
    print("draft stages ok:", tuple(x.shape), flush=True)
    logits, confidence = stages[-1].forward_head(x, pre, golden)
    if logits is not None:
        print("head logits:", tuple(logits.shape), "argmax:", int(logits[0, 0].argmax()), flush=True)
        main_logits = eng.forward(golden, start_pos=0)
        a = int(logits[0, 0].argmax())
        b = int(main_logits[0, -1].argmax())
        top5m = torch.topk(main_logits[0, -1].float(), 5).indices.tolist()
        top5d = torch.topk(logits[0, 0].float(), 5).indices.tolist()
        print("MTP argmax:", a, "| main argmax:", b, "| match:", a == b, flush=True)
        print("MTP top5:", top5d, "| main top5:", top5m, flush=True)
    if confidence is not None:
        print("confidence:", tuple(confidence.shape), [round(v, 4) for v in confidence[0, :3].tolist()], flush=True)


if __name__ == "__main__":
    main()
