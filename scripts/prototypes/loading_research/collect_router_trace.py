"""Collect real router traces from tinyinfer (needs the full 40 shards).

The trace is the input for train_lookahead.py and for replacing the synthetic
routing distribution in sim_router.py.  It records, per decode step and layer:

* hidden:  the sublayer input entering the layer (CPU fp32, [dim])
* logits:  the raw gate logits before top-k (fp32, [n_experts])
* routes:  the top-k router decision (int16, [k])

Only the first --steps generated positions are kept; prefill positions are
skipped because they are not representative of decode routing.

Usage (after every model-000xx shard is complete):
    python collect_router_trace.py --model-dir cache/weights --tokens 128         --prompt-file prompt.txt --out router_trace.npz

Note: this currently works only for layers whose attention path tinyinfer has
implemented (compress_ratio == 0); for the rest it records nothing and says so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tinyinfer.config import TinyConfig  # noqa: E402
from tinyinfer.moe import Gate  # noqa: E402
from tinyinfer.model import TinyModel  # noqa: E402
from tinyinfer.weights import ShardStore  # noqa: E402


class RouterTracer:
    def __init__(self, n_layers: int, n_experts: int, k: int):
        self.hidden: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
        self.logits: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
        self.routes: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
        self.n_layers, self.n_experts, self.k = n_layers, n_experts, k

    def install(self) -> None:
        tracer = self
        original = Gate.forward

        def forward(gate, x, image_mask=None):  # type: ignore[no-untyped-def]
            layer = getattr(gate, "_tracer_layer", None)
            if layer is not None and tracer.n_layers > layer:
                scores = torch.nn.functional.linear(x.float(), gate.weight)
                tracer.logits[layer].append(scores.detach().float().cpu().numpy().reshape(-1))
            weights, indices = original(gate, x, image_mask)
            if layer is not None and tracer.n_layers > layer:
                tracer.routes[layer].append(indices.detach().cpu().numpy().reshape(-1))
            return weights, indices

        Gate.forward = forward  # type: ignore[assignment]

    def snapshot(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for layer in range(self.n_layers):
            if self.logits[layer]:
                out[f"logits_l{layer}"] = np.stack(self.logits[layer])
                out[f"routes_l{layer}"] = np.stack(self.routes[layer])
                out[f"hidden_l{layer}"] = np.stack(self.hidden[layer])
        return out


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path(__file__).resolve().parents[3] / "cache" / "weights"))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--prompt", default="def fibonacci(n):")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = TinyConfig.from_hf(args.model_dir)
    store = ShardStore(args.model_dir)
    model = TinyModel(store, cfg, layers=list(range(cfg.n_layers)))
    tracer = RouterTracer(cfg.n_layers, cfg.n_routed_experts, cfg.n_activated_experts)
    tracer.install()
    for block in model.blocks:
        block.ffn.gate._tracer_layer = block.layer_id
        original = block.forward

        def forward(x, start_pos, pre_mix, _orig=original, _layer=block.layer_id):  # type: ignore[no-untyped-def]
            tracer.hidden[_layer].append(x.detach().float().mean(dim=(0, 1)).cpu().numpy())
            return _orig(x, start_pos, pre_mix)

        block.forward = forward  # type: ignore[assignment]

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        text = args.prompt
    ids = tokenizer.encode(text).ids[-512:]
    input_ids = torch.tensor([ids], dtype=torch.long)
    logits = None
    for _ in range(args.tokens):
        out = model.forward(input_ids[:, -1:] if logits is not None else input_ids)
        logits = out
        nxt = torch.tensor([[int(logits[0, -1].argmax())]])
        input_ids = torch.cat([input_ids, nxt], dim=1)
    data = tracer.snapshot()
    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out}: {len(data)} arrays, shapes e.g. "
          f"{ {k: v.shape for k, v in list(data.items())[:3]} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
