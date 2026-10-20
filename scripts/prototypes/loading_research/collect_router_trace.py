"""Collect a real router trace for E-L3/E-L4 once the 40 shards are complete.

Records, per decode step and layer:
  hidden_l{L}  [steps, dim]      hc-averaged block input (bf16)
  logits_l{L}  [steps, 384]      raw gate logits (fp16)
  routes_l{L}  [steps, 6]        real top-6 (int16)

The array names match train_lookahead.py.  Prefill positions are not recorded:
the predictor is trained for decode.

Runtime warning: reference CPU path.  A short prompt plus N decode steps costs
N x 4.5 GB of expert reads and N x 240 fp4 dequantizations; use --max-steps 16
for a first pass, or a GPU-backed runner once Lead lands it.  --smoke checks the
plumbing with layer 0 and 2 steps before the other shards exist.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.config import TinyConfig  # noqa: E402
from tinyinfer.loading import ExpertStore  # noqa: E402
from tinyinfer.model import Block  # noqa: E402
from tinyinfer.moe import Expert, Gate, MoE  # noqa: E402
from tinyinfer.nn import rms_norm  # noqa: E402
from tinyinfer.quant import dequant_fp4_block  # noqa: E402
from tinyinfer.weights import ShardStore  # noqa: E402


def install_no_bf16_cache() -> None:
    """Replace MoE._expert's unbounded bf16 cache with per-call dequant."""
    original = MoE._expert

    def _expert_no_cache(self, i: int):  # type: ignore[no-untyped-def]
        if self._expert_store is None:
            return original(self, i)
        with self._expert_store.experts(self.layer_id, [i], cache=False) as recs:
            rec = recs[0]
            w = {n: dequant_fp4_block(rec.tensor(n + ".weight"), rec.tensor(n + ".scale"))
                 for n in ("w1", "w2", "w3")}
        return Expert(w["w1"], w["w2"], w["w3"], self.swiglu_limit)

    MoE._expert = _expert_no_cache  # type: ignore[assignment]


class TraceHooks:
    def __init__(self) -> None:
        self.record = False
        self.hidden: dict[int, list[np.ndarray]] = {}
        self.logits: dict[int, list[np.ndarray]] = {}
        self.routes: dict[int, list[np.ndarray]] = {}

    def install(self, blocks: list[Block]) -> None:
        tracer = self
        for block in blocks:
            layer = block.layer_id
            tracer.hidden.setdefault(layer, [])
            original = block.forward

            def forward(x, start_pos, pre_mix, _orig=original, _layer=layer):  # type: ignore[no-untyped-def]
                if tracer.record:
                    tracer.hidden[_layer].append(
                        x.detach().float().mean(dim=2).reshape(-1).to(torch.float16).cpu().numpy())
                return _orig(x, start_pos, pre_mix)

            block.forward = forward  # type: ignore[assignment]
            block.ffn.gate._trace_layer = layer  # type: ignore[attr-defined]

        original_gate = Gate.forward

        def gate_forward(gate, x, image_mask=None):  # type: ignore[no-untyped-def]
            scores = F.linear(x.float(), gate.weight)
            if gate.score_func == "softmax":
                scored = scores.softmax(dim=-1)
            elif gate.score_func == "sigmoid":
                scored = scores.sigmoid()
            else:
                scored = F.softplus(scores).sqrt()
            weights, indices = original_gate(gate, x, image_mask)
            layer = getattr(gate, "_trace_layer", None)
            if tracer.record and layer is not None:
                tracer.logits.setdefault(layer, []).append(
                    scored.reshape(-1).to(torch.float16).cpu().numpy())
                tracer.routes.setdefault(layer, []).append(
                    indices.reshape(-1).to(torch.int16).cpu().numpy())
            return weights, indices

        Gate.forward = gate_forward  # type: ignore[assignment]


class TraceModel:
    def __init__(self, store: ShardStore, cfg: TinyConfig, layers: list[int], expert_store: ExpertStore):
        self.cfg = cfg
        self.embed = store.get("embed.weight")
        self.out_norm = store.get("norm.weight")
        self.head = store.get("head.weight")
        self.blocks = [Block(store, i, cfg, expert_store=expert_store) for i in layers]

    def reset(self) -> None:
        for block in self.blocks:
            if hasattr(block.attn, "reset"):
                block.attn.reset()

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, start_pos: int = 0):
        h = self.embed[ids]
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = torch.zeros(h.size(0), h.size(1), self.cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        for block in self.blocks:
            h, pre_mix = block.forward(h, start_pos, pre_mix)
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2).to(h.dtype)
        h = rms_norm(h, self.out_norm, self.cfg.rms_norm_eps)
        return F.linear(h, self.head).float()


def parse_layers(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "cache" / "weights"))
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-gb", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true", help="layer 0, 2 steps, print-only checks")
    args = ap.parse_args()

    torch.set_num_threads(16)
    model_dir = Path(args.model_dir)
    cfg = TinyConfig.from_hf(model_dir)
    layers = [0] if args.smoke else parse_layers(args.layers)
    steps = 2 if args.smoke else args.max_steps
    out = args.out or ("router_trace_smoke.npz" if args.smoke else "router_trace.npz")
    if args.smoke:
        print("SMOKE MODE: layer 0, 2 decode steps; arrays are not a real trace")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    text = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    ids = tok(text).input_ids[:64]
    print(f"prompt tokens: {len(ids)}, layers: {layers}, decode steps: {steps}")

    install_no_bf16_cache()
    tracer = TraceHooks()
    started = time.perf_counter()
    with ShardStore(model_dir) as store:
        expert_store = ExpertStore.from_store(
            store, cache_bytes=int(args.cache_gb * 1e9), policy="arc",
            vram_slots=0, device="cpu", reader_threads=6, direct=True)
        model = TraceModel(store, cfg, layers, expert_store)
        tracer.install(model.blocks)
        model.reset()

        prompt = torch.tensor([ids], dtype=torch.long)
        logits = model.forward(prompt, 0)
        next_id = int(logits[0, -1].argmax())
        generated = [next_id]
        start = len(ids)
        for step in range(steps):
            tracer.record = True
            logits = model.forward(torch.tensor([[generated[-1]]], dtype=torch.long), start + step)
            tracer.record = False
            nxt = int(logits[0, -1].argmax())
            generated.append(nxt)
        stats = expert_store.stats()
        expert_store.close()

    elapsed = time.perf_counter() - started
    data = {}
    for layer in layers:
        if tracer.hidden.get(layer):
            data[f"hidden_l{layer}"] = np.stack(tracer.hidden[layer])
            data[f"logits_l{layer}"] = np.stack(tracer.logits[layer])
            data[f"routes_l{layer}"] = np.stack(tracer.routes[layer])
    meta = {
        "model_dir": str(model_dir), "layers": layers, "steps": steps,
        "prompt_tokens": len(ids), "generated": generated[:steps],
        "elapsed_s": elapsed, "expert_store": stats, "smoke": args.smoke,
    }
    data["_metadata"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(out, **data)

    routes = np.stack([tracer.routes[l][0] for l in layers if tracer.routes.get(l)])
    flat = routes.reshape(-1)
    counts = np.bincount(flat, minlength=cfg.n_routed_experts)
    share = np.sort(counts)[::-1] / max(1, counts.sum())
    print(f"saved {out}: {len(data) - 1} arrays, {elapsed:.1f}s, "
          f"experts read {stats['stream_gb']:.1f} GB (stream)")
    print(f"step0 routes: unique={int((counts > 0).sum())}/384, "
          f"top24_share={share[:24].sum():.3f}, top96_share={share[:96].sum():.3f}")
    print("next: train_lookahead.py --trace", out, "--lead 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
