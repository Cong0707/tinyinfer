"""Layer-0 router precheck with real weights (E-L3 partial, runs before all shards exist).

Layer 0 needs only model-00002 (embed), model-00003 (layer 0) and the tokenizer.
It records the real gate decisions for a prompt and prints the distribution /
temporal-locality numbers used to sanity-check the synthetic cache simulation.

Usage:
    python precheck_layer0.py --prompt-file ../../deepseek/prompt.txt
    python precheck_layer0.py --prompt "def quicksort(xs):" --max-tokens 192
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.config import TinyConfig  # noqa: E402
from tinyinfer.model import Block  # noqa: E402
from tinyinfer.nn import rms_norm  # noqa: E402
from tinyinfer.weights import ShardStore  # noqa: E402


def lru_hit(routes: np.ndarray, capacity: int) -> float:
    cache: dict[int, None] = {}
    hits = 0
    for route in routes:
        for expert in route.tolist():
            if expert in cache:
                hits += 1
                cache.pop(expert)
                cache[expert] = None
            else:
                if len(cache) >= capacity:
                    cache.pop(next(iter(cache)))
                cache[expert] = None
    return hits / (len(routes) * routes.shape[1])


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "cache" / "weights"))
    ap.add_argument("--prompt", default=(
        "def quicksort(xs):\n"
        "    if len(xs) <= 1:\n        return xs\n"
        "    pivot = xs[len(xs)//2]\n"
        "    left = [x for x in xs if x < pivot]\n"
        "    mid = [x for x in xs if x == pivot]\n"
        "    right = [x for x in xs if x > pivot]\n"
        "    return quicksort(left) + mid + quicksort(right)\n\n"
        "The quick brown fox jumps over the lazy dog. DeepSeek V4.1 Flash uses a sparse "
        "mixture of experts with 384 routed experts per layer. Only six are activated per "
        "token, plus one shared expert. The router scores are sqrt(softplus(gate logits)) "
        "and the top six are normalized and scaled by 1.5."))
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(16)
    model_dir = Path(args.model_dir)
    shard = model_dir / "model-00003-of-00048.safetensors"
    if not shard.exists() or shard.stat().st_size < 7_000_000_000:
        print(f"layer 0 shard is incomplete ({shard})", file=sys.stderr)
        return 1
    cfg = TinyConfig.from_hf(model_dir)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    text = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    ids = tok(text).input_ids[: args.max_tokens]
    print(f"prompt tokens: {len(ids)}")

    with ShardStore(model_dir) as store:
        block = Block(store, 0, cfg)
        embed = store.get("embed.weight")
        h = embed[torch.tensor(ids, dtype=torch.long).unsqueeze(0)]
        h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        pre_mix = torch.zeros(1, len(ids), cfg.hc_mult)
        pre_mix[:, :, 0] = 1.0

        residual = h
        attn_pre, attn_post, attn_comb = block.hc_mixes(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
        x = block.hc_pre(h, pre_mix)
        x = rms_norm(x, block.attn_norm, cfg.rms_norm_eps)
        x = block.attn.forward(x, 0)
        x = block.hc_post(x, residual, attn_post, attn_comb)
        block.hc_mixes(x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
        h2 = block.hc_pre(x, attn_pre)
        h2 = rms_norm(h2, block.ffn_norm, cfg.rms_norm_eps)
        gate = block.ffn.gate
        scores = F.softplus(F.linear(h2.float(), gate.weight)).sqrt()
        indices = (scores + gate.bias).topk(cfg.n_activated_experts, dim=-1)[1]

    routes = indices.reshape(-1, cfg.n_activated_experts).numpy()
    counts = np.bincount(routes.reshape(-1), minlength=cfg.n_routed_experts)
    share = np.sort(counts)[::-1] / counts.sum()
    entropy = -(share[share > 0] * np.log(share[share > 0])).sum()
    overlaps = [len(set(routes[i]) & set(routes[i + 1])) for i in range(len(routes) - 1)]
    result = {
        "tokens": int(len(routes)),
        "unique_experts": int((counts > 0).sum()),
        "top6_share": float(share[:6].sum()),
        "top24_share": float(share[:24].sum()),
        "top96_share": float(share[:96].sum()),
        "effective_experts": float(np.exp(entropy)),
        "adjacent_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "adjacent_p_any": float(np.mean(np.array(overlaps) >= 1)) if overlaps else 0.0,
        "lru": {str(c): lru_hit(routes, c) for c in (12, 20, 27, 40, 64)},
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
