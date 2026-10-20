"""Measure EXTREME-tier approximation cost on real L0 weights."""
import sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.approx import fft_attention_approx, low_rank_linear, skip_layer_plan
from tinyinfer.config import TinyConfig
from tinyinfer.quant import dequant_fp8_block
from tinyinfer.weights import ShardStore

MODEL = ROOT / "cache" / "weights"
cfg = TinyConfig.from_hf(MODEL)


def main():
    store = ShardStore(MODEL)
    print("cfg:", cfg.summary())

    # 1) low-rank fitting on the real grouped output projections
    for name in ("wo_a", "wo_b", "wq_b"):
        w = dequant_fp8_block(store.get(f"layers.0.attn.{name}.weight"), store.get(f"layers.0.attn.{name}.scale"))
        for ratio in (0.75, 0.5, 0.25):
            approx, st = low_rank_linear(w, ratio)
            print(f"{name} {ratio:.2f}: rank={st['rank']} kept_energy={st['kept_energy']:.4f} "
                  f"rel_fro_err={st['rel_fro_err']:.4f}")

    # 2) FFT token-mix error vs a real L0 hidden state
    import tinyinfer.kernels as K
    from tinyinfer.attention import MLAWindowAttention
    attn = MLAWindowAttention(store, 0, cfg)
    K._TL_STATE = False  # CPU reference only
    torch.manual_seed(0)
    x = torch.randn(1, 8, cfg.dim, dtype=torch.bfloat16) * 0.02
    with torch.inference_mode():
        y_ref = attn.forward(x, 0)
    # FFT mixing has no q/k/v wiring in the reference; test its output stats as a baseline
    from tinyinfer.approx import fft_token_mix
    y_fft = fft_token_mix(x)
    rel = (y_fft.float() - x.float()).norm() / x.float().norm()
    print(f"fft_token_mix: rel_change_vs_input={rel:.4f} (no attention parameters)")

    # 3) layer-skip plan
    print("skip plan (every 8):", skip_layer_plan(cfg.n_layers, 8))


if __name__ == "__main__":
    main()
