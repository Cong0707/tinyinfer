"""TinyModel: direct-from-HF blocks and a minimal transformer driver."""
from __future__ import annotations

import torch

from .attention import MLAWindowAttention
from .compressed_attention import MLACompressedAttention
from .config import TinyConfig
from .hc import HCBlockMixin
from .moe import MoE
from .nn import rms_norm
from .weights import ShardStore


class Block(HCBlockMixin):
    """One decoder layer: Hyper-Connection collapse -> norm -> attention -> HC expand -> ... FFN."""

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig):
        self.layer_id = layer_id
        self.cfg = cfg
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        if cfg.compress_ratios[layer_id] == 0:
            self.attn = MLAWindowAttention(store, layer_id, cfg)
        else:
            self.attn = MLACompressedAttention(store, layer_id, cfg)
        self.ffn = MoE(store, layer_id, cfg)
        p = f"layers.{layer_id}."
        self.attn_norm = store.get(p + "attn_norm.weight")
        self.ffn_norm = store.get(p + "ffn_norm.weight")
        self.hc_attn_fn = store.get(p + "hc_attn_fn").float()
        self.hc_attn_base = store.get(p + "hc_attn_base").float()
        self.hc_attn_scale = store.get(p + "hc_attn_scale").float()
        self.hc_ffn_fn = store.get(p + "hc_ffn_fn").float()
        self.hc_ffn_base = store.get(p + "hc_ffn_base").float()
        self.hc_ffn_scale = store.get(p + "hc_ffn_scale").float()

    def forward(self, x: torch.Tensor, start_pos: int, pre_mix: torch.Tensor):
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.hc_pre(x, pre_mix)
        h = rms_norm(h, self.attn_norm, self.cfg.rms_norm_eps)
        h = self.attn.forward(h, start_pos)
        x = self.hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.hc_pre(x, attn_pre)
        h = rms_norm(h, self.ffn_norm, self.cfg.rms_norm_eps)
        h = self.ffn.forward(h)
        x = self.hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class TinyModel:
    """Minimal driver: embed -> (selected blocks) -> norm -> head.

    Loading all 40 blocks would need ~500GB of resident weights, so this driver
    takes an explicit layer list for staged verification.
    """

    def __init__(self, store: ShardStore, cfg: TinyConfig, layers: list[int] | None = None):
        self.store = store
        self.cfg = cfg
        self.embed = store.get("embed.weight")
        self.out_norm = store.get("norm.weight")
        self.head = store.get("head.weight")
        self.layers = layers or []
        self.blocks = [Block(store, i, cfg) for i in self.layers]

    def reset(self) -> None:
        for block in self.blocks:
            if hasattr(block.attn, "reset"):
                block.attn.reset()
            block.ffn._expert_cache.clear()

    def forward(self, ids: torch.Tensor, start_pos: int = 0):
        h = self.embed[ids]  # [b, s, dim]
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = torch.zeros(h.size(0), h.size(1), self.cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        for block in self.blocks:
            h, pre_mix = block.forward(h, start_pos, pre_mix)
        from .hc import hc_split_sinkhorn  # noqa: F401  (kept importable for users)
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2).to(h.dtype)
        h = rms_norm(h, self.out_norm, self.cfg.rms_norm_eps)
        # bf16 matvec (official head stores fp32 weights; bf16 is 2x faster on CPU and
        # numerically fine for sampling) -- see docs/loading.md head measurements
        logits = torch.nn.functional.linear(h, self.head).float()
        return logits
