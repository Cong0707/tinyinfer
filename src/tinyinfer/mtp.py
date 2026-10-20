"""MTP / DSpark speculative stages (mtp.0..2 in the checkpoint).

Each stage is a small block that drafts the next tokens from the main model's
hidden states; the last stage carries the markov logit correction and a
confidence head. Weights live under the mtp.* namespace and are addressed
through a prefix-rewriting store so the shared Block/attention/MoE classes load
them unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import MLAWindowAttention
from .config import TinyConfig
from .moe import MoE
from .nn import apply_rope, rms_norm, window_idxs_general
from .quant import dequant_fp8_block
from .weights import ShardStore


class _PrefixStore:
    """Maps layers.{src_layer}.* reads onto mtp.{stage}.*."""

    def __init__(self, store: ShardStore, stage: int, src_layer: int = 0):
        self._store = store
        self._stage = stage
        self._src = src_layer
        self.model_dir = store.model_dir

    def _map(self, name: str) -> str:
        prefix = f"layers.{self._src}."
        if name.startswith(prefix):
            return f"mtp.{self._stage}." + name[len(prefix):]
        return name

    def has(self, name: str) -> bool:
        return self._store.has(self._map(name))

    def shard_of(self, name: str) -> str:
        return self._store.shard_of(self._map(name))

    def get(self, name: str) -> torch.Tensor:
        return self._store.get(self._map(name))


class MTPAttention(MLAWindowAttention):
    """Window attention whose KV history is seeded by the main model (official DSparkAttention)."""

    def forward_with_main(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        win = self.window
        rd = self.rope_dim
        main_freqs = self.freqs_cis[start_pos:start_pos + s]

        main_kv = rms_norm(F.linear(main_x, self._w(self.wkv)), self.kv_norm, self.eps)
        apply_rope(main_kv[..., -rd:], main_freqs)
        if start_pos == 0:
            # prefill only seeds the window history; no draft compute here
            self._kv_window = main_kv.contiguous()
            self._kv_buf = None
            return x

        freqs = self.freqs_cis[start_pos:start_pos + s]
        qr = rms_norm(F.linear(x, self._w(self.wq_a)), self.q_norm, self.eps)
        q = F.linear(qr, self._w(self.wq_b)).view(b, s, self.n_heads, self.head_dim)
        apply_rope(q[..., -rd:], freqs)
        kv = rms_norm(F.linear(x, self._w(self.wkv)), self.kv_norm, self.eps)
        apply_rope(kv[..., -rd:], freqs)

        old = self._kv_window if self._kv_window is not None else kv.new_empty(b, 0, self.head_dim)
        kv_att = torch.cat([old, kv], dim=1)
        idxs = window_idxs_general(start_pos, s, old.size(1), win, x.device).unsqueeze(0).expand(b, -1, -1)
        from .kernels import sparse_attn_dispatch
        o = sparse_attn_dispatch(q, kv_att, self.sink, idxs, self.scale)
        apply_rope(o[..., -rd:], freqs, inverse=True)
        o = o.view(b, s, self.o_groups, -1)
        wo_a = self._w(self.wo_a).view(self.o_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return F.linear(o.flatten(2), self._w(self.wo_b))

    @staticmethod
    def _w(t):
        return t.weight if hasattr(t, "weight") else t


class MTPBlock:
    """One DSpark stage: attention seeded by main_x, a small MoE, and (last stage) heads."""

    def __init__(self, store: ShardStore, stage: int, cfg: TinyConfig, quant: bool = False):
        import dataclasses

        self.stage = stage
        self.cfg = cfg
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        stage_cfg = cfg
        if cfg.dspark_n_routed_experts and cfg.dspark_n_routed_experts != cfg.n_routed_experts:
            stage_cfg = dataclasses.replace(cfg, n_routed_experts=cfg.dspark_n_routed_experts,
                                            n_activated_experts=cfg.dspark_n_activated_experts)
        shim = _PrefixStore(store, stage, 0)
        self.attn = MTPAttention(shim, 0, cfg, quant=quant)
        self.ffn = MoE(shim, 0, stage_cfg)
        self.attn_norm = shim.get("layers.0.attn_norm.weight")
        self.ffn_norm = shim.get("layers.0.ffn_norm.weight")
        self.hc_attn_fn = shim.get("layers.0.hc_attn_fn").float()
        self.hc_attn_base = shim.get("layers.0.hc_attn_base").float()
        self.hc_attn_scale = shim.get("layers.0.hc_attn_scale").float()
        self.hc_ffn_fn = shim.get("layers.0.hc_ffn_fn").float()
        self.hc_ffn_base = shim.get("layers.0.hc_ffn_base").float()
        self.hc_ffn_scale = shim.get("layers.0.hc_ffn_scale").float()
        self.block_size = cfg.dspark_block_size
        self.eps = cfg.rms_norm_eps
        self.main_proj = self.main_norm = None
        if stage == 0:
            self.main_proj = dequant_fp8_block(shim.get("layers.0.main_proj.weight"),
                                               shim.get("layers.0.main_proj.scale"))                 if shim.has("layers.0.main_proj.weight") else shim.get("layers.0.main_proj.weight")
            self.main_norm = shim.get("layers.0.main_norm.weight")
        self.norm = self.markov_embed = self.markov_head = self.confidence_proj = None
        if stage == cfg.n_mtp_layers - 1:
            self.norm = shim.get("layers.0.norm.weight")
            self.markov_embed = shim.get("layers.0.markov_head.embed.weight")
            self.markov_head = shim.get("layers.0.markov_head.head.weight")
            self.confidence_proj = shim.get("layers.0.confidence_head.proj.weight").float()

    # -- HC helpers (same math as Block) ----------------------------------
    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        from .hc import hc_split_sinkhorn
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.hc_eps)
        mixes = F.linear(xf, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x, pre_mix):
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    # -- forward (official DSparkBlock) -----------------------------------
    def forward(self, x, start_pos, pre_mix, main_x):
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.hc_pre(x, pre_mix)
        h = rms_norm(h, self.attn_norm, self.eps)
        h = self.attn.forward_with_main(h, start_pos, main_x)
        x = self.hc_post(h, x, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.hc_pre(x, attn_pre)
        h = rms_norm(h, self.ffn_norm, self.eps)
        h = self.ffn.forward(h)
        x = self.hc_post(h, x, ffn_post, ffn_comb)
        return x, ffn_pre

    def to(self, device):
        for name in ("attn_norm", "ffn_norm", "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                     "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale", "main_proj", "main_norm",
                     "norm", "markov_embed", "markov_head", "confidence_proj"):
            v = getattr(self, name, None)
            if isinstance(v, torch.Tensor):
                setattr(self, name, v.to(device))
        if self.attn is not None:
            self.attn.to(device)
        self.ffn.to(device)
        return self


def load_mtp(store: ShardStore, cfg: TinyConfig, quant: bool = False) -> list[MTPBlock] | None:
    """Instantiate all MTP stages if the mtp.* weights are present."""
    if cfg.n_mtp_layers <= 0:
        return None
    try:
        stages = [MTPBlock(store, st, cfg, quant=quant) for st in range(cfg.n_mtp_layers)]
    except (KeyError, FileNotFoundError, ValueError):
        return None
    return stages
