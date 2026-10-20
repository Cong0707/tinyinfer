"""MLA sliding-window attention for DeepSeek-V4.1 (window-only path, compress_ratio == 0)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TinyConfig
from .nn import apply_rope, precompute_freqs_cis, rms_norm, sparse_attn_ref, window_idxs_prefill
from .quant import dequant_fp8_block
from .weights import ShardStore


class MLAWindowAttention:
    """All weights of one attention layer, dequantized to bf16.

    Covers layers with compress_ratio == 0 (pure sliding window). Compressed-KV
    layers (ratio > 0) need the Compressor/Indexer paths and are handled separately.
    """

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig):
        p = f"layers.{layer_id}.attn."
        self.eps = cfg.rms_norm_eps
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.o_groups = cfg.o_groups
        self.o_lora_rank = cfg.o_lora_rank
        self.window = cfg.sliding_window

        self.wq_a = dequant_fp8_block(store.get(p + "wq_a.weight"), store.get(p + "wq_a.scale"))
        self.wq_b = dequant_fp8_block(store.get(p + "wq_b.weight"), store.get(p + "wq_b.scale"))
        self.wkv = dequant_fp8_block(store.get(p + "wkv.weight"), store.get(p + "wkv.scale"))
        self.wo_a = dequant_fp8_block(store.get(p + "wo_a.weight"), store.get(p + "wo_a.scale"))
        self.wo_b = dequant_fp8_block(store.get(p + "wo_b.weight"), store.get(p + "wo_b.scale"))
        self.q_norm = store.get(p + "q_norm.weight")
        self.kv_norm = store.get(p + "kv_norm.weight")
        self.sink = store.get(p + "attn_sink").float()
        self.scale = self.head_dim ** -0.5

        # compress_ratio == 0 -> no YaRN, base rope_theta (see official Attention.__init__)
        self.freqs_cis = precompute_freqs_cis(
            self.rope_dim,
            cfg.max_position_embeddings if cfg.max_position_embeddings <= 1_048_576 else 1_048_576,
            0,
            cfg.rope_theta,
            16,
            32.0,
            1.0,
        )

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        b, s, _ = x.shape
        assert start_pos == 0, "reference implementation currently supports fresh prefill only"
        freqs = self.freqs_cis[start_pos:start_pos + s]

        qr = rms_norm(F.linear(x, self.wq_a), self.q_norm, self.eps)
        q = F.linear(qr, self.wq_b).view(b, s, self.n_heads, self.head_dim)
        apply_rope(q[..., -self.rope_dim:], freqs)

        kv = rms_norm(F.linear(x, self.wkv), self.kv_norm, self.eps)
        apply_rope(kv[..., -self.rope_dim:], freqs)

        idxs = window_idxs_prefill(s, self.window).unsqueeze(0).expand(b, -1, -1).contiguous()
        o = sparse_attn_ref(q, kv, self.sink, idxs, self.scale)
        apply_rope(o[..., -self.rope_dim:], freqs, inverse=True)

        o = o.view(b, s, self.o_groups, -1)
        wo_a = self.wo_a.view(self.o_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return F.linear(o.flatten(2), self.wo_b)
