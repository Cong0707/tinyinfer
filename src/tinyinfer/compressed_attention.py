"""MLA attention with compressed/global KV (compress_ratio > 0), plain PyTorch.

Combines the sliding-window KV with the shared compressed KV cache and the indexer's
top-k positions, mirroring official Attention.forward for layers 2..39.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .compressor import Compressor, Indexer, shared_attn
from .config import TinyConfig
from .nn import apply_rope, precompute_freqs_cis, rms_norm, sparse_attn_ref, window_idxs_prefill
from .quant import dequant_fp8_block
from .weights import ShardStore


class MLACompressedAttention:
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig):
        p = f"layers.{layer_id}.attn."
        self.layer_id = layer_id
        self.cfg = cfg
        self.eps = cfg.rms_norm_eps
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.o_groups = cfg.o_groups
        self.o_lora_rank = cfg.o_lora_rank
        self.window = cfg.sliding_window
        self.ratio = cfg.compress_ratios[layer_id]
        assert self.ratio > 0
        self.is_kv_source = layer_id in cfg.kv_source_layer_ids
        self.is_index_source = layer_id in cfg.index_source_layer_ids
        self.scale = self.head_dim ** -0.5

        self.wq_a = dequant_fp8_block(store.get(p + "wq_a.weight"), store.get(p + "wq_a.scale"))
        self.wq_b = dequant_fp8_block(store.get(p + "wq_b.weight"), store.get(p + "wq_b.scale"))
        self.wkv = dequant_fp8_block(store.get(p + "wkv.weight"), store.get(p + "wkv.scale"))
        self.wo_a = dequant_fp8_block(store.get(p + "wo_a.weight"), store.get(p + "wo_a.scale"))
        self.wo_b = dequant_fp8_block(store.get(p + "wo_b.weight"), store.get(p + "wo_b.scale"))
        self.q_norm = store.get(p + "q_norm.weight")
        self.kv_norm = store.get(p + "kv_norm.weight")
        self.sink = store.get(p + "attn_sink").float()

        # compressed layers use compress_rope_theta + YaRN from the training context length
        self.freqs_cis = precompute_freqs_cis(
            self.rope_dim,
            min(cfg.max_position_embeddings, 1_048_576),
            65536,
            cfg.compress_rope_theta,
            16,
            32.0,
            1.0,
        )
        self.compressor = Compressor(store, layer_id, cfg) if self.is_kv_source else None
        self.indexer = Indexer(store, layer_id, cfg, self.freqs_cis) if self.is_index_source else None
        self._kv_cache: torch.Tensor | None = None          # window ring buffer
        self._compress_kv: torch.Tensor | None = None       # own compressed KV (source layers)

    def reset(self) -> None:
        self._kv_cache = None
        self._compress_kv = None
        if self.compressor is not None:
            self.compressor.reset()
        if self.indexer is not None:
            self.indexer.reset()
        shared_attn.compress_kv = None
        shared_attn.index_k = None
        shared_attn.topk_idxs = None
        shared_attn.candidates = None

    def _decode_idxs(self, start_pos: int, b: int, device) -> torch.Tensor:
        oldest = start_pos % self.window + 1
        idx = torch.cat([torch.arange(oldest, self.window, device=device),
                         torch.arange(0, oldest, device=device)])
        idx = torch.where(idx > start_pos, torch.full_like(idx, -1), idx)
        return idx.int().view(1, 1, -1).expand(b, -1, -1).contiguous()

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        b, s, _ = x.shape
        freqs = self.freqs_cis[start_pos:start_pos + s]

        qr = rms_norm(F.linear(x, self.wq_a), self.q_norm, self.eps)
        q = F.linear(qr, self.wq_b).view(b, s, self.n_heads, self.head_dim)
        apply_rope(q[..., -self.rope_dim:], freqs)

        kv = rms_norm(F.linear(x, self.wkv), self.kv_norm, self.eps)
        apply_rope(kv[..., -self.rope_dim:], freqs)

        if self._kv_cache is None or self._kv_cache.shape[0] != b:
            self._kv_cache = torch.zeros(b, self.window, self.head_dim, dtype=kv.dtype, device=kv.device)
        if start_pos == 0:
            if s <= self.window:
                self._kv_cache[:, :s] = kv
            else:
                cutoff = s % self.window
                tail = kv[:, -self.window:]
                self._kv_cache[:, cutoff:self.window] = tail[:, : self.window - cutoff]
                self._kv_cache[:, :cutoff] = tail[:, self.window - cutoff:]
            window_kv = kv
            idxs = window_idxs_prefill(s, self.window).unsqueeze(0).expand(b, -1, -1).contiguous()
        else:
            self._kv_cache[:, start_pos % self.window] = kv[:, 0]
            window_kv = self._kv_cache
            idxs = self._decode_idxs(start_pos, b, x.device)

        # ---- compressed KV ----
        end_pos = start_pos + s
        compress_len = end_pos // self.ratio
        latent = self.compressor.forward(x, start_pos) if self.compressor is not None else None
        if latent is not None:
            if start_pos == 0:
                cfreqs = self.freqs_cis[: s - s % self.ratio : self.ratio]
            else:
                cfreqs = self.freqs_cis[start_pos + 1 - self.ratio].unsqueeze(0)
            apply_rope(latent[..., -self.rope_dim:], cfreqs)
            need = max(end_pos // self.ratio, latent.size(1), 1)
            if self._compress_kv is None or self._compress_kv.shape[1] < need:
                old = self._compress_kv
                self._compress_kv = torch.zeros(b, need, self.head_dim, dtype=latent.dtype, device=latent.device)
                if old is not None:
                    self._compress_kv[:, : old.size(1)] = old
            self._compress_kv[:, start_pos // self.ratio : start_pos // self.ratio + latent.size(1)] = latent
            shared_attn.compress_kv = self._compress_kv
        if compress_len <= 0:
            compress_kv = None
            compress_idxs = None
        else:
            compress_kv = shared_attn.compress_kv[:, :compress_len]
            if self.indexer is not None:
                compress_idxs = self.indexer.forward(x, qr, latent, start_pos, window_kv.size(1))
                shared_attn.topk_idxs = compress_idxs
            else:
                compress_idxs = shared_attn.topk_idxs

        if compress_idxs is not None and compress_idxs.size(-1) > 0:
            kv_all = torch.cat([window_kv, compress_kv], dim=1)
            idxs_all = torch.cat([idxs, compress_idxs], dim=-1)
        else:
            kv_all = window_kv
            idxs_all = idxs

        o = sparse_attn_ref(q, kv_all, self.sink, idxs_all, self.scale)
        apply_rope(o[..., -self.rope_dim:], freqs, inverse=True)

        o = o.view(b, s, self.o_groups, -1)
        wo_a = self.wo_a.view(self.o_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return F.linear(o.flatten(2), self.wo_b)
