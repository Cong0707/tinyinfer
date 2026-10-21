"""Compressed-KV sparse attention path: softmax-gated pooling + indexer top-k.

Mirrors the official Compressor/Indexer (reference/official/model.py) in plain PyTorch.
This version supports arbitrary (start_pos, seqlen) chunks so prefill can be split while
keeping the same math as a single-shot prefill.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TinyConfig
from .nn import apply_rope, rms_norm
from .quant import dequant_fp8_block
from .weights import ShardStore


class SharedAttentionRuntime:
    """Inter-layer handoff slots (one model per process, layers run in order)."""

    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
    src_ratio: int = 0  # compress ratio of the kv_source layer that produced compress_kv
    src_ratio: int = 0  # compress ratio of the kv_source layer that produced compress_kv


shared_attn = SharedAttentionRuntime()


class Compressor:
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, *, quant: bool = False):
        p = f"layers.{layer_id}.attn.compressor."
        self.ratio = cfg.compress_ratios[layer_id]
        self.head_dim = cfg.head_dim
        self.eps = cfg.rms_norm_eps
        self.norm_w = store.get(p + "norm.weight")
        # ratio>1 pools in fp32 (official); the source weights are bf16
        self._quant = quant
        self.wkv = store.get(p + "wkv.weight").float()
        self.wgate = store.get(p + "wgate.weight").float() if self.ratio > 1 else None
        self._states: dict[int, dict] = {}
        self._active_layer = 0

    def _st(self) -> dict:
        st = self._states.get(self._active_layer)
        if st is None:
            st = self._states[self._active_layer] = {}
        return st

    @property
    def _pending_kv(self):
        return self._st().get("pkv")

    @_pending_kv.setter
    def _pending_kv(self, value):
        self._st()["pkv"] = value

    @property
    def _pending_score(self):
        return self._st().get("psc")

    @_pending_score.setter
    def _pending_score(self, value):
        self._st()["psc"] = value

    def reset(self) -> None:
        self._pending_kv = None
        self._pending_score = None

    def to(self, device):
        self.norm_w = self.norm_w.to(device)
        self.wkv = self.wkv.to(device)
        if self.wgate is not None:
            self.wgate = self.wgate.to(device)
        if self._pending_kv is not None:
            self._pending_kv = self._pending_kv.to(device)
            self._pending_score = self._pending_score.to(device)
        return self

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        """Pool complete ratio-sized global groups; returns latent [b, n_groups, dim] or None."""
        b, s, _ = x.shape
        xf = x.float()
        if self.ratio == 1:
            shared_attn.src_ratio = 1  # ratio-1 sources publish without pooling
            return rms_norm(F.linear(xf, self.wkv), self.norm_w, self.eps).to(x.dtype)
        kv = F.linear(xf, self.wkv)
        score = F.linear(xf, self.wgate)
        pkv = self._pending_kv
        psc = self._pending_score
        if pkv is None or pkv.shape[0] != b:
            pkv = kv.new_empty(b, 0, self.head_dim)
            psc = score.new_empty(b, 0, self.head_dim)
        all_kv = torch.cat([pkv, kv], dim=1)
        all_sc = torch.cat([psc, score], dim=1)
        total = all_kv.size(1)
        shared_attn.src_ratio = self.ratio
        n_groups = total // self.ratio
        if n_groups == 0:
            self._pending_kv, self._pending_score = all_kv, all_sc
            return None
        body = all_kv[:, : n_groups * self.ratio]
        bsc = all_sc[:, : n_groups * self.ratio]
        groups = body.unflatten(1, (-1, self.ratio))
        gscore = bsc.unflatten(1, (-1, self.ratio))
        pooled = (groups * gscore.softmax(dim=2)).sum(dim=2)
        rest = total - n_groups * self.ratio
        self._pending_kv = all_kv[:, n_groups * self.ratio :].clone()
        self._pending_score = all_sc[:, n_groups * self.ratio :].clone()
        assert self._pending_kv.size(1) == rest
        return rms_norm(pooled, self.norm_w, self.eps).to(x.dtype)


def select_candidate_blocks(logits, compress_lens, topk_blocks: int, block_size: int) -> torch.Tensor:
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    if isinstance(last, torch.Tensor) and last.dim() > 0:
        # per-query compress_lens (chunk/prefill): [s] -> [s, 1] so the block arange
        # broadcasts to [s, num_blocks] (official: compress_lens broadcasts against
        # logits' leading dims).
        last = last.unsqueeze(-1)
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer:
    """Sparse top-k selector over compressed positions; supports chunked prefill."""

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, freqs_cis: torch.Tensor):
        p = f"layers.{layer_id}.attn.indexer."
        self.layer_id = layer_id
        self.ratio = cfg.compress_ratios[layer_id]
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.index_topk = cfg.index_topk
        self.softmax_scale = self.head_dim ** -0.5
        self.owns_k = layer_id in cfg.kv_source_layer_ids
        self.is_candidate_source = layer_id == cfg.candidate_source_layer_id
        self.uses_candidates = 0 <= cfg.candidate_source_layer_id < layer_id
        self.candidate_topk_blocks = cfg.candidate_topk_blocks
        self.candidate_block_size = cfg.candidate_block_size
        self.freqs_cis = freqs_cis
        self.wq_b = dequant_fp8_block(store.get(p + "wq_b.weight"), store.get(p + "wq_b.scale")).float()
        self.weights_proj = store.get(p + "weights_proj.weight").float()
        if self.owns_k:
            self.wk = store.get(p + "wk.weight").float()
            self.k_norm = store.get(p + "k_norm.weight")
        self._states: dict[int, dict] = {}
        self._active_layer = 0
        self.eps = cfg.rms_norm_eps

    def _st(self) -> dict:
        st = self._states.get(self._active_layer)
        if st is None:
            st = self._states[self._active_layer] = {}
        return st

    @property
    def _k_cache(self):
        return self._st().get("k")

    @_k_cache.setter
    def _k_cache(self, value):
        self._st()["k"] = value

    def reset(self) -> None:
        self._k_cache = None

    def to(self, device):
        self.wq_b = self.wq_b.to(device)
        self.weights_proj = self.weights_proj.to(device)
        self.freqs_cis = self.freqs_cis.to(device)
        self.k_norm = self.k_norm.to(device)
        if self.owns_k:
            self.wk = self.wk.to(device)
        if self._k_cache is not None:
            self._k_cache = self._k_cache.to(device)
        return self

    def _kcache(self, b: int, need: int, device) -> torch.Tensor:
        if self._k_cache is None or self._k_cache.shape[0] != b:
            self._k_cache = torch.zeros(b, max(need, 1), self.head_dim, dtype=torch.bfloat16, device=device)
        elif self._k_cache.shape[1] < need:
            grown = torch.zeros(b, need, self.head_dim, dtype=self._k_cache.dtype, device=self._k_cache.device)
            grown[:, : self._k_cache.shape[1]] = self._k_cache
            self._k_cache = grown
        return self._k_cache

    def forward(self, x, qr, latent, start_pos: int, offset: int):
        b, s, _ = x.shape
        ratio = self.ratio if self.owns_k else (shared_attn.src_ratio or self.ratio)
        rd, end_pos = self.rope_dim, start_pos + s
        groups_so_far = start_pos // ratio

        if self.owns_k and latent is not None:
            # latent groups start at global group index start_pos//ratio (they align to group boundaries)
            n_new = latent.size(1)
            gpos = start_pos + torch.arange(n_new, device=x.device) * ratio
            freqs = self.freqs_cis[gpos]
            k = rms_norm(F.linear(latent.float(), self.wk), self.k_norm, self.eps).to(torch.bfloat16)
            apply_rope(k[..., -rd:], freqs)
            cache = self._kcache(b, groups_so_far + n_new, x.device)
            cache[:, groups_so_far : groups_so_far + n_new] = k
            self._k_cache = cache
        if self.owns_k:
            shared_attn.index_k = self._k_cache
        import os as _os
        if _os.environ.get("TI_DEBUG_IDX"):
            print(f"[idx] start={start_pos} s={s} owns_k={self.owns_k} latent={None if latent is None else tuple(latent.shape)} "
                  f"kcache={None if self._k_cache is None else tuple(self._k_cache.shape)} shared={None if shared_attn.index_k is None else tuple(shared_attn.index_k.shape)}", flush=True)

        q = F.linear(qr.float(), self.wq_b).view(b, s, self.n_heads, self.head_dim)
        apply_rope(q[..., -rd:], self.freqs_cis[start_pos:end_pos])

        t = end_pos // ratio
        if t <= 0:
            return torch.empty(b, s, 0, dtype=torch.int32, device=x.device)
        index_k = shared_attn.index_k[:, :t]
        weights = F.linear(x.float(), self.weights_proj) * (self.softmax_scale * self.n_heads ** -0.5)
        score = torch.einsum("bshd,btd->bsht", q, index_k.float())
        score = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)

        # a query at global p can see compressed positions < (p+1)//ratio
        qpos = torch.arange(start_pos, end_pos, device=x.device).unsqueeze(-1)  # [s,1]
        per_query_lens = (qpos + 1) // ratio  # [s,1]
        ar = torch.arange(t, device=x.device).view(1, t)  # [1, t]
        mask = (ar >= per_query_lens).unsqueeze(0)  # [1, s, t]
        score = score.masked_fill(mask, -torch.inf)
        compress_lens = per_query_lens.squeeze(-1)  # [s]

        if self.is_candidate_source:
            shared_attn.candidates = select_candidate_blocks(
                score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            score = score.masked_fill(~shared_attn.candidates, -torch.inf)

        topk = min(self.index_topk, t)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        valid = idxs < per_query_lens.view(per_query_lens.size(0), 1).unsqueeze(0)
        return torch.where(valid, idxs + offset, -1).int()
