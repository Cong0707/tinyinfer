"""Compressed-KV sparse attention path: softmax-gated pooling + indexer top-k.

Mirrors the official Compressor/Indexer (reference/official/model.py) in plain PyTorch:
- Compressor pools compress_ratio consecutive tokens into one KV latent with a learned
  softmax gate (fp32 math, matching training), returning the pre-RoPE latent.
- Indexer runs a small side attention (fp4 query heads vs one shared key per compressed
  position) and keeps the index_topk best compressed positions per query.
- SharedAttentionRuntime carries caches/idx between source layers and their consumers.
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


shared_attn = SharedAttentionRuntime()


class Compressor:
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig):
        p = f"layers.{layer_id}.attn.compressor."
        self.ratio = cfg.compress_ratios[layer_id]
        self.head_dim = cfg.head_dim
        self.eps = cfg.rms_norm_eps
        self.norm_w = store.get(p + "norm.weight")
        # official: ratio>1 pools in fp32, so promote the stored bf16 weights
        self.wkv = store.get(p + "wkv.weight").float()
        self.wgate = store.get(p + "wgate.weight").float() if self.ratio > 1 else None
        self._kv_state: torch.Tensor | None = None
        self._score_state: torch.Tensor | None = None

    def _state(self, b: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._kv_state is None or self._kv_state.shape[0] != b:
            self._kv_state = torch.zeros(b, self.ratio, self.head_dim, dtype=torch.float32, device=device)
            self._score_state = torch.full((b, self.ratio, self.head_dim), float("-inf"), dtype=torch.float32, device=device)
        return self._kv_state, self._score_state

    def reset(self) -> None:
        self._kv_state = None
        self._score_state = None

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        b, s, _ = x.shape
        if self.ratio == 1:
            return rms_norm(F.linear(x.float(), self.wkv), self.norm_w, self.eps).to(x.dtype)
        kv_state, score_state = self._state(b, x.device)
        xf = x.float()
        kv = F.linear(xf, self.wkv)
        score = F.linear(xf, self.wgate)
        if start_pos == 0:
            remainder = s % self.ratio
            cutoff = s - remainder
            if remainder:
                kv_state[:, :remainder] = kv[:, cutoff:]
                score_state[:, :remainder] = score[:, cutoff:]
                kv, score = kv[:, :cutoff], score[:, :cutoff]
            groups = kv.unflatten(1, (-1, self.ratio))
            gscore = score.unflatten(1, (-1, self.ratio))
            pooled = (groups * gscore.softmax(dim=2)).sum(dim=2)
            return rms_norm(pooled, self.norm_w, self.eps).to(x.dtype)
        # decode: fill one slot; pool only when a group completes
        slot = start_pos % self.ratio
        kv_state[:, slot] = kv.squeeze(1)
        score_state[:, slot] = score.squeeze(1)
        if (start_pos + 1) % self.ratio != 0:
            return None
        pooled = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
        return rms_norm(pooled, self.norm_w, self.eps).to(x.dtype)


def select_candidate_blocks(logits, compress_lens, topk_blocks: int, block_size: int) -> torch.Tensor:
    """Two-level top-k level one: keep the best-scoring blocks (official select_candidate_blocks)."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer:
    """Sparse top-k selector over compressed positions (MQA index attention)."""

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
            self._k_cache: torch.Tensor | None = None
        self.eps = cfg.rms_norm_eps

    def reset(self) -> None:
        self._k_cache = None

    def _kcache(self, b: int, compress_len: int, device) -> torch.Tensor:
        need = max(compress_len, 1)
        if self._k_cache is None or self._k_cache.shape[0] != b:
            self._k_cache = torch.zeros(b, need, self.head_dim, dtype=torch.bfloat16, device=device)
        elif self._k_cache.shape[1] < need:
            # grow while keeping the already published index keys
            grown = torch.zeros(b, need, self.head_dim, dtype=self._k_cache.dtype, device=self._k_cache.device)
            grown[:, : self._k_cache.shape[1]] = self._k_cache
            self._k_cache = grown
        return self._k_cache

    def forward(self, x, qr, latent, start_pos: int, offset: int):
        b, s, _ = x.shape
        ratio, rd, end_pos = self.ratio, self.rope_dim, start_pos + s

        if self.owns_k and latent is not None:
            freqs = (
                self.freqs_cis[: s - s % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = rms_norm(F.linear(latent.float(), self.wk), self.k_norm, self.eps).to(torch.bfloat16)
            apply_rope(k[..., -rd:], freqs)
            cache = self._kcache(b, end_pos // ratio, x.device)
            cache[:, start_pos // ratio : start_pos // ratio + k.size(1)] = k
            self._k_cache = cache
        if self.owns_k:
            shared_attn.index_k = self._k_cache

        q = F.linear(qr.float(), self.wq_b).view(b, s, self.n_heads, self.head_dim)
        apply_rope(q[..., -rd:], self.freqs_cis[start_pos:end_pos])

        index_k = shared_attn.index_k[:, : end_pos // ratio]
        weights = F.linear(x.float(), self.weights_proj) * (self.softmax_scale * self.n_heads ** -0.5)
        score = torch.einsum("bshd,btd->bsht", q, index_k.float())
        score = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)  # [b, s, t]

        if start_pos == 0:
            compress_lens = (torch.arange(1, s + 1, device=x.device) // ratio).unsqueeze(-1)
            score.masked_fill_(torch.arange(s // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            shared_attn.candidates = select_candidate_blocks(
                score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            score = score.masked_fill(~shared_attn.candidates, -torch.inf)

        topk = min(self.index_topk, end_pos // ratio, score.size(-1))
        if topk <= 0:
            return torch.empty(b, s, 0, dtype=torch.int32, device=x.device)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()
