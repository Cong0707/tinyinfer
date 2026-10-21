"""MLA sliding-window attention for DeepSeek-V4.1 (window-only path, compress_ratio == 0).

Reference implementations (consult when fixing this file):
  * DeepInfer  deepinfer/model_executor/models/v4_modules/attn.py
        (MLA attention: wq_a/wq_b/wkv/wo_a/wo_b layout, q_norm/kv_norm, attn_sink,
         sparse_attn dispatch, decode_mla_mqa_sparse)
  * Official   reference/official/kernel.py (sparse_attn, _window_kv, rope application)
  * vLLM       vllm/model_executor/layers/attention/mla/ (MLA paging; same math)

Checkpoint tensor names (do not rename):
  layers.{L}.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}, q_norm/kv_norm.weight, attn_sink
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TinyConfig
from .kernels import sparse_attn_dispatch
from .nn import apply_rope, precompute_freqs_cis, rms_norm, window_idxs_general
from .quant import dequant_fp8_block
from .weights import ShardStore


class MLAWindowAttention:
    """All weights of one attention layer, dequantized to bf16; supports prefill and decode.

    Covers layers with compress_ratio == 0 (pure sliding window). The KV cache is a
    ring buffer of window_size rows, exactly like the official _window_kv().
    """

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, *, quant: bool = False,
                 cache_wo_a: bool = False):
        self.quant = quant
        self.cache_wo_a = cache_wo_a
        p = f"layers.{layer_id}.attn."
        self.eps = cfg.rms_norm_eps
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.o_groups = cfg.o_groups
        self.o_lora_rank = cfg.o_lora_rank
        self.window = cfg.sliding_window

        if quant:
            from .linear import QuantLinear

            self.wq_a = QuantLinear(store.get(p + "wq_a.weight"), store.get(p + "wq_a.scale"))
            self.wq_b = QuantLinear(store.get(p + "wq_b.weight"), store.get(p + "wq_b.scale"))
            self.wkv = QuantLinear(store.get(p + "wkv.weight"), store.get(p + "wkv.scale"))
            self.wo_a = QuantLinear(store.get(p + "wo_a.weight"), store.get(p + "wo_a.scale"),
                                    cache_bf16=cache_wo_a)
            self.wo_b = QuantLinear(store.get(p + "wo_b.weight"), store.get(p + "wo_b.scale"))
        else:
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
            min(cfg.max_position_embeddings, 1_048_576),
            0,
            cfg.rope_theta,
            16,
            32.0,
            1.0,
        )
        # per-layer runtime state; a shared streaming slot serves several layers
        self._states: dict[int, dict] = {}
        self._active_layer = 0

    def _st(self) -> dict:
        st = self._states.get(self._active_layer)
        if st is None:
            st = self._states[self._active_layer] = {}
        return st

    @property
    def _kv_window(self):
        return self._st().get("kv")

    @_kv_window.setter
    def _kv_window(self, value):
        self._st()["kv"] = value

    @property
    def _kv_buf(self):
        return self._st().get("buf")

    @_kv_buf.setter
    def _kv_buf(self, value):
        self._st()["buf"] = value

    @property
    def _decode_idxs(self):
        return self._st().get("idxs")

    @_decode_idxs.setter
    def _decode_idxs(self, value):
        self._st()["idxs"] = value

    def reset(self) -> None:
        self._kv_window = None
        self._kv_buf = None
        self._decode_idxs = None

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        b, s, _ = x.shape
        assert start_pos + s <= self.freqs_cis.size(0), "sequence exceeds rotated position table"
        freqs = self.freqs_cis[start_pos:start_pos + s]

        def _lin(w, x_):
            return w.forward(x_) if hasattr(w, "forward") else F.linear(x_, w)

        qr = rms_norm(_lin(self.wq_a, x), self.q_norm, self.eps)
        q = _lin(self.wq_b, qr).view(b, s, self.n_heads, self.head_dim)
        apply_rope(q[..., -self.rope_dim:], freqs)

        kv = rms_norm(_lin(self.wkv, x), self.kv_norm, self.eps)
        apply_rope(kv[..., -self.rope_dim:], freqs)

        old = self._kv_window
        if old is None or old.shape[0] != b or old.device != x.device:
            old = kv.new_empty(b, 0, self.head_dim)
        # preallocated staging buffer: shift the history left and append the new chunk in place
        cap = self.window + s
        buf = self._kv_buf
        if buf is None or buf.shape[0] != b or buf.shape[1] < cap or buf.dtype != kv.dtype:
            buf = kv.new_empty(b, cap, self.head_dim)
            self._kv_buf = buf
        n_old = old.size(1)
        if n_old > 0:
            # old may alias this very buffer (chunked prefill beyond the window): clone
            # the source before writing so the copy does not read its own destination.
            buf[:, :n_old].copy_(old.clone())
        buf[:, n_old:n_old + s].copy_(kv)
        kv_att = buf[:, : n_old + s]
        full = s == 1 and n_old == self.window
        if full and self._decode_idxs is not None:
            idxs = self._decode_idxs
        else:
            idxs = window_idxs_general(start_pos, s, old.size(1), self.window, x.device).unsqueeze(0).expand(b, -1, -1)
            if full:
                # decode with a full linear history: slots 1..window is a fixed template
                self._decode_idxs = idxs
        self._kv_window = kv_att[:, -self.window:].contiguous()

        o = sparse_attn_dispatch(q, kv_att, self.sink, idxs, self.scale)
        apply_rope(o[..., -self.rope_dim:], freqs, inverse=True)

        o = o.view(b, s, self.o_groups, -1)
        if hasattr(self.wo_a, "weight"):  # fp8 resident: dequant the grouped matrix just for this einsum
            from .quant import dequant_fp8_block
            wo_a = dequant_fp8_block(self.wo_a.weight, self.wo_a.scale)
        else:
            wo_a = self.wo_a
        wo_a = wo_a.view(self.o_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return _lin(self.wo_b, o.flatten(2))
