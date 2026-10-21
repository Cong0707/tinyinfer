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
from .rope_cache import get_table, table_key
from .weights import ShardStore


class MLAWindowAttention:
    """All weights of one attention layer, dequantized to bf16; supports prefill and decode.

    Covers layers with compress_ratio == 0 (pure sliding window). The KV cache is a
    ring buffer of window_size rows, exactly like the official _window_kv().
    """

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, *, quant: bool = False,
                 cache_wo_a: bool = False, tp=None):
        self.quant = quant
        self.cache_wo_a = cache_wo_a
        self.tp = tp          # tensor-parallel shard (None / world 1 = untouched)
        p = f"layers.{layer_id}.attn."
        self.eps = cfg.rms_norm_eps
        self.n_heads = cfg.n_heads
        self.n_local_heads = cfg.n_heads
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
        # DeepSeek's own definition (config.json -> rope_scaling): YaRN only applies to
        # the compressed-KV layers; pure sliding-window layers rotate at the base theta
        # with the scaling disabled.
        _rs = cfg.rope_scaling or {}
        from .rope_cache import get_table, table_key

        _kargs = (self.rope_dim, min(cfg.max_position_embeddings, 1_048_576), 0,
                  cfg.rope_theta, int(_rs.get("factor", 1)),
                  float(_rs.get("beta_fast", 32.0)), float(_rs.get("beta_slow", 1.0)))
        # a model constant shared by every layer; never streamed as a weight
        self._rope_key = table_key(*_kargs)
        self._rope_factory = (lambda _a=tuple(_kargs): precompute_freqs_cis(*_a))
        self.freqs_cis = get_table(self._rope_key, self._rope_factory)
        # per-layer runtime state; a shared streaming slot serves several layers
        self._states: dict[int, dict] = {}
        self._active_layer = 0
        if tp is not None and tp.world > 1:
            self._shard_for_tp(tp)

    def _shard_for_tp(self, tp) -> None:
        """Split MLA over ranks: q_lora and the heads are column-parallel, wo is
        row-parallel (all-reduced after wo_b). wkv is replicated because this model has
        a single KV head -- sharding it would need a different attention layout."""
        from .tp import shard_linear

        if self.n_heads % tp.world:
            raise ValueError(f"n_heads {self.n_heads} not divisible by tp {tp.world}")
        self.n_local_heads = self.n_heads // tp.world
        # q_lora: wq_a output is sliced; its RMSNorm weight follows
        self.wq_a = shard_linear(self.wq_a, tp, axis="out")
        lo, hi = tp.columns(self.q_norm.numel())
        self.q_norm = self.q_norm[lo:hi].contiguous()
        # heads: wq_b rows are n_heads*head_dim, so a head block is a row slice
        self.wq_b = shard_linear(self.wq_b, tp, axis="out")
        lo, hi = tp.columns(self.n_heads)
        self.sink = self.sink[lo:hi].contiguous()
        # wo_a is grouped as (o_groups, o_lora_rank, -1): slice the o_lora dimension
        w = getattr(self.wo_a, "weight", None)
        if w is not None and w.dim() == 3:
            gr, r, rest = w.shape
            rlo, rhi = tp.rows(r)
            if isinstance(self.wo_a, torch.Tensor):
                self.wo_a = w[:, rlo:rhi].contiguous()
            else:
                self.wo_a.weight = w[:, rlo:rhi].contiguous()
                s = getattr(self.wo_a, "scale", None)
                if s is not None and s.dim() >= 2:
                    self.wo_a.scale = s[:, rlo:rhi].contiguous()
            self.o_lora_rank = rhi - rlo
        # wo_b consumes the sliced o_lora dimension (row-parallel input) and its output
        # is summed across ranks
        self.wo_b = shard_linear(self.wo_b, tp, axis="in")
        self.tp_world = tp.world
        self.tp_rank = tp.rank
        self.tp = tp


    def _st(self) -> dict:
        """Per-layer runtime state; a shared streaming slot serves several layers."""
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
    def _kv_win_buf(self):
        """Pre-allocated sliding-window buffer (fixed address for CUDA graphs)."""
        return self._st().get("winbuf")

    @_kv_win_buf.setter
    def _kv_win_buf(self, value):
        self._st()["winbuf"] = value

    @property
    def _decode_idxs(self):
        return self._st().get("idxs")

    @_decode_idxs.setter
    def _decode_idxs(self, value):
        self._st()["idxs"] = value

    def reset(self) -> None:
        # one attention instance is shared by several streamed layers (the slot's
        # _active_layer switches per layer), so clearing only the active state left the
        # other layers' windows alive and the next run started from stale KV
        self._states.clear()

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        b, s, _ = x.shape
        assert start_pos + s <= self.freqs_cis.size(0), "sequence exceeds rotated position table"
        # one shared table per (dim, seqlen, yarn args): resolving it through the cache
        # (rather than .to() per instance) keeps a single device copy instead of 40
        self.freqs_cis = get_table(self._rope_key, self._rope_factory, x.device)
        freqs = self.freqs_cis[start_pos:start_pos + s]

        def _lin(w, x_):
            return w.forward(x_) if hasattr(w, "forward") else F.linear(x_, w)

        qr = rms_norm(_lin(self.wq_a, x), self.q_norm, self.eps)
        q = _lin(self.wq_b, qr).view(b, s, self.n_local_heads, self.head_dim)
        apply_rope(q[..., -self.rope_dim:], freqs)

        _kv_raw = _lin(self.wkv, x)
        kv = None
        if (getattr(self, "quant", False) and _kv_raw.is_cuda and _kv_raw.size(0) == 1
                and __import__("os").environ.get("TI_FUSED_KV") == "1"):
            # DeepSeek's fused norm+rope (deepseek_kernels.norm_rope): one kernel instead
            # of two torch ops. Off by default: the kernel's reduction order differs from
            # torch's by ~4e-3, and levels 0-2 must stay bit-comparable to the reference.
            from .deepseek_kernels import norm_rope as _fused_norm_rope
            kv = _fused_norm_rope(_kv_raw, self.kv_norm, freqs, self.eps, self.rope_dim)
        if kv is None:
            kv = rms_norm(_kv_raw, self.kv_norm, self.eps)
            _fused = None
            # opt-in: the fused path is verified bit-exact on synthetic inputs, but the
            # end-to-end output regressed when it was always on, so it stays behind a flag
            # until the integration difference is explained
            if (getattr(self, "quant", False) and kv.is_cuda
                    and __import__("os").environ.get("TI_FUSED_KV") == "1"):
                # rope + fp8 round trip in one kernel (deepseek_kernels.rope_quant):
                # exact against the two-op reference (verified 0.0)
                from .deepseek_kernels import rope_quant_roundtrip
                _fused = rope_quant_roundtrip(kv, freqs, self.rope_dim)
            if _fused is not None:
                kv = _fused
            else:
                apply_rope(kv[..., -self.rope_dim:], freqs)
        # the reference quantises the window KV to fp8 (and back) AFTER RoPE -- that is
        # the form the cache holds. Order matters: skipping the round trip is a ~4.7%
        # output difference, and doing it before the rotation is a different value again.
        if getattr(self, "quant", False) and kv.is_cuda and kv.dtype != torch.bfloat16:
            from .official_kernel_ref import act_quant
            kv = act_quant(kv, 32, "ue8m0", torch.float8_e8m0fnu, True)

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
        # keep a VIEW into the pre-allocated buffer (no copy, fixed address once the
        # window is full): .contiguous() here allocated a fresh tensor every token,
        # which is exactly what prevents CUDA graph capture of the decode step
        self._kv_window = kv_att[:, -self.window:]

        o = sparse_attn_dispatch(q, kv_att, self.sink, idxs, self.scale)
        apply_rope(o[..., -self.rope_dim:], freqs, inverse=True)

        o = o.view(b, s, self.o_groups, -1)
        if hasattr(self.wo_a, "weight"):  # fp8 resident: dequant the grouped matrix just for this einsum
            from .quant import dequant_fp8_block
            wo_a = dequant_fp8_block(self.wo_a.weight, self.wo_a.scale)
        else:
            wo_a = self.wo_a
        wo_a = wo_a.view(self.o_groups, self.o_lora_rank, -1)
        # the official Attention quantises o to fp8 (and dequantises it back) before the
        # grouped einsum; skipping the round trip is a ~4.8% output difference, so it is
        # part of the reference semantics rather than an optimisation
        if getattr(self, "quant", False) and o.is_cuda:
            from .official_kernel_ref import act_quant
            o = act_quant(o, 32, "ue8m0", torch.float8_e8m0fnu, True)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out = _lin(self.wo_b, o.flatten(2))
        if getattr(self, "tp_world", 1) > 1:
            from .tp import all_reduce_sum

            out = all_reduce_sum(out)
        return out