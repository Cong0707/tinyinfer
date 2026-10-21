"""MLA attention with compressed KV (compress_ratio > 0), chunked-prefill capable.

Window KV uses a linear rolling history; compressed KV is a global group-indexed cache
shared from kv_source layers. Supports arbitrary (start_pos, seqlen) chunks and decode.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .compressor import Compressor, Indexer, shared_attn
from .config import TinyConfig
from .kernels import sparse_attn_dispatch
from .linear import QuantLinear
from .nn import apply_rope, precompute_freqs_cis, rms_norm, window_idxs_general
from .quant import dequant_fp8_block
from .rope_cache import get_table, table_key
from .weights import ShardStore


class MLACompressedAttention:
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, *, quant: bool = False,
                 rt=None, tp=None):
        p = f"layers.{layer_id}.attn."
        self.layer_id = layer_id
        self.tp = tp
        self.n_local_heads = cfg.n_heads
        self.rt = rt if rt is not None else shared_attn  # per-kv-source handoff slots
        self.cfg = cfg
        self.quant = quant
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

        if quant:
            self.wq_a = QuantLinear(store.get(p + "wq_a.weight"), store.get(p + "wq_a.scale"))
            self.wq_b = QuantLinear(store.get(p + "wq_b.weight"), store.get(p + "wq_b.scale"))
            self.wkv = QuantLinear(store.get(p + "wkv.weight"), store.get(p + "wkv.scale"))
            self.wo_a = QuantLinear(store.get(p + "wo_a.weight"), store.get(p + "wo_a.scale"))
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
        if tp is not None and tp.world > 1:
            self._shard_for_tp(tp)

        # DeepSeek's own definition (config.json -> rope_scaling): YaRN with factor 16
        # from the 65536-token training context, applied to the compressed-KV layers.
        # The reference model.py defaults these to 0/40 only because its flat ModelArgs
        # never reads the nested rope_scaling block.
        _rs = cfg.rope_scaling or {}
        from .rope_cache import get_table, table_key

        _kargs = (self.rope_dim, min(cfg.max_position_embeddings, 1_048_576),
                  int(_rs.get("original_max_position_embeddings", 65536)),
                  cfg.compress_rope_theta, int(_rs.get("factor", 1)),
                  float(_rs.get("beta_fast", 32.0)), float(_rs.get("beta_slow", 1.0)))
        self._rope_key = table_key(*_kargs)
        self._rope_factory = (lambda _a=tuple(_kargs): precompute_freqs_cis(*_a))
        self.freqs_cis = get_table(self._rope_key, self._rope_factory)
        self.compressor = Compressor(store, layer_id, cfg, quant=quant, rt=rt) if self.is_kv_source else None
        self.indexer = Indexer(store, layer_id, cfg, self.freqs_cis, rt=rt) if self.is_index_source else None
        # per-layer runtime state (shared streaming slots serve several layers)
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
        """Pre-allocated sliding-window staging buffer (fixed address, no per-token cat)."""
        return self._st().get("winbuf")

    @_kv_buf.setter
    def _kv_buf(self, value):
        self._st()["winbuf"] = value

    @property
    def _compress_kv(self):
        return self._st().get("ckv")

    @_compress_kv.setter
    def _compress_kv(self, value):
        self._st()["ckv"] = value

    @property
    def _decode_win_idxs(self):
        return self._st().get("idxs")

    @_decode_win_idxs.setter
    def _decode_win_idxs(self, value):
        self._st()["idxs"] = value

    def reset(self) -> None:
        self._states.clear()
        if self.compressor is not None:
            self.compressor.reset()
        if self.indexer is not None:
            self.indexer.reset()
        self.rt.compress_kv = None
        self.rt.index_k = None
        self.rt.topk_idxs = None
        self.rt.candidates = None

    def to(self, device):
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            mod = getattr(self, name)
            if hasattr(mod, "to"):
                mod.to(device)
            else:
                setattr(self, name, mod.to(device))
        for name in ("q_norm", "kv_norm", "sink"):
            setattr(self, name, getattr(self, name).to(device))
        self.freqs_cis = self.freqs_cis.to(device)
        if self.compressor is not None:
            self.compressor.to(device)
        if self.indexer is not None:
            self.indexer.to(device)
        return self

    def _shard_for_tp(self, tp) -> None:
        """Same MLA split as MLAWindowAttention: q_lora and heads column-parallel,
        wo row-parallel with an all-reduce at the end; wkv replicated (one KV head)."""
        from .tp import shard_linear

        if self.n_heads % tp.world:
            raise ValueError(f"n_heads {self.n_heads} not divisible by tp {tp.world}")
        self.n_local_heads = self.n_heads // tp.world
        self.wq_a = shard_linear(self.wq_a, tp, axis="out")
        lo, hi = tp.columns(self.q_norm.numel())
        self.q_norm = self.q_norm[lo:hi].contiguous()
        self.wq_b = shard_linear(self.wq_b, tp, axis="out")
        lo, hi = tp.columns(self.n_heads)
        self.sink = self.sink[lo:hi].contiguous()
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
        self.wo_b = shard_linear(self.wo_b, tp, axis="in")
        self.tp_world = tp.world
        self.tp_rank = tp.rank

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        import os as _os
        _pt = {} if _os.environ.get("TI_ATTN_PROF") else None
        _t0 = __import__("time").perf_counter() if _pt is not None else 0.0
        b, s, _ = x.shape
        if self.compressor is not None:
            self.compressor._active_layer = self._active_layer
        if self.indexer is not None:
            self.indexer._active_layer = self._active_layer
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
            from .deepseek_kernels import norm_rope as _fused_norm_rope
            kv = _fused_norm_rope(_kv_raw, self.kv_norm, freqs, self.eps, self.rope_dim)
        if kv is None:
            kv = rms_norm(_kv_raw, self.kv_norm, self.eps)
            apply_rope(kv[..., -self.rope_dim:], freqs)
        # the reference quantises the window KV to fp8 (and back) AFTER RoPE -- that is
        # the form the cache holds. Order matters: skipping the round trip is a ~4.7%
        # output difference, and doing it before the rotation is a different value again.
        if getattr(self, "quant", False) and kv.is_cuda:
            from .official_kernel_ref import act_quant
            kv = act_quant(kv, 32, "ue8m0", torch.float8_e8m0fnu, True)

        old = self._kv_window if self._kv_window is not None else kv.new_empty(b, 0, self.head_dim)
        # pre-allocated staging: shift + append in place instead of torch.cat, which
        # allocated a new tensor every token and made the decode step uncapturable
        cap = self.window + s
        buf = self._kv_buf
        if buf is None or buf.shape[0] != b or buf.shape[1] < cap or buf.dtype != kv.dtype:
            buf = kv.new_empty(b, cap, self.head_dim)
            self._kv_buf = buf
        n_old = old.size(1)
        if n_old > 0:
            buf[:, :n_old].copy_(old.clone())
        buf[:, n_old:n_old + s].copy_(kv)
        kv_w = buf[:, :n_old + s]
        full = s == 1 and old.size(1) == self.window
        if full and self._decode_win_idxs is not None:
            win_idxs = self._decode_win_idxs
        else:
            win_idxs = window_idxs_general(start_pos, s, old.size(1), self.window, x.device).unsqueeze(0).expand(b, -1, -1)
            if full:
                self._decode_win_idxs = win_idxs
        # view into the staging buffer, not a per-token allocation (see MLAWindowAttention)
        self._kv_window = kv_w[:, -self.window:]

        # ---- compressed KV ----
        end_pos = start_pos + s
        ratio_now = self.ratio if self.is_kv_source else (self.rt.src_ratio or self.ratio)
        compress_len = end_pos // ratio_now
        if _pt is not None:
            _t1 = __import__("time").perf_counter()
        latent = self.compressor.forward(x, start_pos) if self.compressor is not None else None
        if _pt is not None:
            _pt["compressor"] = (__import__("time").perf_counter() - _t1) * 1e3
        if latent is not None:
            # the reference rotates the latent at its group position and quantises it to fp4
            # (block 16, E4M3 scales) before it enters the shared cache; both steps are part
            # of the stored value, not an optimisation
            if getattr(self, "quant", False) and latent.is_cuda:
                from .nn import apply_rope as _apply_rope
                from .official_kernel_ref import fp4_act_quant
                if start_pos == 0:
                    fr = self.freqs_cis[: s - s % self.ratio : self.ratio]
                else:
                    # one row per latent group: a group stands for the first token of its
                    # window, so group j rotates at position (start_group + j) * ratio.
                    # (A single row only works when the caller passes exactly one group.)
                    _g0 = start_pos // self.ratio
                    _grp = torch.arange(_g0, _g0 + latent.size(1), device=latent.device)
                    fr = self.freqs_cis[_grp * self.ratio]
                # the reference keeps the pre-RoPE latent for the indexer (its KProject runs
                # on the compressor output before rotation) and only the rotated copy goes
                # into the shared cache -- rotating in place fed the indexer rotated keys
                stored = latent.clone()
                _apply_rope(stored[..., -self.rope_dim:], fr)
                fp4_act_quant(stored, 16, True, scale_dtype=torch.float8_e4m3fn)
                latent = stored
            start_group = start_pos // self.ratio
            need = start_group + latent.size(1)
            if self._compress_kv is None:
                # pre-allocate a decode-sized run of groups so the buffer address stays
                # fixed for the lifetime of the conversation (CUDA graph capture needs
                # it); a long prompt still grows once, outside the captured step.
                cap = max(need, int(self.cfg.max_position_embeddings // max(1, self.ratio)))
                cap = min(cap, max(need, 4096))
                self._compress_kv = torch.zeros(b, cap, self.head_dim,
                                                dtype=latent.dtype, device=latent.device)
            elif self._compress_kv.shape[1] < need:
                grown = torch.zeros(b, need, self.head_dim, dtype=latent.dtype, device=latent.device)
                grown[:, : self._compress_kv.shape[1]] = self._compress_kv
                self._compress_kv = grown
            self._compress_kv[:, start_group : start_group + latent.size(1)] = latent
            self.rt.compress_kv = self._compress_kv
        if compress_len <= 0:
            compress_kv = compress_idxs = None
        else:
            compress_kv = self.rt.compress_kv[:, :compress_len]
            if self.indexer is not None:
                if _pt is not None:
                    _t2 = __import__("time").perf_counter()
                compress_idxs = self.indexer.forward(x, qr, latent, start_pos, kv_w.size(1))
                if _pt is not None:
                    _pt["indexer"] = (__import__("time").perf_counter() - _t2) * 1e3
                self.rt.topk_idxs = compress_idxs
            else:
                compress_idxs = self.rt.topk_idxs

        if __import__("os").environ.get("TI_DEBUG_ATTN") and \
                getattr(self, "_active_layer", -1) == 20:
            print(f"[attn20] start={start_pos} s={s} ratio={ratio_now} "
                  f"compress_len={compress_len} "
                  f"kv={None if compress_kv is None else tuple(compress_kv.shape)} "
                  f"idxs={None if compress_idxs is None else tuple(compress_idxs.shape)} "
                  f"win={tuple(win_idxs.shape)}", flush=True)
        if compress_idxs is not None and compress_idxs.size(-1) > 0:
            kv_all = torch.cat([kv_w, compress_kv], dim=1)
            # Right-pad the compressed slots to the configured top-k. Near the start of a
            # sequence t = min(index_topk, end_pos // ratio) is smaller than top_k, and two
            # layers in one layer-major pass then hand the sparse kernel different widths
            # (8 vs 7 columns) -- the layer-major prefill used to die on that cat(). -1 is
            # the kernel's empty-slot marker, so padding changes no value and makes the
            # kernel shape constant.
            _k = int(self.cfg.index_topk)
            if compress_idxs.size(-1) < _k:
                _pad = torch.full((*compress_idxs.shape[:-1], _k - compress_idxs.size(-1)),
                                  -1, dtype=compress_idxs.dtype, device=compress_idxs.device)
                compress_idxs = torch.cat([compress_idxs, _pad], dim=-1)
            idxs_all = torch.cat([win_idxs, compress_idxs], dim=-1)
        else:
            kv_all, idxs_all = kv_w, win_idxs

        if _pt is not None:
            _t3 = __import__("time").perf_counter()
        o = sparse_attn_dispatch(q, kv_all, self.sink, idxs_all, self.scale)
        if _pt is not None:
            _pt["sparse"] = (__import__("time").perf_counter() - _t3) * 1e3
            _pt["total"] = (__import__("time").perf_counter() - _t0) * 1e3
            print(f"[attn-prof] {_pt}", flush=True)
        apply_rope(o[..., -self.rope_dim:], freqs, inverse=True)

        o = o.view(b, s, self.o_groups, -1)
        if hasattr(self.wo_a, "weight"):  # fp8 resident: dequant just this grouped GEMM
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
