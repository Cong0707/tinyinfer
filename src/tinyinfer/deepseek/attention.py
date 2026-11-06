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

from ..config import TinyConfig
from ..kernels.sparse_attn import sparse_attn_dispatch
from ..nn import apply_rope, precompute_freqs_cis, rms_norm, window_idxs_general
from ..quant import dequant_fp8_block
from ..rope_cache import get_table, table_key
from ..weights import ShardStore


class MLAWindowAttention:
    """All weights of one attention layer, dequantized to bf16; supports prefill and decode.

    Covers layers with compress_ratio == 0 (pure sliding window). The KV cache is a
    ring buffer of window_size rows, exactly like the official _window_kv().
    """

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, *, quant: bool = False,
                 cache_wo_a: bool = False, tp=None, plan_fn=None):
        self.quant = quant
        self.cache_wo_a = cache_wo_a
        self.tp = tp          # tensor-parallel shard (None / world 1 = untouched)
        self.plan_fn = plan_fn  # (name, out_features) -> [(device, (lo, hi))] or None
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
            from ..linear import QuantLinear

            self.wq_a = QuantLinear(store.get(p + "wq_a.weight"), store.get(p + "wq_a.scale"))
            self.wq_b = QuantLinear(store.get(p + "wq_b.weight"), store.get(p + "wq_b.scale"))
            self.wkv = QuantLinear(store.get(p + "wkv.weight"), store.get(p + "wkv.scale"))
            # q_a and kv read the same x, so one GEMM over the stacked weights replaces two
            # cublasLt setups (0.65ms each). The concatenated copy is the device-resident
            # one; the two originals stay on the host and are not migrated.
            self.wqkv = QuantLinear(
                torch.cat([self.wq_a.weight, self.wkv.weight], dim=0),
                torch.cat([self.wq_a.scale, self.wkv.scale], dim=0))
            self.wo_a = QuantLinear(store.get(p + "wo_a.weight"), store.get(p + "wo_a.scale"),
                                    cache_bf16=cache_wo_a)
            self.wo_b = QuantLinear(store.get(p + "wo_b.weight"), store.get(p + "wo_b.scale"))
            if plan_fn is not None:
                self._apply_plan(plan_fn)
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
        from ..rope_cache import get_table, table_key

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

    def _apply_plan(self, plan_fn) -> None:
        """Split this layer's projections across cards, in place.

        plan_fn(name, out_features) returns [(device, (lo, hi)), ...] or None. Every
        projection here is row-parallel by construction -- it owns whole output rows --
        so nothing needs reducing; each card computes its slice and QuantLinear
        concatenates. wqkv is the stacked (q_a, kv) projection and splits the same way:
        the slice boundaries fall inside it, and concatenating the pieces restores the
        original row order, which is what the forward pass indexes into.
        """
        from ..tp import shard_quant_linear

        def split(name, ql):
            if not hasattr(ql, "weight") or ql.weight.dim() != 2:
                return ql
            plan = plan_fn(name, int(ql.weight.shape[0]))
            if not plan or len(plan) <= 1:
                return ql
            return shard_quant_linear(ql.weight, ql.scale, plan)

        # A plan of one card means one range per tensor, which is what the layer already
        # is: nothing to split and nothing to rebuild. Returning early is not just an
        # optimisation -- rebuilding the stacked projection below produces a NEW
        # QuantLinear with freshly concatenated (and therefore differently laid out)
        # storage, so an unnecessary rebuild changed the GEMM selection and with it the
        # engine output on a single card.
        if not any(len(plan_fn(n, int(getattr(self, n).weight.shape[0])) or ()) > 1
                   for n in ("wq_a", "wq_b", "wkv", "wo_b")
                   if hasattr(getattr(self, n, None), "weight")
                   and getattr(self, n).weight.dim() == 2):
            return

        # Order matters: the stacked projection has to be built from the *whole* weights,
        # so it happens before anything is split (after a split, .weight is only the
        # first card's slice and the concatenation would silently use a fragment).
        stacked = None
        if getattr(self, "wqkv", None) is not None:
            try:
                stacked = QuantLinear(
                    torch.cat([self.wq_a.weight, self.wkv.weight], dim=0),
                    torch.cat([self.wq_a.scale, self.wkv.scale], dim=0))
            except Exception:
                stacked = None
        self.wq_a = split("wq_a", self.wq_a)
        self.wq_b = split("wq_b", self.wq_b)
        self.wkv = split("wkv", self.wkv)
        # wo_a is [o_groups, o_lora_rank, r] and is consumed as einsum("bsgd,grd->bsgr"):
        # the middle dimension IS the contracted one, so splitting it is column-parallel
        # over the o-lora rank, and wo_b's input dimension must shrink by the same amount.
        # Together they make the attention output a sum of per-card partial products,
        # which is why this pair (unlike the others) needs a reduce at the end.
        _wa = getattr(self.wo_a, "weight", None)
        _plan_oa = plan_fn("wo_a", int(self.o_lora_rank))
        if _wa is not None and _plan_oa and len(_plan_oa) > 1:
            from ..tp import asplit  # noqa: F401  (only used for its alignment constant)
            from . import tp as _tpmod  # noqa: F401
            from ..tp import shard_quant_linear as _sql
            # slice along dim 1 (o_lora_rank), per card
            _parts = []
            _sc = getattr(self.wo_a, "scale", None)
            for _dev, (_lo, _hi) in _plan_oa:
                _tdev = torch.device(f"cuda:{_dev}") if _wa.is_cuda else _wa.device
                _w = _wa[:, _lo:_hi].to(_tdev).contiguous()
                _s = _sc[:, _lo:_hi].to(_tdev).contiguous() if _sc is not None else None
                _parts.append((_tdev, _w, _s))
            self.wo_a = type(self.wo_a)(_parts[0][1], _parts[0][2], parts=_parts)
            self.o_lora_rank = sum(int(p[1].shape[1]) for p in _parts)
            self._wo_parallel = True
            # wo_b stays whole: with wo_a's contraction split, every card's wo_a part
            # already has the full output width, so wo_b consumes the full o-lora vector
            # once. Splitting it on its input as well would make each card read a
            # different slice of the same vector and the sum would be wrong.
        else:
            self.wo_b = split("wo_b", self.wo_b)
        # The stacked projection is its own split: its rows are q_a's rows followed by
        # kv's, so slicing it by output row keeps that order and forward() still finds the
        # q_lora boundary at wq_a.out_features (the sum over parts).
        # A None here means the two separate projections are used instead, which forward()
        # already handles -- that is the same branch a non-quantised build takes.

    def _shard_for_tp(self, tp) -> None:
        """Split MLA over ranks: q_lora and the heads are column-parallel, wo is
        row-parallel (all-reduced after wo_b). wkv is replicated because this model has
        a single KV head -- sharding it would need a different attention layout."""
        from ..tp import shard_linear

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
        # wo_a is stored [o_groups * o_lora_rank, dim] and wo_b [dim, o_groups * o_lora_rank]:
        # the o_lora axis is INSIDE each group, so both sides need the group-major reshape
        # (shard_grouped). The previous version only split when the stored weight happened
        # to be 3-D, which it never is, so tp>1 left wo_a unsplit while wo_b was cut by a
        # contiguous run of the axis -- two different layouts of the same quantity.
        from ..tp import shard_grouped

        w = getattr(self.wo_a, "weight", None)
        if w is not None:
            rlo, rhi = tp.rows(self.o_lora_rank)
            if isinstance(self.wo_a, torch.Tensor):
                gr, dim = w.shape
                self.wo_a = (w.view(self.o_groups, gr // self.o_groups, dim)[:, rlo:rhi]
                             .contiguous().view(-1, dim))
            else:
                self.wo_a = shard_grouped(self.wo_a, tp, self.o_groups, axis="out")
            self.o_lora_rank = rhi - rlo
        # wo_b consumes the sliced o_lora dimension (row-parallel input) and its output
        # is summed across ranks
        self.wo_b = shard_grouped(self.wo_b, tp, self.o_groups, axis="in")
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

    def forward(self, x: torch.Tensor, start_pos: int = 0,
                positions: torch.Tensor | None = None) -> torch.Tensor:
        # positions is accepted for interface parity with the compressed path (a batch
        # may mix both layer kinds); this layer's cache is per-row already, and the
        # window indices are derived from start_pos plus the row's own offset.
        b, s, _ = x.shape
        assert start_pos + s <= self.freqs_cis.size(0), "sequence exceeds rotated position table"
        # one shared table per (dim, seqlen, yarn args): resolving it through the cache
        # (rather than .to() per instance) keeps a single device copy instead of 40
        self.freqs_cis = get_table(self._rope_key, self._rope_factory, x.device)
        freqs = self.freqs_cis[start_pos:start_pos + s]

        def _lin(w, x_):
            return w.forward(x_) if hasattr(w, "forward") else F.linear(x_, w)

        _wqkv = getattr(self, "wqkv", None)
        if _wqkv is not None:
            _qkv = _lin(_wqkv, x)
            _q_lora = self.wq_a.out_features
            qr = rms_norm(_qkv[..., :_q_lora], self.q_norm, self.eps)
            q = _lin(self.wq_b, qr).view(b, s, self.n_local_heads, self.head_dim)
            apply_rope(q[..., -self.rope_dim:], freqs)
            _kv_raw = _qkv[..., _q_lora:]
        else:
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
        if kv is None and __import__("os").environ.get("TI_FUSED_KV2", "1") != "0":
            # CUDA fusion, bit-identical to the pair (the tilelang norm_rope above is the
            # one that was ~4e-3 off and could not be enabled)
            from .deepseek_kernels import norm_rope_cuda

            kv = norm_rope_cuda(_kv_raw, self.kv_norm, freqs, self.eps, self.rope_dim)
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
        if getattr(self, "quant", False) and kv.is_cuda:
            # the reference ALWAYS round-trips the window KV through fp8 (post-RoPE):
            # that is the form the cache holds. The old "kv.dtype != bfloat16" guard
            # silently skipped it for bf16 inputs, which is where layer 0 diverged.
            from ..kernels.quant import act_quant
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
            if positions is not None and positions.numel() == b and b > 1:
                # Rows at different depths need their own mask AND their own history
                # length: the buffer is [b, n_old+s, D] with one shared n_old, but a row
                # that just joined has no history while one that has been decoding for a
                # while has a full window. Encoding that per row is what the index tensor
                # is for, so old_len is derived from each row's own position instead of
                # from the shared buffer width.
                from ..nn import window_idxs_per_row
                old_per_row = [max(0, min(self.window, int(p) - (s - 1)))
                               for p in positions.tolist()]
                idxs = window_idxs_per_row(positions, s, old_per_row, self.window, x.device)
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
            from ..quant import dequant_fp8_block
            wo_a = dequant_fp8_block(self.wo_a.weight, self.wo_a.scale)
        else:
            wo_a = self.wo_a
        wo_a = wo_a.view(self.o_groups, self.o_lora_rank, -1)
        # the official Attention quantises o to fp8 (and dequantises it back) before the
        # grouped einsum; skipping the round trip is a ~4.8% output difference, so it is
        # part of the reference semantics rather than an optimisation
        if getattr(self, "quant", False) and o.is_cuda:
            from ..kernels.quant import act_quant
            o = act_quant(o, 32, "ue8m0", torch.float8_e8m0fnu, True)
        if getattr(self, "_wo_parallel", False) and getattr(self.wo_a, "is_sharded", False):
            # wo_a's contracted dimension (o_lora_rank) is what gets split, so each card
            # produces a partial product of the SAME output shape; the pieces are summed
            # after wo_b. wo_b is therefore NOT split on its input here -- every card
            # would then see a different slice of the o-lora vector and the sum would be
            # wrong. It keeps whole rows and runs once, on the card that owns the
            # accumulator. (Splitting wo_b on its input *and* wo_a on o_lora is the
            # textbook column+row pair, but that needs an all-reduce per layer and the
            # point of this path is to avoid one.)
            _acc = None
            for _dev, _w, _s in self.wo_a.parts:
                _tgt = _w.device
                _o = o if o.device == _tgt else o.to(_tgt, non_blocking=True)
                _y = torch.einsum("bsgd,grd->bsgr", _o, _w)
                if _acc is None:
                    _acc = _y
                else:
                    _acc = _acc + _y.to(_acc.device, non_blocking=True)
            return _lin(self.wo_b, _acc.flatten(2))
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out = _lin(self.wo_b, o.flatten(2))
        if getattr(self, "tp_world", 1) > 1:
            from ..tp import all_reduce_sum

            out = all_reduce_sum(out)
        return out