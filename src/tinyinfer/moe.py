"""MoE with fp4 routed experts and one fp8 shared expert, loaded straight from HF shards.

Reference implementations (consult when fixing this file):
  * DeepInfer  deepinfer/model_executor/models/v4_modules/mlp.py  (DeepseekMoE:
        sqrtsoftplus gate scoring, n_activated_experts=6, route_scale, shared expert)
  * Official   reference/official/model.py (Gate / Expert FFN) and kernel.py
        (fused_fp4_expert_ffn: gate+up+silu / down, block-scaled fp4)
  * flash-moe  expert fetch/cache/eviction model (384-expert top-6 layout)

Checkpoint tensor names (do not rename):
  layers.{L}.ffn.gate.{weight,bias,bias_vl}
  layers.{L}.ffn.experts.{i}.{w1,w2,w3}.{weight,scale}    fp4 E2M1 + ue8m0
  layers.{L}.ffn.shared_experts.{w1,w2,w3}.{weight,scale} fp8 E4M3
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TinyConfig
from .quant import dequant_fp4_block, dequant_fp8_block
from .weights import ShardStore


class Gate:
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, topk: int = 0):
        p = f"layers.{layer_id}.ffn.gate."
        self.weight = store.get(p + "weight").float()
        self.bias = store.get(p + "bias").float()
        self.bias_vl = store.get(p + "bias_vl").float() if cfg.vision_enabled else None
        # 0 = checkpoint top-k; aggressive tiers pass a smaller k (fewer expert reads,
        # same fp4 format — quality traded, generation still coherent).
        self.topk = int(topk) if topk and 0 < topk < cfg.n_activated_experts else cfg.n_activated_experts
        self.score_func = cfg.scoring_func
        self.norm_topk_prob = cfg.norm_topk_prob
        self.route_scale = cfg.route_scale

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None):
        scores = F.linear(x.float(), self.weight)  # gate_temp == 1.0 in the released config
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        # deterministic top-k: torch.topk's tie order is not stable on CUDA, so the same
        # layer could pick a different expert set on every call (measured: two forwards of
        # an identical input went to disjoint expert sets and the MoE output differed by
        # 0.49). A stable descending argsort always breaks ties by expert index.
        logits = scores + bias
        order = torch.argsort(logits, dim=-1, descending=True, stable=True)
        indices = order[..., :self.topk]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


# Set by Engine.forward: during prefill the expert cache must not admit (it would
# evict the decode working set and compete with the pinned layer slab for host RAM).
_admit_cache = True


class Expert:
    """One SwiGLU FFN; routed experts are fp4, the shared expert is fp8."""

    def __init__(self, w1, w2, w3, swiglu_limit: float):
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.swiglu_limit = swiglu_limit
        self.fp4 = None  # optional (w1,s1,w2,s2,w3,s3) tuple: compute without bf16 dequant

    def to(self, device):
        # torch.Tensor.to returns a NEW tensor; dropping the result left the weights on
        # the CPU (crash: "mat2 is on cpu ... other tensors on cuda")
        for name in ("w1", "w2", "w3"):
            w = getattr(self, name)
            if w is None:
                continue
            if isinstance(w, torch.Tensor):
                setattr(self, name, w.to(device))
            elif hasattr(w, "to"):
                w.to(device)  # QuantLinear.to is in-place
        if self.fp4 is not None:
            self.fp4 = tuple(t.to(device) for t in self.fp4)
        return self

    def forward_fp4(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """fp4 kernel path; M==1 (decode) uses the GEMV kernels (GEMM wastes 31/32 rows)."""
        from .gemm import fp4_linear

        w1, s1, w2, s2, w3, s3 = self.fp4
        # the fused prefill kernel is built for chunk-sized M; at M=2..8 it measured
        # 7.5x slower per layer than the GEMV path (667ms vs 89ms), so only real
        # prefill shapes use it
        if x.size(0) > 8 and x.is_cuda:
            # chunked prefill: one fused kernel does gate+up+silu, another does down+scale
            try:
                from .moe_fused import fused_fp4_expert_ffn
                w = None if weights is None else weights.reshape(-1)
                return fused_fp4_expert_ffn(x, w1, s1, w3, s3, w2, s2, w,
                                            limit=float(self.swiglu_limit))
            except Exception:
                pass
        gate = fp4_linear(x, w1, s1).float()
        up = fp4_linear(x, w3, s3).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        return fp4_linear(y.to(torch.bfloat16), w2, s2)

    def forward_fp4_pq(self, aq: torch.Tensor, asc: torch.Tensor,
                       weights: torch.Tensor | None = None) -> torch.Tensor:
        """fp4 path with an activation quantised once for the whole expert batch.

        The quantiser does not depend on the expert, so serving six experts repeated it
        six times (48 rows at prefill); it is now computed once per layer.
        """
        from .gemv_fp4 import gemv_fp4_m8, gemv_fp4_pq

        # Several tokens of one chunk can route to the same expert: the single-row GEMV
        # flattens a [rows, K] activation into the kernel's [K] slot and reads it wrong,
        # which is what made a multi-token forward disagree with token-by-token decoding.
        def _lin(aqq, assc, w, s, ws=None):
            # The multi-row kernel is exact on synthetic weights (M=1..8, rel 0) but
            # measured 42% off with real expert-34 weights, while the single-row kernel
            # is exact on both. Rows are therefore done one at a time unless the batch
            # kernel is explicitly requested; a chunk routes 1-2 rows per expert anyway.
            if (aqq.dim() == 2 and aqq.size(0) > 1
                    and __import__("os").environ.get("TI_MULTIROW_GEMV") == "1"):
                return gemv_fp4_m8(aqq, assc, w, s)
            if aqq.dim() == 2 and aqq.size(0) > 1:
                rows = [gemv_fp4_pq(aqq[i], assc[i], w, s, ws) for i in range(aqq.size(0))]
                return torch.cat(rows, 0)
            a1 = aqq[0] if aqq.dim() == 2 else aqq
            s1 = assc[0] if assc.dim() == 2 else assc
            return gemv_fp4_pq(a1, s1, w, s, ws)

        w1, s1, w2, s2, w3, s3 = self.fp4
        ws = (None, None, None)
        # NOTE: stacking [w1; w3] into one 2N-row GEMV was tried and removed: the two
        # torch.cat calls it needs cost 768 launches and 22ms of GPU time per token, more
        # than the 18 -> 12 launch saving. The GEMVs are not the bottleneck.
        if aq.dim() == 2 and aq.size(0) == 1:
            # Decode only: the fp32 scale conversion is a 1.5MB elementwise pass and it
            # was done once per matmul (0.18ms x 3 per expert, 130ms per token). Caching
            # it per expert at prefill would hold 3 tensors for every one of the ~2400
            # experts a prompt touches, which OOMs an 8GB card -- so prefill converts as
            # before and decode keeps it for the few experts it serves.
            ws = getattr(self, "_ws_cache", None)
            if ws is None:
                from .gemv_fp4 import _scale

                ws = self._ws_cache = (_scale(s1), _scale(s2), _scale(s3))
        gate = _lin(aq, asc, w1, s1, ws[0]).float()
        up = _lin(aq, asc, w3, s3, ws[2]).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        from .gemm import fp4_linear
        return fp4_linear(y.to(torch.bfloat16), w2, s2)

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        if hasattr(self.w1, "forward"):  # fp8 QuantLinear shared expert
            gate = self.w1.forward(x).float()
            up = self.w3.forward(x).float()
        else:
            x = x.to(self.w1.dtype)
            gate = F.linear(x, self.w1).float()
            up = F.linear(x, self.w3).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        if hasattr(self.w2, "forward"):
            return self.w2.forward(y)
        return F.linear(y.to(self.w2.dtype), self.w2)


class MoE:
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, expert_store=None,
                 max_bf16_experts: int = 8, topk: int = 0, cache_shared_bf16: bool = False):
        self.cfg = cfg
        self.layer_id = layer_id
        self._expert_store = expert_store  # optional tinyinfer.loading.ExpertStore
        # 0 = never keep bf16 copies (production/GPU path); N>0 = bounded reference cache
        self._max_bf16 = int(max_bf16_experts)
        self.gate = Gate(store, layer_id, cfg, topk=topk)
        p = f"layers.{layer_id}.ffn."
        # shared expert stays fp8 on the device (halves its VRAM; official fp8 GEMM path)
        from .linear import QuantLinear
        self.shared = Expert(
            QuantLinear(store.get(p + "shared_experts.w1.weight"), store.get(p + "shared_experts.w1.scale"), cache_bf16=cache_shared_bf16),
            QuantLinear(store.get(p + "shared_experts.w2.weight"), store.get(p + "shared_experts.w2.scale"), cache_bf16=cache_shared_bf16),
            QuantLinear(store.get(p + "shared_experts.w3.weight"), store.get(p + "shared_experts.w3.scale"), cache_bf16=cache_shared_bf16),
            cfg.swiglu_limit,
        )
        self._store = store
        self._expert_cache: dict[int, Expert] = {}
        self._last_ids: list[int] | None = None
        self.swiglu_limit = cfg.swiglu_limit
        self._fused_ok: bool | None = None

    def _expert_from(self, get) -> Expert:
        """Build an Expert from a name->tensor accessor, packed fp4 when on CUDA."""
        target = self.gate.weight.device
        if target.type == "cuda":
            e = Expert(None, None, None, self.swiglu_limit)
            e.fp4 = (
                get("w1.weight").to(target), get("w1.scale").to(target),
                get("w2.weight").to(target), get("w2.scale").to(target),
                get("w3.weight").to(target), get("w3.scale").to(target),
            )
            return e
        return Expert(
            dequant_fp4_block(get("w1.weight"), get("w1.scale")),
            dequant_fp4_block(get("w2.weight"), get("w2.scale")),
            dequant_fp4_block(get("w3.weight"), get("w3.scale")),
            self.swiglu_limit,
        )

    def _expert_from_rec(self, rec) -> Expert:
        """One cudaMemcpyAsync per expert, straight from the host-pool slice.

        Six per-tensor copy_() calls per expert is 36 synchronous pageable copies per
        layer: the link is PCIe 5.0 x8 (measured 26GB/s), but each blocking copy waits for
        the queue to drain, which put ~18ms/layer on the critical path. One async 18.8MB
        DMA is 0.7ms and never blocks.
        """
        target = self.gate.weight.device
        if target.type == "cuda" and hasattr(rec, "tensors_on"):
            t = rec.tensors_on(target)
            e = Expert(None, None, None, self.swiglu_limit)
            e.fp4 = (t["w1.weight"], t["w1.scale"], t["w2.weight"], t["w2.scale"],
                     t["w3.weight"], t["w3.scale"])
            return e
        return self._expert_from(rec.tensor)

    def _make_expert(self, rec) -> Expert:
        """Materialise one expert for the gate's device: fp4 stays packed on CUDA.

        Dequantising on the host costs 74ms per expert (measured), which would dominate
        any CPU-side path, so the packed fp4 tensors go to the device and the fp4 kernel
        consumes them directly.
        """
        return self._expert_from_rec(rec)

    def _expert(self, i: int) -> Expert:
        e = self._expert_cache.get(i)
        if e is None:
            if self._expert_store is not None:
                with self._expert_store.experts(self.layer_id, [i]) as recs:
                    e = self._make_expert(recs[0])
            else:
                p = f"layers.{self.layer_id}.ffn.experts.{i}."
                e = self._expert_from(lambda n: self._store.get(p + n))
            self._expert_cache[i] = e
        return e

    def to(self, device):
        self.shared.to(device)
        self.gate.weight = self.gate.weight.to(device)
        self.gate.bias = self.gate.bias.to(device)
        if self.gate.bias_vl is not None:
            self.gate.bias_vl = self.gate.bias_vl.to(device)
        return self

    def prefetch(self, expert_ids) -> None:
        """Warm RAM and stage predicted experts on VRAM (learned predictor: ~95% hit)."""
        if self._expert_store is not None and expert_ids:
            try:
                self._expert_store.prefetch(self.layer_id, expert_ids)
            except Exception:
                pass

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        import os as _os0
        _prof = _os0.environ.get("TI_MOE_PROF") == "1"
        import time as _tp
        _t_start = _tp.perf_counter() if _prof else 0.0
        self._parts = getattr(self, "_parts", None)
        if _prof and self._parts is None:
            self._parts = {"gate": 0.0, "bincount": 0.0, "sel": 0.0, "serve": 0.0,
                           "fetch": 0.0, "shared": 0.0, "alloc": 0.0}
        shape = x.shape
        xf = x.view(-1, self.cfg.dim)
        _t1 = _tp.perf_counter() if _prof else 0.0
        weights, indices = self.gate.forward(xf, None if image_mask is None else image_mask.flatten())
        if _prof:
            self._parts["gate"] += _tp.perf_counter() - _t1
        y = torch.zeros_like(xf, dtype=torch.float32)
        _t2 = _tp.perf_counter() if _prof else 0.0
        import os as _os
        if _os.environ.get("TI_NO_SYNC") == "1":
            # experiment only: routing needs a host sync (bincount). Skipping it shows how
            # much of the step is the CPU waiting for the GPU to finish this layer.
            active = list(range(min(6, self.cfg.n_routed_experts)))
        else:
            # One small D2H instead of a bincount over 384 classes: decode routes <= 8
            # distinct experts, so the kernel plus its 384-element sync (3.3ms per layer,
            # 130ms per token) buys nothing over fetching the 6 routed ids directly.
            _route = [int(v) for v in indices.reshape(-1).tolist()]
            active = sorted(set(_route))
            if _prof:
                self._parts["bincount"] += _tp.perf_counter() - _t2
        # Expert parallelism: with several tensor-parallel ranks a rank serves only the
        # experts it owns, and the partial outputs are summed at the end (DeepSeek's
        # deployment shards experts rather than splitting each expert's matrices).
        _world = int(getattr(self, "ep_world", 1) or 1)
        _rank = int(getattr(self, "ep_rank", 0) or 0)
        if _world > 1:
            active = [i for i in active if (i % _world) == _rank]
        self._last_ids = list(active)  # already the unique routed set (one host sync total)
        target = xf.device
        # Decode (one token) reads each expert's packed fp4 tiles directly through the
        # single-row GEMV kernel, which is exact (the 42% divergence was the multi-row
        # kernel) and skips materialising ~70MB of bf16 weights per expert. Prefill keeps
        # the bf16 dequant path: at M>8 it uses the fused kernel, and the row-by-row
        # fallback measured 2.7x slower on a 13-token prompt. TI_MOE_FP4=0 restores the
        # old always-bf16 behaviour, TI_MOE_FP4=1 forces the kernel path everywhere.
        _fp4_env = _os0.environ.get("TI_MOE_FP4")
        _single_token = xf.size(0) == 1
        use_fp4_kernel = target.type == "cuda" and (
            _fp4_env == "1" or (_single_token and _fp4_env != "0")
        )
        limit = self._max_bf16

        # one activation quantisation for every routed expert of this layer: the
        # quantiser does not depend on the expert, and per-expert (per-row) quantising
        # repeated it 6x per token / 48x at prefill
        _aq = _asc = None
        if use_fp4_kernel and active:
            try:
                from .gemv_fp4 import quantize_act
                _aq, _asc = quantize_act(xf, xf.size(-1))
            except Exception:
                _aq = _asc = None

        # per-expert (token, slot) selection computed once and reused by both passes
        sels: dict[int, tuple] = {}
        _single = xf.size(0) == 1
        _groups: dict[int, list[int]] = {}
        if _single and not _os.environ.get("TI_NO_SYNC") == "1":
            # decode: one row, six experts -- group them from the route list we already
            # pulled to the host instead of one torch.where launch per expert (6 per
            # layer, 240 per token)
            for _col, _e in enumerate(_route):
                _groups.setdefault(_e, []).append(_col)

        # decode: the routing weight of each expert, summed over the top-k slots it won.
        # Built once per layer (six small tensors) instead of one torch.tensor per expert.
        _wsum: dict[int, object] = {}
        if _single and _groups:
            # a routed expert usually wins exactly one slot; indexing the weight with a
            # python int avoids the 240 torch.tensor() calls per token that building an
            # index tensor per expert cost (~118ms of host time at 320 calls/step)
            for _e, _cols in _groups.items():
                if len(_cols) == 1:
                    _wsum[_e] = weights[0, _cols[0]].view(1, 1)
                else:
                    _wsum[_e] = weights[0][_cols].sum().view(1, 1)

        def sel_of(i: int):
            s = sels.get(i)
            if s is None:
                _cols = _groups.get(i)
                if _cols:
                    rows = torch.zeros(len(_cols), dtype=torch.long, device=indices.device)
                    s = sels[i] = (rows, torch.tensor(_cols, dtype=torch.long,
                                                     device=indices.device))
                else:
                    s = sels[i] = torch.where(indices == i)
            return s

        # 1) start the disk reads for the experts we do not hold BEFORE serving the
        # cached ones, so the SSD fills its queue while the GPU works on the cache.
        # Flare's rule is explicit ("offload never blocks the model worker thread");
        # we had the opposite order, which serialised the two.
        served = []
        # Decode can serve the whole routed set in four hand-written launches (see
        # .moe_gemv): the fp4 weights, the e8m0 scales and the fp8 activation are decoded
        # inside the kernel, so the per-expert tilelang GEMV and the fp32 scale copies it
        # needed both disappear. Only a single-token forward has that shape. The expert
        # cache is consulted inside the fused path too, so the cache pass is skipped here
        # rather than splitting one layer's experts across two code paths.
        _fused = False
        if _single and use_fp4_kernel and _aq is not None and active:
            _ok = self._fused_ok
            if _ok is None:
                from . import moe_gemv

                _ok = (moe_gemv.available()
                       and moe_gemv.supported_shape(self.cfg.moe_inter_dim, self.cfg.dim)
                       and moe_gemv.supported_shape(self.cfg.dim, self.cfg.moe_inter_dim))
                self._fused_ok = _ok
            _fused = bool(_ok)
            if _os0.environ.get("TI_FUSED_STATS") == "1":
                import sys as _sys
                print(f"[moe] layer {self.layer_id}: fused={_fused} "
                      f"single={_single} kernel={use_fp4_kernel} aq={_aq is not None}",
                      file=_sys.stderr, flush=True)
        # the fused path has to hand every routed expert to one set of launches, so it
        # fetches the whole active set rather than splitting off cache hits (reusing a
        # cached Expert there measured WRONG output -- see the note in the fused branch)
        cached_hits = [] if _fused else (
            [i for i in active if self._expert_cache.get(i) is not None] if limit > 0 else [])
        to_fetch = [i for i in active if i not in cached_hits]
        _pre_fetch = None
        _prefetch_ids: list[int] = list(to_fetch)
        if getattr(self, "_async_fetch", False) and self._expert_store is not None and to_fetch:
            try:
                _pre_fetch = self._expert_store.fetch_async(self.layer_id, list(to_fetch),
                                                            cache=_admit_cache)
            except Exception:
                _pre_fetch = None
        # The shared expert does not depend on the routing, so compute it while the routed
        # experts' reads are in flight. It used to be the last statement of this method,
        # which put its ~1ms/layer (plus its 3 GEMM submissions) on the critical path of
        # every decode step.
        _sh0 = _tp.perf_counter() if _prof else 0.0
        if _world == 1 or _rank == 0:
            y += self.shared.forward(xf)
        if _prof:
            self._parts["shared"] += _tp.perf_counter() - _sh0

        if limit > 0:
            for i in cached_hits:
                e = self._expert_cache.get(i)
                if e is None:
                    continue
                # sel_of unpacks where() into (rows, cols): only the token rows index y
                # (y[rows, cols] would touch six columns of the dim axis).
                rows, cols = sel_of(i)
                if rows.dim() == 0:
                    # a 1-D routing table yields scalar positions; keep a row axis so the
                    # weights leave with shape [rows, dim]
                    rows = rows.unsqueeze(0)
                    cols = cols.unsqueeze(0)
                if e.fp4 is not None:
                    if _aq is not None:
                        y[rows] += e.forward_fp4_pq(_aq[rows], _asc[rows],
                                                    weights[rows, cols, None]).float()
                    else:
                        y[rows] += e.forward_fp4(xf[rows], weights[rows, cols, None]).float()
                else:
                    y[rows] += e.forward(xf[rows], weights[rows, cols, None])
                served.append(i)
        to_fetch = [i for i in active if i not in served]

        # 2) fetch the rest: the read was already issued above, so this only waits for
        # it. Routing is only known after the gate, so this is the one overlap available
        # without predicting the next token (prediction measured 14% hit rate).
        if self._expert_store is not None:
            if to_fetch:
                store = self._expert_store

                def serve(ids_batch, recs):
                    nonlocal y
                    # the DMA for these slots was started inside fetch(); make the
                    # compute stream wait for it here, immediately before the tiles are
                    # read, so the routing sync of the next layer is not blocked by it
                    import torch as _tt
                    for _rec in recs:
                        if getattr(_rec, "event", None) is not None:
                            _tt.cuda.current_stream().wait_event(_rec.event)
                            _rec.event = None
                    rec_by_id = {rec.key[1]: rec for rec in recs}
                    if _fused:
                        from . import moe_gemv

                        _ids = [i for i in ids_batch if i in rec_by_id]
                        _exps = []
                        for i in _ids:
                            _e = self._expert_cache.get(i)
                            if _e is None or _e.fp4 is None:
                                _e = self._expert_from_rec(rec_by_id[i])
                                if limit > 0:
                                    if len(self._expert_cache) >= limit:
                                        self._expert_cache.pop(next(iter(self._expert_cache)))
                                    self._expert_cache[i] = _e
                            _exps.append(_e)
                        if _exps:
                            _fp = [e.fp4 for e in _exps]
                            _wv = _tt.stack([_wsum[i].reshape(()) for i in _ids]).float()
                            _gu = moe_gemv.gate_up(_aq, _asc,
                                                   [f[0] for f in _fp], [f[1] for f in _fp],
                                                   [f[4] for f in _fp], [f[5] for f in _fp])
                            _aq2, _asc2 = moe_gemv.act_quant(_gu, _wv,
                                                            float(self.swiglu_limit))
                            _dn = moe_gemv.down(_aq2, _asc2,
                                                [f[2] for f in _fp], [f[3] for f in _fp])
                            moe_gemv.acc(_dn, y, len(_ids))
                            served.extend(_ids)
                            if (_os0.environ.get("TI_FUSED_CHECK") == "1"
                                    and not getattr(self, "_fused_checked", False)):
                                self._fused_checked = True
                                import sys as _sys

                                from .gemv_fp4 import gemv_fp4_pq

                                _e0 = _exps[0]
                                _f0 = _e0.fp4
                                _gr = gemv_fp4_pq(_aq, _asc, _f0[0], _f0[1])[0]
                                _ur = gemv_fp4_pq(_aq, _asc, _f0[4], _f0[5])[0]
                                _dn_ref = _tt.stack([
                                    _exps[_k].forward_fp4_pq(_aq, _asc, _wsum[_ids[_k]])
                                    .reshape(-1).float() for _k in range(len(_ids))])
                                _n = self.cfg.moe_inter_dim
                                print(f"[check] L{self.layer_id} shapes w={tuple(_f0[0].shape)} "
                                      f"s={tuple(_f0[1].shape)} w2={tuple(_f0[2].shape)} "
                                      f"s2={tuple(_f0[3].shape)} n_exp={len(_ids)} "
                                      f"aq={tuple(_aq.shape)} asc={tuple(_asc.shape)}",
                                      file=_sys.stderr, flush=True)
                                print(f"[check] gate {(_gu[0, :_n].float()-_gr.float()).abs().max().item():.3e} "
                                      f"up {(_gu[0, _n:].float()-_ur.float()).abs().max().item():.3e} "
                                      f"|g| {_gr.float().abs().max().item():.3e} "
                                      f"accdown {(_dn[:, :self.cfg.dim].float()-_dn_ref).abs().max().item():.3e} "
                                      f"|dref| {_dn_ref.abs().max().item():.3e} "
                                      f"w={_wsum[_ids[0]].reshape(()).item():.4f} "
                                      f"y={y.abs().max().item():.3e}",
                                      file=_sys.stderr, flush=True)
                        return
                    # Accumulate the batch and scatter once: six separate y[idx] += ops
                    # are six read-modify-write passes over the same rows, and at T=1
                    # each is a launch that cannot overlap the expert GEMVs.
                    _idx_acc: list = []
                    _val_acc: list = []
                    for i in ids_batch:
                        idx, top = sel_of(i)
                        rec = rec_by_id[i]
                        if use_fp4_kernel:
                            # one async DMA per expert (see _expert_from_rec). This used to
                            # be six blocking rec.tensor(...).to(target) copies per expert =
                            # 480 synchronous pageable copies per decode step, which was the
                            # single largest host cost in the step.
                            e = self._expert_from_rec(rec)
                            if _single and i in _wsum:
                                # one row, one expert: accumulate straight into y instead
                                # of building row/col index tensors and scattering
                                _w = _wsum[i]
                                if _aq is not None:
                                    _val = e.forward_fp4_pq(_aq, _asc, _w).float()
                                else:
                                    _val = e.forward_fp4(xf, _w).float()
                                y += _val
                                served.append(i)
                                continue
                            if _aq is not None:
                                # idx holds *every* row routed to this expert. Taking
                                # idx[0] (as this did) fed only the first row through the
                                # kernel, which is why a batch and the same rows one at a
                                # time disagreed while a single token was exact.
                                _val = e.forward_fp4_pq(_aq[idx], _asc[idx],
                                                        weights[idx, top, None]).float()
                            else:
                                _val = e.forward_fp4(xf[idx], weights[idx, top, None]).float()
                        else:
                            e = self._make_expert(rec)
                            if e.fp4 is not None:
                                _val = e.forward_fp4(xf[idx], weights[idx, top, None]).float()
                            else:
                                _val = e.forward(xf[idx], weights[idx, top, None]).float()
                        _idx_acc.append(idx.reshape(-1))
                        _val_acc.append(_val.reshape(-1, y.size(-1)))

                        if limit > 0:
                            if len(self._expert_cache) >= limit:
                                self._expert_cache.pop(next(iter(self._expert_cache)))
                            self._expert_cache[i] = e
                    if _val_acc:
                        if getattr(self, "_batched_scatter", True):
                            y.index_add_(0, torch.cat(_idx_acc), torch.cat(_val_acc).to(y.dtype))
                        else:
                            for _ix, _vx in zip(_idx_acc, _val_acc):
                                y[_ix] += _vx.to(y.dtype)

                # One async read per expert, consumed in two batches of three: the GPU
                # works on the first three while the last three are still on disk.
                # (An IOCP batch with QD=12 was tried too: 7.3GB/s standalone vs 6.3 for
                # the thread pool, but two extra copies per blob made it 2.3x slower
                # in-engine -- see TI_IOCP in loading.ExpertStore.)
                import time as _t
                # One batched read for the whole layer. A single-expert read runs at
                # ~0.7GB/s (QD=1 each, measured 27.6ms for one 18.8MB blob); batching the
                # six through the pinned reader reaches ~7.4GB/s, which is the difference
                # between 44ms and 16ms per layer. The six GEMVs only need ~0.3ms of GPU
                # time, so splitting the read to overlap compute buys nothing.
                # prefill streams every expert once per layer: admitting those into the
                # RAM cache evicts the decode working set and, worse, competes with the
                # pinned layer slab for host memory (see Engine.forward)
                # reuse the read issued before the cached-expert pass when the fetch set
                # is unchanged; otherwise issue it now
                if _pre_fetch is not None and set(to_fetch) == set(_prefetch_ids):
                    fut = _pre_fetch
                else:
                    fut = store.fetch_async(self.layer_id, list(to_fetch), cache=_admit_cache)
                keys = list(to_fetch)
                _w0 = _t.perf_counter()
                recs_all = fut.result()
                self._disk_wait_ms = getattr(self, "_disk_wait_ms", 0.0) + (_t.perf_counter() - _w0) * 1e3
                if _prof:
                    self._parts["fetch"] += _t.perf_counter() - _w0
                by_id = {rec.key[1]: rec for rec in recs_all}
                try:
                    _s0 = _tp.perf_counter() if _prof else 0.0
                    _bs = max(1, int(__import__("os").environ.get("TI_SERVE_BATCH", "3")))
                    if _fused:
                        # the fused kernels are indexed by expert, so they need the whole
                        # routed set in one call rather than three-expert slices
                        _bs = max(1, len(keys))
                    for _i in range(0, len(keys), _bs):
                        batch = keys[_i:_i + _bs]
                        if not batch:
                            continue
                        serve(batch, [by_id[i] for i in batch if i in by_id])
                    if _prof:
                        self._parts["serve"] += _tp.perf_counter() - _s0
                finally:
                    store.release(recs_all)
        else:
            for i in active:
                idx, top = sel_of(i)
                e = self._expert(i)
                if limit > 0 and len(self._expert_cache) > limit:
                    self._expert_cache.pop(next(iter(self._expert_cache)))
                if e.fp4 is not None:
                    y[idx] += e.forward_fp4(xf[idx], weights[idx, top, None]).float()
                else:
                    y[idx] += e.forward(xf[idx], weights[idx, top, None])

        # the shared expert was already accumulated above (rank 0 only, so the all-reduce
        # lands it once instead of once per rank)
        if _world > 1:
            from .tp import all_reduce_sum

            y = all_reduce_sum(y)
        return y.to(x.dtype).view(shape)
