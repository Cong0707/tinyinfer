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
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


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
                 max_bf16_experts: int = 8, topk: int = 0):
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
            QuantLinear(store.get(p + "shared_experts.w1.weight"), store.get(p + "shared_experts.w1.scale")),
            QuantLinear(store.get(p + "shared_experts.w2.weight"), store.get(p + "shared_experts.w2.scale")),
            QuantLinear(store.get(p + "shared_experts.w3.weight"), store.get(p + "shared_experts.w3.scale")),
            cfg.swiglu_limit,
        )
        self._store = store
        self._expert_cache: dict[int, Expert] = {}
        self._last_ids: list[int] | None = None
        self.swiglu_limit = cfg.swiglu_limit

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

    def _make_expert(self, rec) -> Expert:
        """Materialise one expert for the gate's device: fp4 stays packed on CUDA.

        Dequantising on the host costs 74ms per expert (measured), which would dominate
        any CPU-side path, so the packed fp4 tensors go to the device and the fp4 kernel
        consumes them directly.
        """
        return self._expert_from(rec.tensor)

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
        shape = x.shape
        xf = x.view(-1, self.cfg.dim)
        weights, indices = self.gate.forward(xf, None if image_mask is None else image_mask.flatten())
        y = torch.zeros_like(xf, dtype=torch.float32)
        import os as _os
        if _os.environ.get("TI_NO_SYNC") == "1":
            # experiment only: routing needs a host sync (bincount). Skipping it shows how
            # much of the step is the CPU waiting for the GPU to finish this layer.
            active = list(range(min(6, self.cfg.n_routed_experts)))
        else:
            counts = torch.bincount(indices.flatten(), minlength=self.cfg.n_routed_experts).tolist()
            active = [i for i, cnt in enumerate(counts) if cnt > 0]
        self._last_ids = list(active)  # already the unique routed set (one host sync total)
        target = xf.device
        use_fp4_kernel = target.type == "cuda"
        limit = self._max_bf16

        # per-expert (token, slot) selection computed once and reused by both passes
        sels: dict[int, tuple] = {}

        def sel_of(i: int):
            s = sels.get(i)
            if s is None:
                s = sels[i] = torch.where(indices == i)
            return s

        # 1) serve cached experts (bf16 or fp4) without touching the stores
        served = []
        if limit > 0:
            for i in active:
                e = self._expert_cache.get(i)
                if e is None:
                    continue
                idx, top = sel_of(i)
                if e.fp4 is not None:
                    y[idx] += e.forward_fp4(xf[idx], weights[idx, top, None]).float()
                else:
                    y[idx] += e.forward(xf[idx], weights[idx, top, None])
                served.append(i)
        to_fetch = [i for i in active if i not in served]

        # 2) fetch the rest in two halves: the second half starts its disk read while
        # the GPU is still busy with the first. Routing is only known after the gate,
        # so this intra-layer split is the one overlap that exists without predicting
        # the next token (prediction measured 14% hit rate and lost time).
        if self._expert_store is not None:
            if to_fetch:
                store = self._expert_store

                def serve(ids_batch, recs):
                    rec_by_id = {rec.key[1]: rec for rec in recs}
                    for i in ids_batch:
                        idx, top = sel_of(i)
                        rec = rec_by_id[i]
                        if use_fp4_kernel:
                            e = Expert(None, None, None, self.swiglu_limit)
                            e.fp4 = (
                                rec.tensor("w1.weight").to(target), rec.tensor("w1.scale").to(target),
                                rec.tensor("w2.weight").to(target), rec.tensor("w2.scale").to(target),
                                rec.tensor("w3.weight").to(target), rec.tensor("w3.scale").to(target),
                            )
                            y[idx] += e.forward_fp4(xf[idx], weights[idx, top, None]).float()
                        else:
                            e = self._make_expert(rec)
                            if e.fp4 is not None:
                                y[idx] += e.forward_fp4(xf[idx], weights[idx, top, None]).float()
                            else:
                                y[idx] += e.forward(xf[idx], weights[idx, top, None])
                        if limit > 0:
                            if len(self._expert_cache) >= limit:
                                self._expert_cache.pop(next(iter(self._expert_cache)))
                            self._expert_cache[i] = e

                # One async read per expert, consumed in two batches of three: the GPU
                # works on the first three while the last three are still on disk.
                # (An IOCP batch with QD=12 was tried too: 7.3GB/s standalone vs 6.3 for
                # the thread pool, but two extra copies per blob made it 2.3x slower
                # in-engine -- see TI_IOCP in loading.ExpertStore.)
                futs = {i: store.fetch_async(self.layer_id, [i]) for i in to_fetch}
                keys = list(to_fetch)
                import time as _t
                # two batches of three: one GEMM per batch keeps kernel launch overhead
                # down, and the GPU works on the first three while the last three land.
                # (Six single-expert batches were measured too: 2.70s/token vs 1.54s --
                # the extra launches cost more than the finer overlap gains.)
                for batch in (keys[:3], keys[3:]):
                    if not batch:
                        continue
                    _w0 = _t.perf_counter()
                    recs = [futs[i].result()[0] for i in batch]
                    self._disk_wait_ms = getattr(self, "_disk_wait_ms", 0.0) + (_t.perf_counter() - _w0) * 1e3
                    try:
                        serve(batch, recs)
                    finally:
                        store.release(recs)
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

        y += self.shared.forward(xf)
        return y.to(x.dtype).view(shape)
