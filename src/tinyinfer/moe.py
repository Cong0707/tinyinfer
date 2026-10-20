"""MoE with fp4 routed experts and one fp8 shared expert, loaded straight from HF shards."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TinyConfig
from .quant import dequant_fp4_block, dequant_fp8_block
from .weights import ShardStore


class Gate:
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig):
        p = f"layers.{layer_id}.ffn.gate."
        self.weight = store.get(p + "weight").float()
        self.bias = store.get(p + "bias").float()
        self.bias_vl = store.get(p + "bias_vl").float() if cfg.vision_enabled else None
        self.topk = cfg.n_activated_experts
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

    def forward_fp4(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Official fp4 kernel path: fp8 activations x fp4 weights (no bf16 dequant copy)."""
        from .gemm import fp4_linear

        w1, s1, w2, s2, w3, s3 = self.fp4
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
        x = x.to(self.w1.dtype)
        gate = F.linear(x, self.w1).float()
        up = F.linear(x, self.w3).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        return F.linear(y.to(self.w2.dtype), self.w2)


class MoE:
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, expert_store=None,
                 max_bf16_experts: int = 0):
        self.cfg = cfg
        self.layer_id = layer_id
        self._expert_store = expert_store  # optional tinyinfer.loading.ExpertStore
        # 0 = never keep bf16 copies (production/GPU path); N>0 = bounded reference cache
        self._max_bf16 = int(max_bf16_experts)
        self.gate = Gate(store, layer_id, cfg)
        p = f"layers.{layer_id}.ffn."
        self.shared = Expert(
            dequant_fp8_block(store.get(p + "shared_experts.w1.weight"), store.get(p + "shared_experts.w1.scale")),
            dequant_fp8_block(store.get(p + "shared_experts.w2.weight"), store.get(p + "shared_experts.w2.scale")),
            dequant_fp8_block(store.get(p + "shared_experts.w3.weight"), store.get(p + "shared_experts.w3.scale")),
            cfg.swiglu_limit,
        )
        self._store = store
        self._expert_cache: dict[int, Expert] = {}
        self.swiglu_limit = cfg.swiglu_limit

    def _expert(self, i: int) -> Expert:
        e = self._expert_cache.get(i)
        if e is None:
            if self._expert_store is not None:
                # zero-copy views are only valid inside the context; dequant copies out
                with self._expert_store.experts(self.layer_id, [i]) as recs:
                    rec = recs[0]
                    w = {n: dequant_fp4_block(rec.tensor(n + ".weight"), rec.tensor(n + ".scale"))
                         for n in ("w1", "w2", "w3")}
                e = Expert(w["w1"], w["w2"], w["w3"], self.swiglu_limit)
            else:
                p = f"layers.{self.layer_id}.ffn.experts.{i}."
                e = Expert(
                    dequant_fp4_block(self._store.get(p + "w1.weight"), self._store.get(p + "w1.scale")),
                    dequant_fp4_block(self._store.get(p + "w2.weight"), self._store.get(p + "w2.scale")),
                    dequant_fp4_block(self._store.get(p + "w3.weight"), self._store.get(p + "w3.scale")),
                    self.swiglu_limit,
                )
            # match the device Block.to() moved the shared expert / gate to
            target = self.shared.w1.device
            e.w1 = e.w1.to(target)
            e.w2 = e.w2.to(target)
            e.w3 = e.w3.to(target)
            self._expert_cache[i] = e
        return e

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        xf = x.view(-1, self.cfg.dim).to(self.shared.w1.dtype if self.shared.w1 is not None else x.dtype)
        weights, indices = self.gate.forward(xf, None if image_mask is None else image_mask.flatten())
        y = torch.zeros_like(xf, dtype=torch.float32)
        n_experts = self.cfg.n_routed_experts
        counts = torch.bincount(indices.flatten(), minlength=n_experts)
        for i in range(n_experts):
            if counts[i].item() == 0:
                continue
            idx, top = torch.where(indices == i)
            if self._expert_store is not None:
                if self._max_bf16 > 0:
                    cached = self._expert_cache.get(i)
                    if cached is not None:
                        y[idx] += cached.forward(xf[idx].float(), weights[idx, top, None])
                        continue
                # production path: fetch fp4, dequant in place, drop the bf16 copy right away
                with self._expert_store.experts(self.layer_id, [i]) as recs:
                    rec = recs[0]
                    target = self.shared.w1.device
                    # fp4 kernels only pay off for larger M (official measurements: M=64+);
                    # decode-sized groups use GPU dequant into bf16 instead
                    use_fp4_kernel = target.type == "cuda" and xf.shape[0] >= 32
                    if use_fp4_kernel:
                        w1 = rec.tensor("w1.weight").to(target)
                        s1 = rec.tensor("w1.scale").to(target)
                        w2 = rec.tensor("w2.weight").to(target)
                        s2 = rec.tensor("w2.scale").to(target)
                        w3 = rec.tensor("w3.weight").to(target)
                        s3 = rec.tensor("w3.scale").to(target)
                        e = Expert(None, None, None, self.swiglu_limit)
                        e.fp4 = (w1, s1, w2, s2, w3, s3)
                    else:
                        def _dq(name):
                            w = rec.tensor(name + ".weight")
                            s = rec.tensor(name + ".scale")
                            return dequant_fp4_block(w, s)
                        e = Expert(_dq("w1"), _dq("w2"), _dq("w3"), self.swiglu_limit)
                    if use_fp4_kernel:
                        y[idx] += e.forward_fp4(xf[idx], weights[idx, top, None]).float()
                    else:
                        y[idx] += e.forward(xf[idx].float(), weights[idx, top, None])
                if self._max_bf16 > 0:
                    if len(self._expert_cache) >= self._max_bf16:
                        self._expert_cache.pop(next(iter(self._expert_cache)))
                    self._expert_cache[i] = e
            else:
                y[idx] += self._expert(i).forward(xf[idx].float(), weights[idx, top, None])
        y += self.shared.forward(xf.float())
        return y.to(x.dtype).view(shape)
