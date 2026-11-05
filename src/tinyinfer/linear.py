"""
Quantized linear layers.

One class covers both a single card and a split across many, because the difference is
data, not control flow: a layer holds a *list* of partitions, each a (device, weight
slice, scale slice) on the card that owns it. One partition is the single-GPU case and
takes the same path a split does -- there is no `if multi_gpu:` anywhere below, only
`len(parts) == 1` skipping the concatenate.

Row-parallel by construction: the output dimension is what gets divided, so each card
computes a disjoint slice of y and nothing has to be reduced. The only traffic is the
activation going out and the slices coming back. That is what lets n cards read n x K
bytes in parallel, which is the whole point at decode: a token is bounded by reading the
weights once, and reading them across cards divides that bound by n.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import gemm
from .quant import dequant_fp8_block


class QuantLinear:
    """x @ W.T for a block-quantized W, optionally split across cards.

    parts[i] = (device, weight, scale) where weight holds rows [lo_i, hi_i) of the full
    W and the matching scale rows. The list is in output order, so forward() only has to
    concatenate. One part means one card and no copy in or out.
    """

    def __init__(self, weight, scale, *, cache_bf16: bool = False, device=None,
                 parts=None):
        if parts is not None:
            self.parts = list(parts)
            self.weight = self.parts[0][1]      # first slice: shape/identity for callers
            self.scale = self.parts[0][2]
            self._bf16 = None
            self._n_out = sum(int(w.shape[0]) for _, w, _s in self.parts)
            return
        if device is not None:
            weight = weight.to(device)
            scale = scale.to(device)
        self.parts = [(weight.device, weight, scale)]
        self.weight = weight
        self.scale = scale
        self._bf16 = None
        self._n_out = int(weight.shape[0])
        if cache_bf16:
            self._bf16 = dequant_fp8_block(weight, scale)

    # ---- shape ----
    @property
    def out_features(self) -> int:
        return self._n_out

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    @property
    def is_sharded(self) -> bool:
        return len(self.parts) > 1

    # ---- placement ----
    def to(self, device):
        # A sharded layer ignores this: every part is already pinned to the card the
        # plan chose, and moving it would defeat the plan.
        if self.is_sharded:
            return self
        self.weight = self.weight.to(device)
        self.scale = self.scale.to(device)
        self.parts = [(self.weight.device, self.weight, self.scale)]
        if self._bf16 is not None:
            self._bf16 = self._bf16.to(device)
        return self

    def _sync_parts_from_weight(self) -> None:
        """Re-point a single-part `parts` at the current .weight/.scale.

        Called by set_weight() (deepseek/model.py) when flat streaming swaps a layer's
        bytes. A sharded layer is left alone: there `parts` IS the multi-card placement,
        and rewriting it from one attribute would drop the other cards' slices.
        """
        if len(self.parts) == 1:
            p = self.parts[0]
            if p[1] is not self.weight or p[2] is not self.scale:
                self.parts = [(self.weight.device, self.weight, self.scale)]

    # ---- compute ----
    def _one(self, x: torch.Tensor, weight, scale) -> torch.Tensor:
        if self._bf16 is not None:
            # The bf16 copy is built on the host in __init__ and nothing moves it with the
            # layer, so migrate lazily, once -- the old code did this and the rewrite
            # dropped it, which surfaced as "mat2 is on cpu" inside the shared expert.
            w = self._bf16
            if w.device != x.device:
                w = w.to(x.device)
                self._bf16 = w
            return F.linear(x, w.to(x.dtype) if w.dtype != x.dtype else w)
        if x.device.type == 'cuda' and gemm.available():
            shape = x.shape
            y = gemm.fp8_linear(x.reshape(-1, shape[-1]), weight, scale)
            return y.view(*shape[:-1], int(weight.shape[0]))
        w = dequant_fp8_block(weight, scale)
        if x.dtype != w.dtype:
            x = x.to(w.dtype)
        return F.linear(x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_sharded or not x.is_cuda:
            # Read self.weight/self.scale, NOT parts[0]. Flat streaming swaps a layer's
            # weights by calling set_weight(), which assigns .weight (and .data on an
            # nn.Parameter) but cannot reach into the parts tuple. parts[0] therefore kept
            # pointing at the host copy the layer was built with, so this branch saw
            # weight.device != x.device on EVERY call and re-uploaded the matrix: measured
            # 632 real copies per decode token in the attention linears alone (7.45GB) and
            # another 490 in the MoE, 1.13s of a 1.5s step, all of it redundant. The
            # sharded path below still uses parts, which is what multi-card placement
            # owns.
            weight, scale = self.weight, self.scale
            if weight.device != x.device:
                return self._one(x, weight.to(x.device), scale.to(x.device))
            return self._one(x, weight, scale)
        xdev = x.device
        outs = []
        for _dev, weight, scale in self.parts:
            tgt = weight.device
            xs = x if x.device == tgt else x.to(tgt, non_blocking=True)
            ys = self._one(xs, weight, scale)
            if ys.device != xdev:
                ys = ys.to(xdev, non_blocking=True)
            outs.append(ys)
        return torch.cat(outs, dim=-1)

    __call__ = forward
