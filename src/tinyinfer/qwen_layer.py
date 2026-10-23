"""Whole-layer GEMV execution for Qwen3.5-0.8B decode.

The layer's weights are packed once into one contiguous bf16 block, in execution order,
and every projection of the layer runs from a single kernel launch (grid.y / prefix-sum
selects the projection). Measured on layer 0: torch's per-module linears 154-172us vs one
launch over the packed block 125us, and each individual projection called separately on
the packed block is *slower* (139 GB/s) than torch -- the win only exists when the whole
layer is one stream.
"""
from __future__ import annotations

import torch

from . import native


class Layer:
    def __init__(self, parts: dict, ks: dict, ns: dict, order: list):
        self.order = order
        self.ns = ns
        self.ks = ks
        offs, o = {}, 0
        for name in order:
            offs[name] = o
            o += ns[name] * ks[name]
        self.offsets = offs
        self.buf = torch.empty(o, dtype=torch.bfloat16, device="cuda")
        for name in order:
            t = parts[name]
            self.buf[offs[name]:offs[name] + t.numel()].copy_(t.reshape(-1))
        # descriptors {flat_offset, rows, K, out_offset} + prefix sums of block counts
        BROWS = 8
        desc, bstarts, acc = [], [0], 0
        for name in order:
            desc += [offs[name], ns[name], ks[name], acc]
            acc += ns[name]
            bstarts.append(bstarts[-1] + (ns[name] + BROWS - 1) // BROWS)
        self.out_size = acc
        self.desc = torch.tensor(desc, dtype=torch.int32, device="cuda")
        self.bstarts = torch.tensor(bstarts, dtype=torch.int32, device="cuda")
        self.grid = (bstarts[-1],)
        self.ptr = torch.zeros(len(order), dtype=torch.int64, device="cuda")
        self.out = torch.empty(acc, dtype=torch.float32, device="cuda")
        self.nbytes = o * 2

    def set_inputs(self, xs: dict):
        """xs: name -> 1-D bf16 tensor (its data must stay alive while the launch runs)"""
        for i, name in enumerate(self.order):
            self.ptr[i] = xs[name].data_ptr()

    def run(self):
        native.launch(native.kernel("qwen_fused", "qwen_gemv_layer_flat"), self.grid,
                      (256,), [self.buf, self.desc, self.bstarts, self.ptr, self.out,
                               len(self.order), 0])

    def get(self, name: str) -> torch.Tensor:
        o = self.offsets_name(name)
        return self.out[o:o + self.ns[name]]

    def offsets_name(self, name: str) -> int:
        a = 0
        for n in self.order:
            if n == name:
                return a
            a += self.ns[n]
        raise KeyError(name)


def available() -> bool:
    try:
        native.kernel("qwen_fused", "qwen_gemv_layer_flat")
        return True
    except Exception:
        return False
