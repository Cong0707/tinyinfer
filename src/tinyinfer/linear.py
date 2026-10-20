"""Quantized linear layers: keep HF fp8 weights resident and compute with kernels.

Two backends:
- CUDA + tilelang available: official fp8_gemm on the fp8 weights (no bf16 copy; block
  scale 32x32 matches the HF layout).
- otherwise: dequant to bf16 once (reference path).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import gemm
from .quant import dequant_fp8_block


class QuantLinear:
    """x @ w_fp8.T with the HF block scales; optionally keeps the bf16 copy around."""

    def __init__(self, w_fp8: torch.Tensor, scale: torch.Tensor, *, cache_bf16: bool = False, device=None):
        if device is not None:
            w_fp8 = w_fp8.to(device)
            scale = scale.to(device)
        self.weight = w_fp8
        self.scale = scale
        self._bf16 = None
        if cache_bf16:
            self._bf16 = dequant_fp8_block(w_fp8, scale)

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    def to(self, device):
        self.weight = self.weight.to(device)
        self.scale = self.scale.to(device)
        if self._bf16 is not None:
            self._bf16 = self._bf16.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._bf16 is not None:
            return F.linear(x, self._bf16.to(x.dtype) if self._bf16.dtype != x.dtype else self._bf16)
        if x.device.type == "cuda" and gemm.available():
            shape = x.shape
            y = gemm.fp8_linear(x.reshape(-1, shape[-1]), self.weight, self.scale)
            return y.view(*shape[:-1], self.out_features)
        if self.weight.device != x.device:
            self.to(x.device)
        w = dequant_fp8_block(self.weight, self.scale)
        if x.dtype != w.dtype:
            x = x.to(w.dtype)
        return F.linear(x, w)
