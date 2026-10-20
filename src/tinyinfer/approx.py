"""EXTREME-tier approximations (research): low-rank fitting and FFT token mixing.

These are lossy by construction; the point is to measure the quality/size tradeoff
on real weights instead of guessing. Nothing here is enabled by default.
"""
from __future__ import annotations

import torch


def low_rank_linear(weight: torch.Tensor, rank_ratio: float = 0.75) -> tuple[torch.Tensor, dict]:
    """SVD-truncate a linear weight; returns (approx_weight, stats).

    weight: [out, in]. The approximation keeps the top-r singular directions and is
    returned in the same shape so it can be dropped into the existing forward.
    """
    w = weight.float()
    out_f, in_f = w.shape
    r = max(1, int(min(out_f, in_f) * rank_ratio))
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    approx = (u[:, :r] * s[:r]) @ vh[:r]
    energy = float((s[:r] ** 2).sum() / (s ** 2).sum().clamp_min(1e-20))
    err = (w - approx).norm() / w.norm().clamp_min(1e-20)
    return approx.to(weight.dtype), {"rank": r, "kept_energy": energy, "rel_fro_err": float(err)}


def skip_layer_plan(n_layers: int, every: int = 8) -> list[int]:
    """EXTREME: names of layers whose residual contribution can be dropped (pattern)."""
    return [i for i in range(n_layers) if every > 0 and i > 0 and i % every == every - 1]


def fft_token_mix(x: torch.Tensor) -> torch.Tensor:
    """FNet-style token mixing: replace attention's token mixing with FFTs.

    x: [b, s, d] -> Re(mean over even/odd? ) Here the classic FNet block:
    y = real(ifft(fft(x, dim=1) * fft(x, dim=-1))) -- a parameter-free global mixing
    used only as an EXTREME-tier research baseline.
    """
    fx = torch.fft.fft(x.float(), dim=1)
    fy = torch.fft.fft(fx, dim=-1)
    z = torch.fft.ifft(fy, dim=-1).real
    return z.to(x.dtype)


def fft_attention_approx(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Research baseline: replace softmax attention with FFT convolution of q/k.

    Computes kv = fft-conv(k, v) then y = q @ kv (linear attention with a fixed kernel).
    It cannot match a pretrained softmax-attention model; measured, not assumed.
    """
    # [b, s, h, d] q ; [b, s, d] kv (MQA)
    b, s, h, d = q.shape
    zero_pad = torch.zeros(b, s, d, dtype=v.dtype, device=v.device)
    fk = torch.fft.fft(torch.cat([k, zero_pad], dim=1).float(), dim=1)
    fv = torch.fft.fft(torch.cat([v, zero_pad], dim=1).float(), dim=1)
    conv = torch.fft.ifft(fk * fv, dim=1).real[:, :s]
    return (q.float() @ conv.unsqueeze(2)).to(q.dtype) * scale
