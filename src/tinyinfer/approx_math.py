"""Approximate math for EXTREME-tier kernels: curves instead of expensive functions.

Motivation: on GPU, exp/sin/cos/rsqrt go through SFU with limited throughput and long
latency; a low-degree polynomial (Taylor/minimax) or a bit-trick curve fit can trade a
bounded relative error for a few FMAs. The heavy user in V4.1 is RoPE (sin/cos -- we
precompute the whole table on CPU already), softmax (exp), SwiGLU (silu), RMSNorm (rsqrt).

All functions here are drop-in, purely torch, and measured for error in tests.
"""
from __future__ import annotations

import torch

LOG2E = 1.4426950408889634


def fast_rsqrt(x: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """1/sqrt(x) via the classic bit-trick initial guess + Newton iterations (no SFU)."""
    xf = x.float()
    i = xf.view(torch.int32)
    y = (0x5F3759DF - (i >> 1)).view(torch.float32)
    for _ in range(max(1, iters)):
        y = y * (1.5 - 0.5 * xf * y * y)
    return y.to(x.dtype)


def fast_exp(x: torch.Tensor) -> torch.Tensor:
    """exp(x) as 2^(x*log2e) with a polynomial mantissa (max rel err ~5e-3)."""
    xf = x.float().clamp(-80.0, 80.0)
    z = xf * LOG2E
    zi = z.floor()
    f = z - zi  # fractional part in [0,1)
    p = 1.0 + f * (0.6931471805599453 + f * (0.2402265069591007 + f * 0.0555041086648216))
    out = torch.ldexp(p, zi.to(torch.int32))
    return out.to(x.dtype)


def fast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """1/(1+exp(-x)) with the approximate exp (max abs error ~2e-3)."""
    xf = x.float().clamp(-20.0, 20.0)
    e = fast_exp(-xf)
    return (1.0 / (1.0 + e)).to(x.dtype)


def fast_silu(x: torch.Tensor) -> torch.Tensor:
    return (x.float() * fast_sigmoid(x).float()).to(x.dtype)


def _reduce_quadrant(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x.float() + torch.pi, 2 * torch.pi) - torch.pi  # [-pi, pi]


def poly_sin(x: torch.Tensor) -> torch.Tensor:
    """sin(x): 7th-order Taylor on the reduced quadrant (max abs err ~1e-5)."""
    t = _reduce_quadrant(x)
    s = torch.where(t > torch.pi / 2, torch.pi - t,
                    torch.where(t < -torch.pi / 2, -torch.pi - t, t))
    t2 = s * s
    return (s * (1 - t2 * (1 / 6 - t2 * (1 / 120 - t2 * (1 / 5040 - t2 / 362880))))).to(x.dtype)


def poly_cos(x: torch.Tensor) -> torch.Tensor:
    """cos(x): 8th-order Taylor on the reduced quadrant (max abs err ~1e-5)."""
    t = _reduce_quadrant(x).abs()
    s = torch.where(t > torch.pi / 2, torch.pi - t, t)
    c = 1 - s * s * (1 / 2 - s * s * (1 / 24 - s * s * (1 / 720 - s * s / 40320)))
    sign = torch.where(t > torch.pi / 2, -1.0, 1.0)
    return (c * sign).to(x.dtype)
