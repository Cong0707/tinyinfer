"""Hyper-Connections: Sinkhorn-normalized mix coefficients, collapse/expand of hc copies."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .nn import rms_norm


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch port of kernel.hc_split_sinkhorn (pre / post / doubly-stochastic comb)."""
    b, s, _ = mixes.shape
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
    comb = mixes[..., 2 * hc_mult:].view(b, s, hc_mult, hc_mult) * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult)

    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class HCBlockMixin:
    """The coefficient machinery every Block uses (official Block.hc_mixes/hc_pre/hc_post)."""

    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    hc_attn_fn: torch.Tensor
    hc_attn_scale: torch.Tensor
    hc_attn_base: torch.Tensor
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor

    def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [b,s,hc,d] -> pre/post/comb coefficients (fused kernel on CUDA)."""
        if x.is_cuda:
            try:
                from .hc_fused import fused_hc_mixes, fused_sinkhorn
                b, s = x.shape[0], x.shape[1]
                x2 = x.reshape(b * s, -1).to(torch.bfloat16)
                raw = fused_hc_mixes(x2, hc_fn.float(), self.hc_eps)  # [b*s, 24] fp32
                pre, post, comb = fused_sinkhorn(raw, hc_scale.float(), hc_base.float(),
                                                 self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
                return (pre.view(b, s, self.hc_mult), post.view(b, s, self.hc_mult),
                        comb.view(b, s, self.hc_mult, self.hc_mult))
            except Exception:
                pass
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.hc_eps)
        mixes = F.linear(xf, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            try:
                from .hc_fused import fused_hc_pre
                return fused_hc_pre(x, pre_mix)
            except Exception:
                pass
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """post * x + comb @ residual, HC-space residual update.

        Measured on this box: the GPU path costs ~3.2ms per call (many tiny launches on
        a 164KB problem), while the host does the same in 0.2ms; with ~2 calls/layer the
        CPU path saves >0.2s/token on 40 layers. Set hc_post_on_cpu=False to force the
        device path instead.
        """
        if x.is_cuda and getattr(self, "hc_post_on_cpu", False) and x.numel() <= (1 << 22):
            return self._hc_post_cpu(x, residual, post, comb)
        if x.is_cuda:
            try:
                from .hc_fused import fused_hc_post
                return fused_hc_post(x, residual, post, comb)
            except Exception:
                pass
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    @staticmethod
    def _hc_post_cpu(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                     comb: torch.Tensor) -> torch.Tensor:
        xc, rc = x.cpu(), residual.cpu()
        y = post.cpu().unsqueeze(-1) * xc.unsqueeze(-2) + torch.sum(
            comb.cpu().unsqueeze(-1) * rc.unsqueeze(-2), dim=2)
        return y.type_as(xc).to(x.device, non_blocking=False)
