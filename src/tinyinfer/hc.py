"""Hyper-Connections: Sinkhorn-normalized mix coefficients, collapse/expand of hc copies."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .nn import rms_norm


# Fused-kernel availability is a process-wide fact. Latching it per Block meant 40
# instances each paid the failing compile/raise (~18ms) twice per token = 1.5s/token.
_FUSED_PRE_OK = True
_FUSED_POST_OK = True
_FUSED_MIXES_OK = True


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

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        """x: [b,s,hc,d] -> pre/post/comb coefficients (fused kernel on CUDA)."""
        global _FUSED_MIXES_OK
        if x.is_cuda and _FUSED_MIXES_OK:
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
                _FUSED_MIXES_OK = False
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.hc_eps)
        mixes = F.linear(xf, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        global _FUSED_PRE_OK
        if x.is_cuda and _FUSED_PRE_OK:
            try:
                from .hc_fused import fused_hc_pre
                return fused_hc_pre(x, pre_mix)
            except Exception:
                _FUSED_PRE_OK = False
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """post * x + comb @ residual, HC-space residual update.

        Measured on this box: the torch path is 0.06ms for a 164KB problem. The fused
        kernel raises "RES shape[0] expected 4, got 1" for single-token steps, so its
        availability is latched once per process rather than re-raised per call.
        """
        global _FUSED_POST_OK
        import os as _os
        if x.is_cuda and _FUSED_POST_OK and _os.environ.get("TI_HC_NOFUSE") != "1":
            try:
                from .hc_fused import fused_hc_post
                if _os.environ.get("TI_HC_T") == "1":
                    import time as _t
                    _t0 = _t.perf_counter()
                    y = fused_hc_post(x, residual, post, comb)
                    self._hc_f = getattr(self, "_hc_f", [0.0, 0])
                    self._hc_f[0] += (_t.perf_counter() - _t0) * 1e3
                    self._hc_f[1] += 1
                    if self._hc_f[1] == 8 and getattr(self, "layer_id", -1) == 5:
                        print(f"[hc_post fused] mean={self._hc_f[0]/self._hc_f[1]:.2f}ms", flush=True)
                    return y
                return fused_hc_post(x, residual, post, comb)
            except Exception as _e:
                _FUSED_POST_OK = False
                import os as _os
                if _os.environ.get("TI_HC_DEBUG"):
                    print(f"[hc_post] fused kernel disabled ({_e!r})", flush=True)
        import os as _os
        if _os.environ.get("TI_HC_T") == "1":
            import time as _t
            _t0 = _t.perf_counter()
            y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
            _dt = (_t.perf_counter() - _t0) * 1e3
            self._hc_t = getattr(self, "_hc_t", [0.0, 0])
            self._hc_t[0] += _dt
            self._hc_t[1] += 1
            if self._hc_t[1] == 8 and getattr(self, "layer_id", -1) in (5, 20):
                print(f"[hc_post L{self.layer_id}] contig x={x.is_contiguous()} res={residual.is_contiguous()} "
                      f"post={post.is_contiguous()} dtype x={x.dtype} post={post.dtype} "
                      f"x={tuple(x.shape)} res={tuple(residual.shape)} "
                      f"mean={self._hc_t[0]/self._hc_t[1]:.2f}ms", flush=True)
            return y.type_as(x)
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)
