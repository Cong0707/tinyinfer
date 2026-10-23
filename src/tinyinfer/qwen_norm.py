"""Point Qwen3.5's two RMSNorm variants at the hand-written kernels.

Each call is seven torch kernels (to fp32, pow, mean, rsqrt, mul, mul, to dtype) and both
run on every one of 24 layers, plus the gated one inside all 18 linear-attention layers.
Capturing the decode step removes the launch overhead but not the ~2.5us gap between
adjacent kernels, and with 1188 kernels per token that gap alone is most of the difference
between the measured 7.0ms and the 4.08ms the weights imply at this card's bandwidth.

Both replacements reproduce the reference arithmetic exactly, including where the
reference rounds to the activation dtype (RMSNormGated rounds the normalised value before
scaling by the weight).
"""
from __future__ import annotations

import torch


def install() -> bool:
    from . import native

    try:
        native.kernel("qwen_norm", "qwen_rms_norm_bf")
    except Exception:
        return False
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    def rms_forward(self, x):
        if x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == self.weight.numel():
            shape = x.shape
            d = shape[-1]
            rows = 1
            for v in shape[:-1]:
                rows *= int(v)
            xc = x.reshape(rows, d).contiguous()
            y = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
            native.launch(native.kernel("qwen_norm", "qwen_rms_norm_bf"), (rows,), (32,),
                          [xc, self.weight.detach().float().contiguous(), y, rows, d,
                           float(self.eps), 1])   # Qwen uses (1 + weight)
            return y.view(*shape)
        out = x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)

    def gated_forward(self, hidden_states, gate):
        if (hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
                and gate.shape == hidden_states.shape
                and hidden_states.shape[-1] == self.weight.numel()):
            shape = hidden_states.shape
            d = shape[-1]
            rows = 1
            for v in shape[:-1]:
                rows *= int(v)
            xc = hidden_states.reshape(rows, d).contiguous()
            gc = gate.reshape(rows, d).contiguous()
            y = torch.empty((rows, d), dtype=torch.bfloat16, device=hidden_states.device)
            native.launch(native.kernel("qwen_norm", "qwen_rms_norm_gated_bf"), (rows,),
                          (32,),
                          [xc, gc, self.weight.detach().float().contiguous(), y, rows, d,
                           float(self.variance_epsilon)])
            return y.view(*shape)
        dt = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.variance_epsilon)
        h = self.weight * h.to(dt)
        h = h * torch.nn.functional.silu(gate.to(torch.float32))
        return h.to(dt)

    if not getattr(M.Qwen3_5RMSNorm, "_ti_patched", False):
        M.Qwen3_5RMSNorm.forward = rms_forward
        M.Qwen3_5RMSNorm._ti_patched = True
    if not getattr(M.Qwen3_5RMSNormGated, "_ti_patched", False):
        M.Qwen3_5RMSNormGated.forward = gated_forward
        M.Qwen3_5RMSNormGated._ti_patched = True
    return True
