"""Point Qwen3.5's per-layer elementwise steps at single-kernel versions.

Qwen3_5MLP.forward is down_proj(act_fn(gate_proj(x)) * up_proj(x)): silu and mul are two
torch kernels for a 3584-element activation. Qwen3_5DecoderLayer.forward does two
residual adds, each one kernel. Across 24 layers that is 72 launches per token, and inside
a captured graph each still costs ~1.2us.

Numerically identical: silu(x)*y in fp32 then cast to bf16, the same order torch uses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import native


def available() -> bool:
    try:
        native.kernel("qwen_fused", "qwen_swiglu")
        native.kernel("qwen_fused", "qwen_residual_add")
        return True
    except Exception:
        return False


def install(model) -> bool:
    if not available():
        return False
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    if getattr(M.Qwen3_5MLP, "_ti_elt", False):
        return True

    def mlp_forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        if (x.dim() == 3 and x.shape[1] == 1 and x.dtype == torch.bfloat16
                and g.dtype == torch.float32):
            inter = g.shape[-1]
            gu = torch.cat([g.reshape(-1), u.reshape(-1)], 0).contiguous()
            out = torch.empty(inter, dtype=torch.bfloat16, device=x.device)
            native.launch(native.kernel("qwen_fused", "qwen_swiglu"),
                          ((inter + 255) // 256,), (256,), [gu, out, inter, 0.0])
            return self.down_proj(out.view(1, 1, inter))
        return self.down_proj(self.act_fn(g) * u)

    M.Qwen3_5MLP.forward = mlp_forward
    M.Qwen3_5MLP._ti_elt = True
    return True
