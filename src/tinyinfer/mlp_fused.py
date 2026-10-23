"""Fuse the MLP's gate/up projections into one GEMV.

gate_proj and up_proj are two [intermediate, hidden] matrices fed the same row, so at
decode they are two separate GEMVs over two halves of one logical weight (336MB per layer
on Qwen3.5-0.8B). Measured: four such projections as separate small GEMVs run at 245 GB/s,
while the same bytes joined into one contiguous read run at 700 GB/s -- the card needs a
deep stream of requests and a 2KB-row kernel cannot provide it.

Joining them changes nothing numerically: a matmul row is independent of the others, so
splitting vs concatenating the output dimension only picks a different kernel.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def install(model) -> int:
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    from . import gemv_bf16
    if not gemv_bf16.available():
        return 0
    n = 0
    for lay in getattr(getattr(model, "model", model), "layers", []):
        mlp = getattr(lay, "mlp", None)
        if mlp is None or getattr(mlp, "_ti_fused", False):
            continue
        g, u, d = (getattr(mlp, k, None)
                   for k in ("gate_proj", "up_proj", "down_proj"))
        if g is None or u is None:
            continue
        if g.bias is not None or u.bias is not None:
            continue
        mlp._ti_gate_up = torch.cat([g.weight.detach(), u.weight.detach()], 0).contiguous()
        mlp._ti_inter = g.weight.shape[0]
        mlp._ti_fused = True
        n += 1
    if n == 0:
        return 0
    if getattr(M.Qwen3_5MLP, "_ti_patched", False):
        return n
    orig = M.Qwen3_5MLP.forward

    def forward(self, x, **kw):
        w = getattr(self, "_ti_gate_up", None)
        if w is None or x.dim() != 3 or x.shape[1] != 1 or x.dtype != torch.bfloat16:
            return orig(self, x, **kw)
        inter2 = w.shape[0]
        obuf = getattr(self, "_ti_out", None)
        if obuf is None or obuf.numel() != inter2:
            obuf = self._ti_out = torch.empty(inter2, dtype=torch.float32, device=x.device)
        gemv_bf16.gemv(x[0, 0], w, obuf)
        inter = self._ti_inter
        gate = obuf[:inter].to(x.dtype).view(1, 1, inter)
        up = obuf[inter:].to(x.dtype).view(1, 1, inter)
        out = self.act_fn(gate) * up
        return self.down_proj(out)

    M.Qwen3_5MLP.forward = forward
    M.Qwen3_5MLP._ti_patched = True
    return n
