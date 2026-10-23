"""Hand-written decode kernels for Qwen3.5's GatedDeltaNet (linear attention).

transformers falls back to a pure-torch reference for these when fla/causal-conv1d are
missing (and torch.compile is broken on this box, so even installed fla cannot be used).
The reference is exact but costs hundreds of launch per layer; these are one launch each
for the seq_len == 1 case, which is all a decode step needs.
"""
from __future__ import annotations

import torch

from . import native


def available() -> bool:
    try:
        native.kernel("gdn", "gdn_step_f")
        return True
    except Exception:
        return False


def step(q, k, v, g, beta, state, l2norm=True):
    """One gated-delta-rule step.

    q,k: [B,H,Dk]  v: [B,H,Dv]  g,beta: [B,H]  state: [B,H,Dk,Dv] fp32 (updated in place)
    Returns out [B,H,Dv] in v's dtype.
    """
    B, H, Dk = q.shape
    Dv = v.shape[-1]
    bfs = q.dtype == torch.bfloat16
    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc = g.float().contiguous()
    bc = beta.float().contiguous()
    if bfs:
        out = torch.empty((B, H, Dv), dtype=torch.bfloat16, device=v.device)
        # v2 keeps the same arithmetic but gives each thread one dv column and walks dk in
        # a single pass, so the [Dk, Dv] state is touched as whole 128-thread rows instead
        # of 512-byte strided columns: 41.8us -> 32.3us for one layer's 2MB.
        fn = native.kernel("gdn", "gdn_step_v2")
    else:
        out = torch.empty((B, H, Dv), dtype=torch.float32, device=v.device)
        fn = native.kernel("gdn", "gdn_step_f")
        qc, kc, vc = qc.float(), kc.float(), vc.float()
    native.launch(fn, (B * H,), (128,), [qc, kc, vc, gc, bc, state, out, B, H, Dk, Dv,
                                       1 if l2norm else 0])
    return out


def conv_step(x, w, bias, conv_state, silu=True):
    """Depthwise causal conv for one new token, updating conv_state in place.

    x: [B, C]  w: [C, K]  conv_state: [B, C, K]  -> out [B, C]
    """
    B, C = x.shape
    K = conv_state.shape[-1]
    wc = w.float().contiguous()
    bc = (bias.float().contiguous() if bias is not None
          else torch.zeros(1, device=x.device))
    bf_state = conv_state.dtype == torch.bfloat16
    out = torch.empty((B, C), dtype=conv_state.dtype, device=x.device)
    xc = x.contiguous()
    fn = native.kernel("gdn", "gdn_conv_step_bf" if bf_state else "gdn_conv_step_f")
    native.launch(fn, ((B * C + 255) // 256,), (256,),
                  [xc if bf_state else xc.float(), wc, bc, conv_state, out,
                   B, C, K, 1 if silu else 0, 0 if bias is None else 1])
    return out


def norm_gated(x, z, w, eps=1e-6):
    """Gated RMSNorm over the last axis (bf16 in/out, fp32 reduction)."""
    shape = x.shape
    D = shape[-1]
    R = 1
    for v in shape[:-1]:
        R *= int(v)
    xc = x.reshape(R, D).contiguous()
    zc = z.reshape(R, D).contiguous()
    wf = w.float().contiguous()
    y = torch.empty((R, D), dtype=torch.bfloat16, device=x.device)
    native.launch(native.kernel("gdn", "gdn_norm_gated_bf"), (R,), (32,),
                  [xc, zc, wf, y, R, D, float(eps)])
    return y.view(*shape)
