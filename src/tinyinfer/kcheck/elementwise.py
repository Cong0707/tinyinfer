"""Elementwise and reduction kernels used on the decode path.

These are small, but they run every layer / every token, and a wrong epsilon or a
missing scale silently shifts every activation downstream instead of crashing.
"""
from __future__ import annotations

import torch

from . import Case, check_close


def _rms_norm(report):
    try:
        from ..kernels import native
    except Exception as exc:
        report.add(Case('rms_norm', f'unavailable: {exc!r}', False))
        return
    try:
        native.kernel('rms_norm', 'rms_norm_kernel')
    except Exception as exc:
        report.add(Case('rms_norm', f'kernel not built: {exc!r}', False))
        return
    torch.manual_seed(20)
    for (rows, d) in ((1, 5120), (4, 5120), (8, 128)):
        x = torch.randn(rows, d, dtype=torch.bfloat16, device='cuda')
        w = torch.randn(d, dtype=torch.float32, device='cuda')
        eps = 1e-6
        out = torch.empty_like(x)
        native.launch(native.kernel('rms_norm', 'rms_norm_kernel'), (rows,), (256,),
                      [x, w, out, rows, d, eps])
        want = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w
        check_close(report, f'rms_norm [{rows},{d}]', out, want, rtol=2e-2)


def _moe_act(report):
    try:
        from ..kernels import native
    except Exception as exc:
        report.add(Case('moe_act', f'unavailable: {exc!r}', False))
        return
    try:
        native.kernel('moe_act', 'moe_act_kernel')
    except Exception as exc:
        report.add(Case('moe_act', f'kernel not built: {exc!r}', False))
        return
    import torch.nn.functional as F

    torch.manual_seed(21)
    n = 4096
    gate = torch.randn(n, dtype=torch.bfloat16, device='cuda') * 3
    up = torch.randn(n, dtype=torch.bfloat16, device='cuda') * 3
    out = torch.empty_like(gate)
    limit = 4.0
    scale = 2.5
    native.launch(native.kernel('moe_act', 'moe_act_kernel'), ((n + 255) // 256,), (256,),
                  [gate, up, out, n, limit, scale, 1])
    g = gate.float().clamp(max=limit)
    u = up.float().clamp(-limit, limit)
    want = (F.silu(g) * u) * scale
    check_close(report, 'moe_act silu*up*scale', out, want, rtol=2e-2)

    # has_scale=0 is the shared-expert path: no routing weight applied
    native.launch(native.kernel('moe_act', 'moe_act_kernel'), ((n + 255) // 256,), (256,),
                  [gate, up, out, n, limit, scale, 0])
    want0 = F.silu(g) * u
    check_close(report, 'moe_act without scale', out, want0, rtol=2e-2)


def _head_gemv(report):
    try:
        from ..kernels import native
    except Exception as exc:
        report.add(Case('head_gemv', f'unavailable: {exc!r}', False))
        return
    try:
        native.kernel('head_gemv', 'head_gemv')
    except Exception as exc:
        report.add(Case('head_gemv', f'kernel not built: {exc!r}', False))
        return
    # One warp per OUTPUT ROW of the head, over ONE activation vector: this is the lm-head
    # shape, out[n] = sum_k h[k] * w[n][k]. The h argument is the single hidden vector,
    # not a [rows, k] batch -- feeding it 4 rows makes every warp read h[0..k] and the
    # rows are not independent. The first version of this case did that and reported a
    # 2.2 relative error against a per-row reference.
    torch.manual_seed(22)
    for (n_rows, k) in ((1, 5120), (4, 5120), (1024, 256)):
        h = torch.randn(k, dtype=torch.float32, device='cuda')
        w = torch.randn(n_rows, k, dtype=torch.bfloat16, device='cuda')
        out = torch.empty(n_rows, dtype=torch.float32, device='cuda')
        warps = n_rows
        blocks = max(1, (warps * 32 + 255) // 256)
        native.launch(native.kernel('head_gemv', 'head_gemv'), (blocks,), (256,),
                      [h, w, out, n_rows, k])
        want = (w.float() @ h)
        check_close(report, f'head_gemv [{n_rows} rows, k={k}]', out, want, rtol=3e-2)


def register(report) -> None:
    _rms_norm(report)
    _moe_act(report)
    _head_gemv(report)
