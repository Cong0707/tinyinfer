"""Hyper-Connection: the DeepSeek-specific mixing path.

hc.py has already had two silent correctness bugs (a reduction over the wrong axis, and a
broadcast that made post multiply only its own residual). Both were found end to end, not
by a test. These cases pin the arithmetic down so a third one cannot hide: each fast path
is compared against the plain-torch form written beside it in hc.py.
"""
from __future__ import annotations

import torch

from . import Case, check_close


def _hc_pre(report):
    try:
        import torch.nn.functional as F

        from ..deepseek import hc
    except Exception as exc:
        report.add(Case('hc_pre', f'unavailable: {exc!r}', False))
        return
    torch.manual_seed(0)
    b, s, hcm, d = 1, 4, 4, 128
    x = torch.randn(b, s, hcm, d, device='cuda', dtype=torch.bfloat16)
    pre_mix = torch.randn(b, s, hcm, device='cuda', dtype=torch.float32)
    w = torch.randn(d, device='cuda', dtype=torch.bfloat16)
    eps = 1e-6
    collapsed = hc._hc_pre_ref(x, pre_mix)
    ref = F.rms_norm(collapsed.float(), (d,), w.float(), eps)

    import os
    prev = os.environ.get('TI_FUSE')
    try:
        for fuse, label in (('1', 'fused'), ('0', 'two-step')):
            os.environ['TI_FUSE'] = fuse
            hc._FUSED_PRE_NORM = None
            got = hc.hc_pre_rms_norm(x, pre_mix, w, eps)
            check_close(report, f'hc_pre_rms_norm ({label})', got, ref, rtol=2e-2)
    finally:
        if prev is None:
            os.environ.pop('TI_FUSE', None)
        else:
            os.environ['TI_FUSE'] = prev
        hc._FUSED_PRE_NORM = None


def _hc_post(report):
    try:
        import types

        from ..deepseek import hc
    except Exception as exc:
        report.add(Case('hc_post', f'unavailable: {exc!r}', False))
        return
    torch.manual_seed(1)
    b, s, hcm, d = 1, 4, 4, 128
    x = torch.randn(b, s, d, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn(b, s, hcm, d, device='cuda', dtype=torch.bfloat16)
    post = torch.randn(b, s, hcm, device='cuda', dtype=torch.float32)
    comb = torch.randn(b, s, hcm, hcm, device='cuda', dtype=torch.float32)
    ref = hc._hc_post_ref(x, residual, post, comb)
    obj = types.SimpleNamespace(hc_mult=hcm, layer_id=0, _device='cuda')

    import os
    prev = os.environ.get('TI_HC_NOFUSE')
    try:
        for env, label in (({'TI_HC_NOFUSE': '1'}, 'native/torch'), ({}, 'default')):
            os.environ.pop('TI_HC_NOFUSE', None)
            os.environ.update(env)
            hc._FUSED_POST_OK = None
            got = hc.HCBlockMixin.hc_post(obj, x, residual, post, comb)
            check_close(report, f'hc_post ({label})', got, ref, rtol=2e-2)
    finally:
        if prev is None:
            os.environ.pop('TI_HC_NOFUSE', None)
        else:
            os.environ['TI_HC_NOFUSE'] = prev
        hc._FUSED_POST_OK = None


def _sinkhorn(report):
    try:
        from ..deepseek import hc
    except Exception as exc:
        report.add(Case('sinkhorn', f'unavailable: {exc!r}', False))
        return
    torch.manual_seed(2)
    b, s, hcm = 1, 4, 4
    mixes = torch.randn(b, s, hcm * hcm + 2 * hcm, device='cuda', dtype=torch.float32)
    # three scale entries: one for pre, one for post, one for the comb
    scale = torch.rand(3, device='cuda', dtype=torch.float32) * 0.5 + 0.5
    base = torch.randn(2 * hcm + hcm * hcm, device='cuda', dtype=torch.float32)
    eps, iters = 1e-6, 20
    ref = hc._sinkhorn_host(mixes, scale, base, hcm, iters, eps)
    for label, fn in (('native', hc._native_sinkhorn), ('eager', hc._sinkhorn_body)):
        try:
            got = fn(mixes, scale, base, hcm, iters, eps)
        except Exception as exc:
            report.add(Case(f'sinkhorn {label}', f'raised {exc!r}', False))
            continue
        if got is None:
            report.add(Case(f'sinkhorn {label}', 'returned None', False))
            continue
        for i, part in enumerate(('pre', 'post', 'comb')):
            check_close(report, f'sinkhorn {label} {part}', got[i].float(),
                        ref[i].float(), rtol=1e-3)
    # the comb must be doubly stochastic: a row or column sum drifting off 1 means the
    # normalisation loop lost a division somewhere
    comb = ref[2].float()
    row = (comb.sum(-1) - 1).abs().max().item()
    col = (comb.sum(-2) - 1).abs().max().item()
    report.add(Case('sinkhorn comb is doubly stochastic',
                    f'row dev {row:.2e} col dev {col:.2e}', row < 1e-5 and col < 1e-5))


def register(report) -> None:
    _hc_pre(report)
    _hc_post(report)
    _sinkhorn(report)
