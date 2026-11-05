"""Attention: prefill and stepwise decode must agree.

The engine runs a prompt through one batched forward and then one token at a time. Those
two paths share a KV cache but not a code path, so they can drift: a wrong causal mask in
the batched form, or a window boundary computed from the wrong position, gives a result
that looks fine (finite, right shape) and decodes into different tokens. The existing
test only asserts shape and finiteness; this one compares the numbers.
"""
from __future__ import annotations

import os

from . import Case, check_close


def _prefill_vs_stepwise(report):
    try:
        import pathlib

        import torch

        from ..config import TinyConfig
        from ..deepseek.attention import MLAWindowAttention
        from ..weights import ShardStore
    except Exception as exc:
        report.add(Case('attention', f'unavailable: {exc!r}', False))
        return
    if not torch.cuda.is_available():
        report.add(Case('attention', 'skipped (no CUDA device)', True))
        return
    model = os.environ.get('TI_MODEL')
    if not model or not os.path.isdir(model):
        report.add(Case('attention', 'skipped (set TI_MODEL)', True))
        return

    cfg = TinyConfig.from_hf(pathlib.Path(model))
    store = ShardStore(model)
    # a window layer (compress_ratio == 0) is the plain sliding-window path
    lid = next(i for i, r in enumerate(cfg.compress_ratios) if r == 0)
    torch.manual_seed(0)
    T = 6
    x = torch.randn(1, T, cfg.dim, dtype=torch.bfloat16, device='cuda') * 0.02

    attn = MLAWindowAttention(store, lid, cfg)
    # MLAWindowAttention has no to(): the engine's Block owns the device placement, so a
    # bare attention module keeps its weights on the host. Move each submodule here or
    # the first matmul mixes a cuda activation with a cpu weight.
    for _name, _v in list(vars(attn).items()):
        if _name.startswith('_'):
            continue
        if torch.is_tensor(_v):
            setattr(attn, _name, _v.to('cuda'))
        elif hasattr(_v, 'to'):
            try:
                _v.to('cuda')
            except Exception:
                pass
    attn.reset()
    batched = attn.forward(x, 0)

    attn.reset()
    steps = [attn.forward(x[:, t:t + 1], t) for t in range(T)]
    stepwise = torch.cat(steps, dim=1)

    report.add(Case('attention prefill shape', str(tuple(batched.shape)),
                    batched.shape == x.shape))
    check_close(report, 'attention prefill == stepwise decode',
                stepwise, batched, rtol=2e-2,
                note='(shares the KV cache, different code paths)')


def register(report) -> None:
    _prefill_vs_stepwise(report)
