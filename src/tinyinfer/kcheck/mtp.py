"""MTP / speculative decode: does a draft-verify round produce the same tokens?

The 8GB card cannot hold the three draft stages beside the full model, so this path has
never run here: the planner turns speculation off below 16GB and forcing it on with the
full model hits a CUDA allocator assert. Running the engine with only two layers leaves
room, which is enough to check the thing that matters for correctness -- that a
speculative round returns the SAME token sequence as plain decoding. The acceptance rate
is meaningless at two layers (the draft sees fragments of a context) and is not asserted.
"""
from __future__ import annotations

import os

from . import Case


def _speculative(report):
    try:
        import torch
        from transformers import AutoTokenizer

        from ..engine import Engine
        from ..tiers import OptLevel
    except Exception as exc:
        report.add(Case('mtp', f'unavailable: {exc!r}', False))
        return
    if not torch.cuda.is_available():
        report.add(Case('mtp', 'skipped (no CUDA device)', True))
        return
    model = os.environ.get('TI_MODEL')
    if not model or not os.path.isdir(model):
        report.add(Case('mtp', 'skipped (set TI_MODEL)', True))
        return

    prev = os.environ.get('TI_MTP')
    os.environ['TI_MTP'] = '1'
    try:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        prompt = tok('The capital of', return_tensors='pt').input_ids[0].tolist()
        # two layers, not forty: the draft stages need ~7.6GB and will not fit beside a
        # full model on this card
        eng = Engine(model, tier=OptLevel.AGGRESSIVE, device='cuda', layers=[0, 1])
        if not getattr(eng, 'mtp', None):
            report.add(Case('mtp', 'draft stages did not load', False))
            return
        with torch.inference_mode():
            eng.reset()
            spec = eng.speculative_generate(prompt, max_new_tokens=6, temperature=0.0)
            eng.reset()
            base = list(eng.stream_generate(prompt, max_new_tokens=6, temperature=0.0))
        # speculative_generate returns (tokens, accept_rate)
        toks = spec[0] if isinstance(spec, tuple) else spec
        n = min(len(toks), len(base))
        same = sum(1 for i in range(n) if toks[i] == base[i])
        report.add(Case('mtp speculative matches plain decoding',
                        f'{same}/{n} tokens identical',
                        n > 0 and same == n))
        if isinstance(spec, tuple) and len(spec) > 1:
            rate = float(spec[1])
            report.add(Case('mtp reports an accept rate in [0,1]',
                            f'rate={rate}', 0.0 <= rate <= 1.0))
    except Exception as exc:
        report.add(Case('mtp', f'raised {exc!r}', False))
    finally:
        if prev is None:
            os.environ.pop('TI_MTP', None)
        else:
            os.environ['TI_MTP'] = prev


def register(report) -> None:
    _speculative(report)
