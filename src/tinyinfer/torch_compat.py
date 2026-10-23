"""Restore symbols that moved between torch 2.x releases.

torch 2.9 relocated several functorch helpers, but torch._dynamo.backends.debugging still
expects them at their old paths. Anything that touches torch.compile -- including fla's
Triton kernels, which import through torch._dynamo -- fails until those names exist
again. This module puts them back; import it before torch.compile, fla, or any
transformers code that may use either.
"""
from __future__ import annotations

_MOVED = {
    "min_cut_rematerialization_partition": (
        "torch._functorch.partitioners",
        "torch._functorch._aot_autograd",
    ),
    "draw_graph": (
        "torch._functorch.compilers",
        "torch._functorch.aot_autograd",
        "functorch.compile",
    ),
    "create_aot_config": ("torch._functorch._aot_autograd",),
    "default_partition": ("torch._functorch.partitioners",),
}


def _resolve(name: str):
    for mod in _MOVED.get(name, ()):
        try:
            m = __import__(mod, fromlist=[name])
        except Exception:
            continue
        fn = getattr(m, name, None)
        if fn is not None:
            return fn
    return None


def install() -> None:
    targets = []
    for mod in ("functorch.compile", "torch._functorch"):
        try:
            targets.append(__import__(mod, fromlist=["_"]))
        except Exception:
            pass
    for name in _MOVED:
        fn = _resolve(name)
        if fn is None:
            continue
        for t in targets:
            if not hasattr(t, name):
                try:
                    setattr(t, name, fn)
                except Exception:
                    pass


install()
