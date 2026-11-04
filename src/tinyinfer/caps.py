"""What the running GPU can actually do, decided at import time.

The engine has to run on anything from Turing to Blackwell, and the differences are not
cosmetic:

  SM 75  Turing (RTX 20, T4)   no bf16 hardware, no fp8, no async copy -- the CUDA
                               headers fall back to software for bf16 conversions and
                               fp8 converts, so results stay correct but those paths
                               cost tens of instructions each
  SM 80  Ampere                 bf16 native, cp.async, tf32
  SM 89  Ada                    + fp8 converts in hardware
  SM 90  Hopper                 + TMA, wgmma, thread-block clusters
  SM 120 Blackwell              + tcgen05 (5th-gen tensor core)

Everything below is a property measured from the device, not a table of part numbers:
the integer is torch's compute capability, and each predicate is a threshold on it. A
kernel that needs fp8 conversion hardware must not be selected on SM 75 even though it
compiles -- the header's software path is correct but an order of magnitude slower, so
picking it there turns a fast path into the slowest one in the engine.
"""
from __future__ import annotations

import torch


class Caps:
    """Thresholds and per-feature availability for one device."""

    def __init__(self, major: int = 0, minor: int = 0, name: str = ""):
        self.major = int(major)
        self.minor = int(minor)
        self.name = name

    @property
    def sm(self) -> int:
        """Compute capability as one integer: 7.5 -> 75, 12.0 -> 120."""
        return self.major * 10 + self.minor

    # ---- feature gates ----
    @property
    def bf16_native(self) -> bool:
        """True when bf16 arithmetic is one instruction rather than an emulation.

        Turing has no bf16 instructions at all; the CUDA headers emulate every convert
        and every add. Correct, but the bf16 kernels cost many times more there, so a
        caller that has an fp16 path should prefer it when this is False.
        """
        return self.sm >= 80

    @property
    def fp8_native(self) -> bool:
        """Hardware e4m3 <-> half conversion (SM 89 and up).

        Kernels that decode fp8 weights per element (the fp8 GEMV, the fp4 swiglu
        quantiser) must be gated on this: on SM 75 they still compile and still return
        the right answer, but each conversion becomes a software sequence and the
        kernel ends up slower than the plain dequantise-once path it was meant to beat.
        """
        return self.sm >= 89

    @property
    def async_copy(self) -> bool:
        return self.sm >= 80

    @property
    def tf32(self) -> bool:
        return self.sm >= 80

    @property
    def tma(self) -> bool:
        return self.sm >= 90

    @property
    def warp_specialized(self) -> bool:
        return self.sm >= 90

    # ---- what to do with the facts ----
    def summary(self) -> str:
        f = []
        for name in ("bf16", "fp8", "async", "tf32", "tma"):
            f.append(f"{name}={'y' if getattr(self, _gate(name)) else 'n'}")
        return f"sm_{self.sm} ({self.name}) " + " ".join(f)

    def __repr__(self) -> str:
        return f"Caps({self.summary()})"


_GATE = {"bf16": "bf16_native", "fp8": "fp8_native", "async": "async_copy",
         "tf32": "tf32", "tma": "tma"}


def _gate(name: str) -> str:
    return _GATE[name]


_CAPS: Caps | None = None


def caps(index: int | None = None) -> Caps:
    """The capabilities of one device (default: the current one), cached."""
    global _CAPS
    if index is not None:
        return _probe(index)
    if _CAPS is None:
        _CAPS = _probe(torch.cuda.current_device() if torch.cuda.is_available() else 0)
    return _CAPS


def _probe(index: int) -> Caps:
    if not torch.cuda.is_available():
        return Caps(0, 0, "cpu")
    p = torch.cuda.get_device_properties(index)
    return Caps(p.major, p.minor, p.name)


def all_caps() -> list[Caps]:
    """Every visible device, so a heterogeneous machine plans for its slowest card."""
    if not torch.cuda.is_available():
        return []
    return [_probe(i) for i in range(torch.cuda.device_count())]


def worst(all_devices: list[Caps]) -> Caps:
    """The weakest capability in the set.

    A tensor-parallel group runs the same kernels on every card, so the group is only as
    capable as its slowest member: a 3090 next to a 2080 Ti must not take the fp8 path
    even though the 3090 could, because the 2080 Ti would run the software conversion.
    """
    if not all_devices:
        return Caps(0, 0, "cpu")
    return min(all_devices, key=lambda c: c.sm)
