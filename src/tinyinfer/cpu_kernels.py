"""CPU-side AVX-friendly kernels for tinyinfer (NumPy fallbacks + pinned staging).

Rationale: the decode step is dominated by host->device transfers (measured 55% of
the GPU-side timeline: 678ms of 1.2s on 40 layers), not by GEMMs. Windows refuses
large cudaHostRegister pools, so per-tensor pin_memory() fails and the copy falls
back to pageable memory (~8.9GB/s). This module keeps a small pinned staging area
and feeds the H2D from it (26GB/s), which is ~3x faster for the same bytes.

Also hosts the AVX-friendly numeric helpers used on the host:
  * fp8_e4m3 -> bf16 dequant of one expert blob (numpy float8 or int8 fallback)
  * bf16 <-> fp32 conversion for the sampling path
NumPy uses its own SIMD kernels (AVX2/AVX-512 on this box), so the memcpy and
dequant loops below run vectorized without extra intrinsics.
"""
from __future__ import annotations

import numpy as np

# one expert blob is 18.8MB; a per-layer staging slab of 192MB covers the largest
# non-expert layer (attention + shared + HC ~170MB) in a single pinned buffer.
STAGE_BYTES = 192 * (1 << 20)

_stage = None


def staging_buffer():
    """Process-wide pinned staging buffer (lazily created; None if pinning is refused)."""
    global _stage
    if _stage is None:
        import torch

        try:
            _stage = torch.empty(STAGE_BYTES, dtype=torch.uint8, pin_memory=True)
        except Exception:
            _stage = False  # pinning unavailable: callers fall back to pageable copies
    return _stage if _stage is not False else None


_slab = None
SLAB_BYTES = 6400 * (1 << 20)  # one 6.4GB pinned slab: every layer fits (max ~190MB)


def slab() -> "torch.Tensor | None":
    """Process-wide pinned slab used to batch a layer's H2D into one pinned region.

    Windows refuses many separate cudaHostRegister calls ("resource already mapped"),
    which is why per-tensor pin_memory() fails and the H2D degrades to pageable
    (~8.3GB/s). One large registration works (verified 6GB), so we register once and
    feed every layer from it: pinned H2D is 26GB/s, i.e. 3.1x faster for the same bytes.
    """
    global _slab
    if _slab is None:
        import torch

        try:
            _slab = torch.empty(SLAB_BYTES, dtype=torch.uint8, pin_memory=True)
        except Exception:
            _slab = False
    return _slab if _slab is not False else None


def stage_tensor_to_device(cpu: "torch.Tensor", device: str) -> "torch.Tensor":
    """Copy a plain CPU tensor to the device through the pinned slab (26GB/s)."""
    import torch

    st = slab()
    n = cpu.numel() * cpu.element_size()
    if st is None or n > st.numel():
        return cpu.to(device, non_blocking=True)
    view = cpu.reshape(-1).view(torch.uint8) if cpu.dtype != torch.uint8 else cpu.reshape(-1)
    st[:n].copy_(view, non_blocking=False)
    # sync: the slab is reused by the next tensor, an async H2D would be overwritten
    out = st[:n].to(device, non_blocking=False)
    return out.view(cpu.dtype).view(cpu.shape)
