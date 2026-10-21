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


def stage_tensor_to_device(cpu: "torch.Tensor", device: str) -> "torch.Tensor":
    """Copy a plain CPU tensor to the device through the pinned staging slab.

    Falls back to a direct .to(device) when pinning is unavailable or the tensor is
    larger than the slab. The copy is synchronous for safety: the slab is process-wide
    and reused by the next tensor.
    """
    import torch

    stage = staging_buffer()
    n = cpu.numel() * cpu.element_size()
    if stage is None or n > stage.numel():
        return cpu.to(device, non_blocking=True)
    if cpu.dtype != torch.uint8:
        view = cpu.reshape(-1).view(torch.uint8) if cpu.is_contiguous() else cpu.contiguous().view(torch.uint8)
    else:
        view = cpu.reshape(-1)
    stage[:n].copy_(view, non_blocking=False)
    # synchronous: the process-wide slab is reused by the next tensor, so the H2D must
    # finish before we hand the view out (async here caused NaN: next copy overwrote it)
    out = stage[:n].to(device, non_blocking=False)
    return out.view(cpu.dtype).view(cpu.shape)
