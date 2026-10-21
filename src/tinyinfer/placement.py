"""Placement decisions: which ops run on the CPU vs the GPU, derived from the machine.

Rule (measured on this box, but re-evaluable per machine):
    run on CPU  <=>  cpu_ms < gpu_ms + 2 * bytes / h2d_bandwidth

The H2D cost only exists when the operand has to travel; on a big-VRAM box the layers
are resident, the operands are already on the device, and the GPU wins every op. On an
8GB box most layers stream, so every activation round-trips and small ops should stay
on the host. The decision is therefore a function of the hardware plan, not a flag.
"""
from __future__ import annotations

# measured host<->device cost for small transfers (per direction, GB/s)
H2D_PINNED_GB_S = 26.0
H2D_PAGEABLE_GB_S = 8.9


def h2d_bandwidth_gb_s() -> float:
    """Best known host->device bandwidth for this machine (conservative)."""
    return H2D_PAGEABLE_GB_S


def prefer_cpu(nbytes: int, gpu_ms: float, cpu_ms: float, *, pinned: bool = False) -> bool:
    """Decide whether an op should run on the CPU given its size and both timings."""
    bw = (H2D_PINNED_GB_S if pinned else h2d_bandwidth_gb_s()) * 1e9
    round_trip_ms = 2.0 * nbytes / bw * 1e3
    return cpu_ms < gpu_ms + round_trip_ms


def small_op_on_cpu(gpu_layers: int, n_layers: int) -> bool:
    """HC-style small ops should stay on the host while layers stream (H2D round-trips)."""
    return gpu_layers < n_layers
