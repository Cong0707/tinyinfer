"""Hardware detection + automatic resource planning for tinyinfer.

Everything that used to be a hand-tuned constant (resident gpu_layers, expert slot
count, RAM cache size, prefill chunk) is derived here from the machine the engine
runs on, so tinyinfer adapts to 8GB laptops, 24GB desktops and unified-memory boxes
without manual flags.

Calibration (measured on RTX 5060 8GB Laptop + 47GB RAM):
  * one resident fp8 layer  : ~0.44 GB VRAM (weights 0.19 + allocator/state)
  * embed (129280x5120 bf16) : 1.32 GB VRAM
  * head  (same shape)       : 1.32 GB VRAM (can stay on CPU)
  * one expert slot          : 18.8 MB VRAM (48 slots ~= one decode token's experts)
  * one streamed layer RVAM : ~0.19 GB host RAM (weights) + experts in the RAM cache
  * practical VRAM ceiling   : physical - 0.6 GB (display/context/FX) - 0.8 GB headroom
"""
from __future__ import annotations

import dataclasses
import os

GB = 1 << 30
MB = 1 << 20

# calibrated model-side constants (fp8 non-expert weights, bf16 embed/head)
LAYER_GB = 0.55           # resident per-layer VRAM incl. allocator/state (recalibrated:
                          # 11 layers + embed + slots measured 8671MB, i.e. 0.55GB/layer)
LAYER_HOST_GB = 0.19      # per-layer host RAM for a streamed layer
EMBED_GB = 1.32
HEAD_GB = 1.32
EXPERT_MB = 18.8
KV_GB_PER_KTOK = 0.02     # compressed+window KV for 40 layers per 1k tokens (rough)


@dataclasses.dataclass
class Hardware:
    name: str
    vram_gb: float
    ram_gb: float
    ram_avail_gb: float
    unified: bool = False

    @property
    def vram_budget_gb(self) -> float:
        vram = self.vram_gb
        if self.unified:
            vram = max(vram, self.ram_gb * 0.55)  # unified memory: share the pool
        return max(0.0, vram - 0.6)  # display/context/FX reserve

    @property
    def ram_budget_gb(self) -> float:
        return max(0.0, self.ram_avail_gb - 4.0)  # leave 4GB for the OS/process


def detect() -> Hardware:
    """Detect the machine and return a Hardware descriptor (never raises)."""
    vram_gb = 0.0
    name = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / GB
            name = props.name
    except Exception:
        pass
    ram_gb = ram_avail = 0.0
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_gb = vm.total / GB
        ram_avail = vm.available / GB
    except Exception:
        pass
    return Hardware(name=name, vram_gb=vram_gb, ram_gb=ram_gb, ram_avail_gb=ram_avail)


@dataclasses.dataclass
class Plan:
    gpu_layers: int
    head_on_cpu: bool
    expert_slots: int
    cache_bytes: int
    prefill_chunk: int
    stream_layers: int
    reason: str


def plan(hw: Hardware | None = None, n_layers: int = 40) -> Plan:
    """Derive the engine plan from detected hardware (8GB..80GB, any RAM)."""
    hw = hw or detect()
    budget = hw.vram_budget_gb
    # one expert slot costs EXPERT_MB; allow enough slots for ~2 tokens of top-6
    # v6 flat streaming reserves one extra streamed-layer buffer pair, so the VRAM
    # expert staging pool is smaller than it used to be (still 1-2 layers of top-6).
    slot_budget = min(0.9, max(0.0, budget * 0.12))
    slots = max(0, int(slot_budget * 1024 / EXPERT_MB / 6) * 6)  # multiple of 6
    # TI_FLAT=1 reserves a device buffer pair for v6 flat streaming (two buffers sized
    # to the largest layer, 719MB each here) instead of per-layer allocations.
    flat_gb = 1.45 if os.environ.get("TI_FLAT", "0") == "1" else 0.0
    # tilelang keeps a workspace for the prompt-length kernel set (~1.6GB measured for
    # a 21-token prefill on this box). Without reserving it the allocator has to page
    # once the decode kernels get their own workspace (reserved went past 11.9GB on an
    # 8GB card -> 0.10 tok/s). Small cards reserve it, big ones do not need to.
    ws_gb = 2.3 if budget < 12 else 0.0
    resident = budget - EMBED_GB - slot_budget - flat_gb - ws_gb
    head_gpu = resident - HEAD_GB >= 20 * LAYER_GB
    if not head_gpu:
        resident -= 0.0  # head stays on CPU
    gpu_layers = max(1, min(n_layers, int(resident / LAYER_GB)))
    # host RAM: streamed layers + their RAM cache + KV; keep a quarter for the OS
    host = hw.ram_budget_gb
    cache_gb = min(16.0, max(0.5, host * 0.35))
    # prefill chunk: weights (resident+streamed) + activations per token; chunk 64 on
    # a 47GB box OOM'd, so bound it by the host budget with a safety margin.
    # fixed chunk: tilelang compiles per (M, ...) shape, so a chunk that floats with
    # the free-RAM estimate recompiles the whole kernel set on every start (measured
    #: 60-100s of the first prefill). 64 is the measured sweet spot on this box.
    chunk = 64
    reason = (
        f"{hw.name} vram={hw.vram_gb:.1f}GB ram={hw.ram_gb:.1f}GB(avail {hw.ram_avail_gb:.1f}) "
        f"-> gpu_layers={gpu_layers} slots={slots} cache={cache_gb:.1f}GB chunk={chunk}"
    )
    return Plan(
        gpu_layers=gpu_layers,
        head_on_cpu=not head_gpu,
        expert_slots=slots,
        cache_bytes=int(cache_gb * GB),
        prefill_chunk=chunk,
        stream_layers=n_layers - gpu_layers,
        reason=reason,
    )
