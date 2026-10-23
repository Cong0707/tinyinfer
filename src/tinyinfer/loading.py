"""Tiered weight loading, expert caches and router prefetching for tinyinfer.

Reference implementations (consult when fixing this file):
  * flash-moe  expert fetch/cache/eviction design (same 384-expert top-6 layout:
        LRU/ARC caches, VRAM slot ring, prefetch gating, -18% result when ungated)
  * llama.cpp  mmap row reads for shared weights, O_DIRECT experiments, page-aligned IO
  * DeepInfer  deepinfer/model_executor/layers/... (expert aggregation, engram row reads)
  * vLLM       vllm/model_executor/models/... (paged KV, block manager ideas)
  Measured here: single-shard read ceiling ~5.3 GB/s (buffered readinto, 6 threads),
  page-aligned OVERLAPPED direct 4.76 GB/s, H2D pinned 26.1 GB/s / pageable 8.96 GB/s.

This module is the storage/IO half of tinyinfer.  The model code (tinyinfer.moe)
only knows how to compute; this module decides *where* the bytes live and *when*
they are read:

    NVMe (510 GB checkpoint) -> RAM expert cache (LRU/ARC/S3-FIFO + pin)
                              -> VRAM expert slots (fixed staging ring)
    RAM (17.7 GB non-expert) -> VRAM attention weights (fp8, pinned layers)

The reference numbers used by the planner come from the HF checkpoint itself
(see docs/loading.md):

* routed expert blob = 18,800,640 B (w1/w2/w3 fp4 + their ue8m0 scales)
* one decode token reads 6 experts x 40 layers = 240 blobs = 4.512 GB
* non-expert weights per layer = 170,055,128 B (attention + shared expert + HC)
* Engram tables are 2 x 101.38 GB and are read by row (24 rows x 256 B per token
  per Engram layer), never in bulk.

Nothing here imports tinyinfer.weights at import time.  A ShardStore from
tinyinfer.weights can be passed through ExpertStore.from_store; the only
attributes used are model_dir and weight_map.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import random
import struct
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

__all__ = [
    "ModelGeometry", "TierBudget", "IoBudget", "TensorRegion", "ExpertBlob",
    "ShardIndex", "ShardReader", "ExpertCache", "VramSlotCache", "ExpertStore",
    "ExpertRecord", "Prediction", "PredictionContext", "Predictor",
    "TemporalPredictor", "LookaheadRouterPredictor", "NgramRouterPredictor",
    "OraclePredictor", "EnsemblePredictor", "PrefetchConfig", "Prefetcher",
    "SessionScheduler", "StepPlan", "ExpertSlotPlan", "AttnResidencyPlan",
    "VramResidencyPlan", "PrefillPlanner", "PrefillPlan", "PrefillLayerPlan",
    "PrefillWavePlan", "simulate_workload", "simulate_cache",
    "simulate_prefetch", "run_simulation_suite", "format_sim_table", "selftest",
]

# ---------------------------------------------------------------------------
# Geometry and budgets
# ---------------------------------------------------------------------------

DEFAULT_LAYER_OTHER_BYTES = 170_055_128  # attention + shared expert + HC, fp8/bf16
DEFAULT_EMBED_BYTES = 1_323_827_200      # [129280, 5120] bf16
DEFAULT_HEAD_BYTES = 1_323_827_200
DEFAULT_ENGRAM_TABLE_BYTES = 101_377_628_352  # weight + ue8m0 scale per table
DEFAULT_ENGRAM_LAYERS = (1, 14)
DEFAULT_MTP_LAYERS = 3
DEFAULT_MTP_LAYER_BYTES = 2_655_000_000  # ~128 experts + attention/HC of one DSpark layer
ENGRAM_ROWS_PER_TOKEN = (4 - 1) * 8  # (max_ngram_size-1) * n_heads = 24
ENGRAM_ROW_BYTES = 256
PAGE_BYTES = 4096


@dataclass(frozen=True)
class ModelGeometry:
    """Shape of the released checkpoint; used only for budgets, not for loading."""

    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_experts: int = 384
    n_activated: int = 6
    weight_block: int = 32
    fp4_bits: int = 4
    hc_mult: int = 4
    n_heads: int = 64
    head_dim: int = 512

    def expert_weight_bytes(self) -> int:
        """w1 + w2 + w3 with fp4 packing: 3 * (dim * moe_inter_dim / 2)."""
        return 3 * (self.dim * self.moe_inter_dim) // 2

    def expert_scale_bytes(self) -> int:
        """3 * (dim * moe_inter_dim / weight_block) one-byte ue8m0 scales."""
        return 3 * (self.dim * self.moe_inter_dim) // self.weight_block

    def expert_bytes(self) -> int:
        return self.expert_weight_bytes() + self.expert_scale_bytes()

    def layer_expert_bytes(self) -> int:
        return self.n_experts * self.expert_bytes()

    def token_expert_bytes(self) -> int:
        return self.n_layers * self.n_activated * self.expert_bytes()

    def token_expert_bytes_with_engram(self) -> int:
        return self.token_expert_bytes() + len(DEFAULT_ENGRAM_LAYERS) * ENGRAM_ROWS_PER_TOKEN * ENGRAM_ROW_BYTES


@dataclass(frozen=True)
class IoBudget:
    """Per-token IO cost under a given cache hit rate and drive bandwidth."""

    expert_bytes_per_token: int
    hit_rate: float
    read_bytes_per_token: int
    nvme_ms: float
    pcie_ms: float
    engram_logical_bytes: int
    engram_page_bytes: int
    tokens_per_second: float
    note: str = ""

    @classmethod
    def compute(
        cls,
        geo: ModelGeometry,
        hit_rate: float = 0.0,
        nvme_gbs: float = 8.8,
        pcie_gbs: float = 26.0,
        attention_stream_gb: float = 1.36,
        fixed_ms: float = 60.0,
        engram_hit_rate: float = 0.15,
    ) -> "IoBudget":
        experts = geo.token_expert_bytes()
        read = int(experts * (1.0 - hit_rate))
        nvme_ms = read / (nvme_gbs * 1e9) * 1e3
        pcie_bytes = read + int(attention_stream_gb * 1e9)
        pcie_ms = pcie_bytes / (pcie_gbs * 1e9) * 1e3
        engram_logical = len(DEFAULT_ENGRAM_LAYERS) * ENGRAM_ROWS_PER_TOKEN * ENGRAM_ROW_BYTES
        engram_pages = int(engram_logical / ENGRAM_ROW_BYTES * PAGE_BYTES * (1.0 - engram_hit_rate))
        total_ms = nvme_ms + pcie_ms * 0.35 + fixed_ms  # PCIe partly overlaps NVMe
        return cls(
            expert_bytes_per_token=experts,
            hit_rate=hit_rate,
            read_bytes_per_token=read,
            nvme_ms=nvme_ms,
            pcie_ms=pcie_ms,
            engram_logical_bytes=engram_logical,
            engram_page_bytes=engram_pages,
            tokens_per_second=1000.0 / total_ms if total_ms > 0 else 0.0,
            note="PCIe counted at 35% of its standalone cost (overlaps NVMe/GPU).",
        )


@dataclass(frozen=True)
class TierBudget:
    """VRAM/RAM/NVMe split for the target laptop (8 GB VRAM, 47 GB RAM)."""

    vram_total: float
    vram_desktop_reserve: float
    vram_runtime_reserve: float
    vram_available: float
    attention_layer_bytes: int
    pinned_attention_layers: int
    streamed_attention_layers: int
    attention_pinned_bytes: int
    attention_stream_bytes: int
    expert_slots: int
    expert_slot_bytes: int
    activation_bytes: int
    kv_window_bytes: int
    vram_used: int
    ram_total: float
    ram_os_reserve: float
    ram_non_expert: float
    ram_mtp: float
    ram_kv: float
    ram_staging: float
    ram_expert_cache: float
    ram_headroom: float
    ram_expert_entries: int
    cache_coverage: float

    @classmethod
    def plan(
        cls,
        geo: ModelGeometry = ModelGeometry(),
        *,
        vram_gb: float = 8.0,
        ram_gb: float = 47.2,
        ram_available_gb: float | None = None,
        desktop_reserve_gb: float = 0.9,
        runtime_reserve_gb: float = 0.35,
        activation_gb: float = 0.40,
        kv_window_gb: float = 0.02,
        expert_slots: int = 24,
        os_reserve_gb: float = 6.0,
        include_mtp: bool = False,  # Lead decision: trade the 8 GB MTP set for expert cache
        staging_gb: float = 0.8,
        headroom_gb: float = 2.0,
        target_expert_cache_gb: float | None = None,
    ) -> "TierBudget":
        attn_layer = DEFAULT_LAYER_OTHER_BYTES
        slot_bytes = geo.expert_bytes()
        vram_avail = vram_gb - desktop_reserve_gb - runtime_reserve_gb
        fixed = activation_gb + kv_window_gb + expert_slots * slot_bytes / 1e9
        attn_budget = vram_avail - fixed
        pinned = int(max(0, min(geo.n_layers, math.floor(attn_budget / (attn_layer / 1e9)))))
        streamed = geo.n_layers - pinned
        non_expert = (geo.n_layers * DEFAULT_LAYER_OTHER_BYTES + DEFAULT_EMBED_BYTES + DEFAULT_HEAD_BYTES
                      + len(DEFAULT_ENGRAM_LAYERS) * 157_440_000) / 1e9
        mtp = (DEFAULT_MTP_LAYERS * DEFAULT_MTP_LAYER_BYTES / 1e9) if include_mtp else 0.0
        kv = 0.25
        avail = ram_available_gb if ram_available_gb is not None else ram_gb
        cache = avail - os_reserve_gb - non_expert - mtp - kv - staging_gb - headroom_gb
        if target_expert_cache_gb is not None:
            cache = min(cache, target_expert_cache_gb)
        cache = max(0.0, cache)
        entries = int(cache * 1e9 // slot_bytes)
        coverage = entries / (geo.n_layers * geo.n_experts)
        return cls(
            vram_total=vram_gb,
            vram_desktop_reserve=desktop_reserve_gb,
            vram_runtime_reserve=runtime_reserve_gb,
            vram_available=vram_avail,
            attention_layer_bytes=attn_layer,
            pinned_attention_layers=pinned,
            streamed_attention_layers=streamed,
            attention_pinned_bytes=pinned * attn_layer,
            attention_stream_bytes=streamed * attn_layer,
            expert_slots=expert_slots,
            expert_slot_bytes=expert_slots * slot_bytes,
            activation_bytes=int(activation_gb * 1e9),
            kv_window_bytes=int(kv_window_gb * 1e9),
            vram_used=int((pinned * attn_layer + expert_slots * slot_bytes
                           + activation_gb * 1e9 + kv_window_gb * 1e9)),
            ram_total=ram_gb,
            ram_os_reserve=os_reserve_gb,
            ram_non_expert=non_expert,
            ram_mtp=mtp,
            ram_kv=kv,
            ram_staging=staging_gb,
            ram_expert_cache=cache,
            ram_headroom=headroom_gb,
            ram_expert_entries=entries,
            cache_coverage=coverage,
        )

    def table(self) -> str:
        gb = 1e9
        rows = [
            ("VRAM total", f"{self.vram_total:.2f} GB"),
            ("  - desktop/WDDM reserve", f"{self.vram_desktop_reserve:.2f} GB"),
            ("  - CUDA/runtime reserve", f"{self.vram_runtime_reserve:.2f} GB"),
            ("  attention weights pinned (fp8, 170 MB/layer)", f"{self.pinned_attention_layers} layers = {self.attention_pinned_bytes / gb:.2f} GB"),
            ("  attention weights streamed per token", f"{self.streamed_attention_layers} layers = {self.attention_stream_bytes / gb:.2f} GB"),
            ("  expert slots (18.8 MB each)", f"{self.expert_slots} = {self.expert_slot_bytes / gb:.2f} GB"),
            ("  activations + KV window", f"{(self.activation_bytes + self.kv_window_bytes) / gb:.2f} GB"),
            ("VRAM used", f"{self.vram_used / gb:.2f} GB (free {self.vram_total - self.vram_used / gb:.2f})"),
            ("RAM total", f"{self.ram_total:.2f} GB"),
            ("  - OS + apps reserve", f"{self.ram_os_reserve:.2f} GB"),
            ("  non-expert resident (40 layers + embed + head + engram proj)", f"{self.ram_non_expert:.2f} GB"),
            ("  MTP/DSpark layers (3)", f"{self.ram_mtp:.2f} GB"),
            ("  KV cache (RAM)", f"{self.ram_kv:.2f} GB"),
            ("  pinned staging buffers", f"{self.ram_staging:.2f} GB"),
            ("  headroom", f"{self.ram_headroom:.2f} GB"),
            ("  expert cache", f"{self.ram_expert_cache:.2f} GB = {self.ram_expert_entries} experts "
                               f"({self.cache_coverage * 100:.1f}% of all 15360)"),
        ]
        return "\n".join(f"{a:<62s} {b}" for a, b in rows)

    def io_scenarios(self, geo: ModelGeometry = ModelGeometry(), nvme_gbs: float = 8.8) -> list[IoBudget]:
        stream_gb = self.attention_stream_bytes / 1e9
        return [IoBudget.compute(geo, h, nvme_gbs=nvme_gbs, attention_stream_gb=stream_gb)
                for h in (0.0, 0.2, 0.3, 0.5)]


# ---------------------------------------------------------------------------
# VRAM residency, expert slots and layer-major prefill planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpertSlotPlan:
    """How many VRAM expert slots each phase needs (one 18.8 MB blob per slot).

    Decode needs current-layer experts plus a stage group for the next layer;
    prefill needs a compute group plus a stage group so the next wave's reads
    overlap the current wave's grouped GEMM.  24 slots -> 12+12 prefill and
    6+6 decode, which is what the docs budget for.
    """

    total_slots: int
    blob_bytes: int
    decode_current: int
    decode_stage: int
    prefill_compute: int
    prefill_stage: int

    @classmethod
    def plan(cls, geo: ModelGeometry = ModelGeometry(), slots: int = 24,
             *, double_buffer: bool = True) -> "ExpertSlotPlan":
        slots = max(0, int(slots))
        current = min(geo.n_activated, slots)
        decode_stage = min(current, slots - current) if double_buffer else 0
        prefill_stage = (max(1, slots // 2) if double_buffer and slots > 1 else 0)
        prefill_compute = slots - prefill_stage
        if prefill_compute <= 0:
            prefill_compute, prefill_stage = slots, 0
        return cls(
            total_slots=slots,
            blob_bytes=geo.expert_bytes(),
            decode_current=current,
            decode_stage=decode_stage,
            prefill_compute=prefill_compute,
            prefill_stage=prefill_stage,
        )

    @property
    def total_bytes(self) -> int:
        return self.total_slots * self.blob_bytes

    def slots_for_phase(self, phase: str) -> int:
        if phase == "decode":
            return self.decode_current + self.decode_stage
        if phase == "prefill":
            return self.prefill_compute + self.prefill_stage
        return self.total_slots

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.total_slots < self.decode_current + 1:
            out.append("fewer slots than the 6 experts of one decode layer")
        elif self.total_slots < ExpertSlotPlan.plan(ModelGeometry(), 12).total_slots:
            out.append("less than 12 slots: no double buffering for decode/prefill")
        if self.prefill_compute < 6:
            out.append("prefill wave < 6 experts: grouped GEMM efficiency drops")
        return out

    def table(self) -> str:
        rows = [
            ("total slots", f"{self.total_slots} = {self.total_bytes / 1e6:.1f} MB"),
            ("decode", f"{self.decode_current} current + {self.decode_stage} staged"),
            ("prefill", f"{self.prefill_compute} compute + {self.prefill_stage} staged"),
        ]
        out = "\n".join(f"  {a:<24s} {b}" for a, b in rows)
        warnings = self.warnings()
        if warnings:
            out += "\n  warnings: " + "; ".join(warnings)
        return out


@dataclass(frozen=True)
class AttnResidencyPlan:
    """Which attention/non-expert layer weights stay pinned in VRAM.

    Every layer's weights are needed once per token anyway; pinning N layers
    removes N x 170 MB/token of PCIe traffic.  mode: auto (fit the budget),
    full (all 40; 6.8 GB, only on a bigger card), or an explicit layer count.
    """

    n_layers: int
    layer_bytes: int
    mode: str
    resident_layers: tuple[int, ...]
    streamed_layers: tuple[int, ...]

    @classmethod
    def plan(cls, geo: ModelGeometry = ModelGeometry(), budget_bytes: int = 0,
             *, mode: Any = "auto", layer_bytes: int | None = None) -> "AttnResidencyPlan":
        layer_bytes = int(layer_bytes or DEFAULT_LAYER_OTHER_BYTES)
        if isinstance(mode, str) and mode == "full":
            count = geo.n_layers
        elif isinstance(mode, str) and mode.isdigit():
            count = int(mode)
        elif isinstance(mode, int) and not isinstance(mode, bool):
            count = int(mode)
        else:
            count = int(max(0, budget_bytes) // layer_bytes)
        count = max(0, min(geo.n_layers, count))
        return cls(
            n_layers=geo.n_layers,
            layer_bytes=layer_bytes,
            mode=str(mode),
            resident_layers=tuple(range(count)),
            streamed_layers=tuple(range(count, geo.n_layers)),
        )

    def is_resident(self, layer: int) -> bool:
        return layer in self.resident_layers

    @property
    def pinned_bytes(self) -> int:
        return len(self.resident_layers) * self.layer_bytes

    @property
    def stream_bytes_per_token(self) -> int:
        return len(self.streamed_layers) * self.layer_bytes

    def table(self) -> str:
        return "\n".join([
            f"  mode                {self.mode}",
            f"  pinned layers       {len(self.resident_layers)} / {self.n_layers} "
            f"= {self.pinned_bytes / 1e9:.2f} GB",
            f"  streamed layers     {len(self.streamed_layers)} "
            f"= {self.stream_bytes_per_token / 1e6:.1f} MB/token over PCIe",
        ])


@dataclass(frozen=True)
class VramResidencyPlan:
    """Whole-VRAM budget: attention residency + expert slots + fixed buffers."""

    vram_total_gb: float
    desktop_reserve_gb: float
    runtime_reserve_gb: float
    activation_bytes: int
    kv_bytes: int
    head_bytes: int
    slots: ExpertSlotPlan
    attn: AttnResidencyPlan
    available_bytes: int
    used_bytes: int
    warnings: tuple[str, ...] = ()

    @classmethod
    def plan(
        cls,
        geo: ModelGeometry = ModelGeometry(),
        *,
        vram_gb: float = 8.0,
        desktop_reserve_gb: float = 0.9,
        runtime_reserve_gb: float = 0.35,
        activation_gb: float = 0.40,
        kv_gb: float = 0.02,
        head_gb: float = 0.0,
        slots: int = 24,
        attn_mode: Any = "auto",
    ) -> "VramResidencyPlan":
        available = int((vram_gb - desktop_reserve_gb - runtime_reserve_gb) * 1e9)
        activation = int(activation_gb * 1e9)
        kv = int(kv_gb * 1e9)
        head = int(head_gb * 1e9)
        slot_plan = ExpertSlotPlan.plan(geo, slots)
        fixed = activation + kv + head + slot_plan.total_bytes
        attn = AttnResidencyPlan.plan(geo, available - fixed, mode=attn_mode)
        used = fixed + attn.pinned_bytes
        warns: list[str] = []
        if used > available:
            warns.append(f"over budget by {(used - available) / 1e9:.2f} GB "
                         f"(mode={attn_mode}, slots={slots}); reduce pinned layers or slots")
        if attn.stream_bytes_per_token:
            warns.append(f"{len(attn.streamed_layers)} attention layers stream "
                         f"{attn.stream_bytes_per_token / 1e9:.2f} GB/token over PCIe")
        return cls(
            vram_total_gb=vram_gb,
            desktop_reserve_gb=desktop_reserve_gb,
            runtime_reserve_gb=runtime_reserve_gb,
            activation_bytes=activation,
            kv_bytes=kv,
            head_bytes=head,
            slots=slot_plan,
            attn=attn,
            available_bytes=available,
            used_bytes=used,
            warnings=tuple(warns),
        )

    @property
    def free_bytes(self) -> int:
        return self.available_bytes - self.used_bytes

    @property
    def feasible(self) -> bool:
        return self.used_bytes <= self.available_bytes

    def table(self) -> str:
        rows = [
            ("VRAM total", f"{self.vram_total_gb:.2f} GB"),
            ("  desktop/WDDM reserve", f"{self.desktop_reserve_gb:.2f} GB"),
            ("  CUDA/runtime reserve", f"{self.runtime_reserve_gb:.2f} GB"),
            ("  activations", f"{self.activation_bytes / 1e9:.2f} GB"),
            ("  KV window", f"{self.kv_bytes / 1e9:.3f} GB"),
            ("  head on GPU", f"{self.head_bytes / 1e9:.2f} GB"),
            ("  expert slots", f"{self.slots.total_slots} = {self.slots.total_bytes / 1e9:.2f} GB"),
            ("  attention pinned", f"{len(self.attn.resident_layers)} layers = {self.attn.pinned_bytes / 1e9:.2f} GB"),
            ("VRAM used", f"{self.used_bytes / 1e9:.2f} GB (free {self.free_bytes / 1e9:.2f})"),
        ]
        out = "\n".join(f"  {a:<38s} {b}" for a, b in rows)
        if self.warnings:
            out += "\n  warnings: " + "; ".join(self.warnings)
        return out


@dataclass(frozen=True)
class PrefillWavePlan:
    experts: tuple[int, ...]
    token_counts: tuple[int, ...]
    nbytes: int

    @property
    def tokens(self) -> int:
        return sum(self.token_counts)


@dataclass(frozen=True)
class PrefillLayerPlan:
    layer: int
    n_tokens: int
    requested: int
    n_unique: int
    waves: tuple[PrefillWavePlan, ...]
    expert_bytes: int

    @property
    def mean_tokens_per_expert(self) -> float:
        return self.n_tokens * 6 / max(1, self.n_unique)

    def table(self) -> str:
        return (f"  L{self.layer:02d} tokens={self.n_tokens:6d} unique={self.n_unique:3d}/384 "
                f"waves={len(self.waves):2d} bytes={self.expert_bytes / 1e9:.2f} GB "
                f"tokens/expert={self.mean_tokens_per_expert:.1f}")


@dataclass(frozen=True)
class PrefillPlan:
    n_tokens: int
    block: int
    slots: ExpertSlotPlan
    resident_attn_layers: int
    streamed_attn_layers: int
    layers: tuple[PrefillLayerPlan, ...]
    projected: bool
    expert_bytes: int
    attn_stream_bytes: int
    engram_bytes: int
    memory: Mapping[str, int]
    bandwidth_gbs: float
    time_seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.n_tokens / self.time_seconds if self.time_seconds else 0.0

    def table(self) -> str:
        gb = 1e9
        rows = [
            ("prompt tokens", f"{self.n_tokens}"),
            ("token block", f"{self.block}"),
            ("mode", "projected (routes unknown)" if self.projected else "actual routes"),
            ("expert I/O", f"{self.expert_bytes / gb:.2f} GB"),
            ("attention stream I/O", f"{self.attn_stream_bytes / gb:.3f} GB"),
            ("engram row I/O", f"{self.engram_bytes / 1e6:.2f} MB"),
            ("I/O time @8.8GB/s", f"{self.time_seconds:.2f} s"),
            ("prefill throughput", f"{self.tokens_per_second:.1f} tok/s"),
            ("hidden buffer", f"{self.memory['hidden'] / gb:.2f} GB"),
            ("KV (prompt)", f"{self.memory['kv'] / gb:.2f} GB"),
            ("expert wave slots", f"{self.memory['expert_wave'] / gb:.2f} GB"),
            ("block attention scratch", f"{self.memory['attention_scratch'] / 1e6:.1f} MB"),
        ]
        out = "\n".join(f"  {a:<28s} {b}" for a, b in rows)
        if self.layers:
            shown = self.layers[:4]
            out += "\n  first layers:\n" + "\n".join(p.table() for p in shown)
            if len(self.layers) > len(shown):
                out += f"\n  ... {len(self.layers) - len(shown)} more layers"
        return out

    def pseudo_code(self) -> str:
        return (
            "H = embed(ids)                      # [1, T, hc, dim], RAM or VRAM\n"
            "for layer in range(40):\n"
            "    attn = load_attn_weights(layer)  # fp8 resident or PCIe stream\n"
            "    for b in blocks(T, BLOCK):       # chunked prefill attention\n"
            "        H[b] = attn.forward_prefill(H[b], start_pos=b.start)\n"
            "    logits, routes = gate(all_positions(H))   # [T, 6]\n"
            "    groups = group_tokens_by_expert(routes)   # sort by expert id\n"
            "    for wave in waves(sorted(groups), slots.prefill_compute):\n"
            "        stage_experts(wave.next)      # 12 slots, async pread\n"
            "        for expert in wave.current:   # 12 slots pinned\n"
            "            x = gather(H, groups[expert])  # grouped tokens\n"
            "            y = swiglu_fp4(expert, x)      # dequant inside GEMM\n"
            "            scatter_add(H, groups[expert], y * weight)\n"
            "    release(attn, wave slots)\n"
        )


class PrefillPlanner:
    """Plan a layer-major prefill: experts are read once per layer, not per block.

    The planner is model-agnostic: it needs the token count and (optionally) the
    real route ids per layer.  With routes it produces exact wave lists and
    token counts per expert; without routes it projects the unique-expert set
    (large prompts touch nearly all 384 experts in every layer).
    """

    def __init__(
        self,
        geo: ModelGeometry = ModelGeometry(),
        n_tokens: int = 1024,
        *,
        slots: int = 24,
        block: int = 1024,
        attn_resident_layers: int = 40,
        hidden_dtype_bytes: int = 2,
        kv_bytes_per_token: int = 1400,
        bandwidth_gbs: float = 8.8,
        double_buffer: bool = True,
    ):
        self.geo = geo
        self.n_tokens = max(1, int(n_tokens))
        self.slots = ExpertSlotPlan.plan(geo, slots, double_buffer=double_buffer)
        self.block = max(1, int(block))
        self.attn_resident_layers = max(0, min(geo.n_layers, int(attn_resident_layers)))
        self.hidden_dtype_bytes = hidden_dtype_bytes
        self.kv_bytes_per_token = kv_bytes_per_token
        self.bandwidth_gbs = bandwidth_gbs

    # -- projections -------------------------------------------------------
    def expected_unique_experts(self) -> float:
        p = self.geo.n_activated / max(1, self.geo.n_experts)
        return min(self.geo.n_experts,
                   self.geo.n_experts * (1.0 - (1.0 - p) ** self.n_tokens))

    def memory(self) -> dict[str, int]:
        g = self.geo
        hidden = self.n_tokens * g.hc_mult * g.dim * self.hidden_dtype_bytes
        pre_mix = self.n_tokens * g.hc_mult * 4
        kv = self.n_tokens * self.kv_bytes_per_token
        q_block = self.block * g.n_heads * g.head_dim * self.hidden_dtype_bytes
        scores = self.block * 128 * g.n_heads * 2  # sliding window x heads (fp16 scores)
        wave = self.slots.prefill_compute * self.slots.blob_bytes
        grouped = self.n_tokens * g.n_activated * g.dim * 4
        return {
            "hidden": hidden + pre_mix,
            "kv": kv,
            "attention_scratch": q_block + scores,
            "expert_wave": wave,
            "grouped_activations": grouped,
        }

    # -- exact planning ----------------------------------------------------
    def _flatten_routes(self, routes: Any) -> list[int]:
        out: list[int] = []
        for row in routes:
            if isinstance(row, (list, tuple)):
                out.extend(int(e) for e in row)
            else:  # numpy array / torch tensor row
                out.extend(int(e) for e in row.tolist())
        return out

    def plan_layer(self, layer: int, routes: Any, *, wave_size: int | None = None) -> PrefillLayerPlan:
        ids = self._flatten_routes(routes)
        counts: dict[int, int] = {}
        for e in ids:
            counts[e] = counts.get(e, 0) + 1
        order = sorted(counts, key=lambda e: (-counts[e], e))
        wave = max(1, int(wave_size or self.slots.prefill_compute))
        waves: list[PrefillWavePlan] = []
        for i in range(0, len(order), wave):
            experts = tuple(order[i:i + wave])
            waves.append(PrefillWavePlan(
                experts=experts,
                token_counts=tuple(counts[e] for e in experts),
                nbytes=len(experts) * self.slots.blob_bytes,
            ))
        return PrefillLayerPlan(
            layer=layer,
            n_tokens=self.n_tokens,
            requested=len(ids),
            n_unique=len(order),
            waves=tuple(waves),
            expert_bytes=len(order) * self.slots.blob_bytes,
        )

    def plan(self, routes_by_layer: Mapping[int, Any] | None = None) -> PrefillPlan:
        mem = self.memory()
        engram_per_layer = 24 * 256
        engram = self.n_tokens * engram_per_layer * len(DEFAULT_ENGRAM_LAYERS)
        attn_stream = (self.geo.n_layers - self.attn_resident_layers) * DEFAULT_LAYER_OTHER_BYTES
        if routes_by_layer:
            layers = tuple(self.plan_layer(layer, routes)
                           for layer, routes in sorted(routes_by_layer.items()))
            expert_bytes = sum(p.expert_bytes for p in layers)
            projected = False
        else:
            unique = self.expected_unique_experts()
            layers = ()
            expert_bytes = int(round(unique)) * self.slots.blob_bytes * self.geo.n_layers
            projected = True
        total = expert_bytes + attn_stream + engram
        return PrefillPlan(
            n_tokens=self.n_tokens,
            block=self.block,
            slots=self.slots,
            resident_attn_layers=self.attn_resident_layers,
            streamed_attn_layers=self.geo.n_layers - self.attn_resident_layers,
            layers=layers,
            projected=projected,
            expert_bytes=expert_bytes,
            attn_stream_bytes=attn_stream,
            engram_bytes=engram,
            memory=mem,
            bandwidth_gbs=self.bandwidth_gbs,
            time_seconds=total / (self.bandwidth_gbs * 1e9),
        )


# ---------------------------------------------------------------------------
# safetensors index and expert blobs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorRegion:
    name: str
    shard: str
    file_offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str

    @property
    def end(self) -> int:
        return self.file_offset + self.nbytes


@dataclass(frozen=True)
class ReadSegment:
    """One contiguous file read that lands at dest_offset of a blob buffer."""

    shard: str
    file_offset: int
    nbytes: int
    dest_offset: int


@dataclass(frozen=True)
class TensorSlot:
    name: str
    dest_offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ExpertBlob:
    layer: int
    expert: int
    total_bytes: int
    segments: tuple[ReadSegment, ...]
    tensors: Mapping[str, TensorSlot]

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer, self.expert)


class ShardIndex:
    """Read-only view of the HF safetensors index (headers parsed lazily)."""

    INDEX_FILE = "model.safetensors.index.json"

    def __init__(self, model_dir, *, require_identical: bool = True):
        """Index one checkpoint that may be SPLIT ACROSS SEVERAL DIRECTORIES/DISKS.

        Passing a list (or a ';'/',' separated string) lets the same model live as
        shards spread over multiple drives, which multiplies streaming bandwidth. The
        sources are required to describe the same model: every directory that carries
        an index must agree tensor-for-tensor on the weight_map, and a shard present in
        more than one place must have the same size. A mismatch raises instead of
        silently mixing two checkpoints.
        """
        raw = ([model_dir] if isinstance(model_dir, (str, Path))
               else [p for chunk in model_dir for p in str(chunk).split(";")])
        dirs: list[Path] = []
        for entry in raw:
            for part in str(entry).split(","):
                part = part.strip().strip('"')
                if not part:
                    continue
                p = Path(part)
                if p.exists() and p not in dirs:
                    dirs.append(p)
        if not dirs:
            raise FileNotFoundError(f"no model directory exists among {raw!r}")
        self.model_dirs = dirs
        self.model_dir = dirs[0]          # primary: config/tokenizer live here
        indexes: list[tuple[Path, dict[str, str]]] = []
        for d in dirs:
            p = d / self.INDEX_FILE
            if p.is_file():
                indexes.append((d, json.loads(p.read_text(encoding="utf-8"))["weight_map"]))
        if not indexes:
            raise FileNotFoundError(f"{self.INDEX_FILE} not found under any of {[str(d) for d in dirs]}")
        base_dir, base_map = indexes[0]
        for d, m in indexes[1:]:
            if m == base_map:
                continue
            diff = sorted(k for k in set(base_map) | set(m) if base_map.get(k) != m.get(k))
            if require_identical:
                sample = diff[0] if diff else "<unknown>"
                raise ValueError(
                    "model sources are not the same checkpoint: "
                    f"{len(diff)} tensors differ, e.g. {sample} -> "
                    f"{base_map.get(sample)} in {base_dir} vs {m.get(sample)} in {d}"
                )
        self.weight_map: dict[str, str] = base_map
        self.total_bytes = int(indexes[0][1].get("metadata", {}).get("total_size", 0)) if False else 0
        # resolve every shard to the first directory that has it, and refuse when two
        # copies of the same shard disagree in size (same name, different bytes)
        self._shard_dir: dict[str, Path] = {}
        shards = sorted(set(base_map.values()))
        for shard in shards:
            found: list[tuple[Path, int]] = []
            for d in dirs:
                f = d / shard
                if f.is_file():
                    found.append((d, f.stat().st_size))
            if not found:
                raise FileNotFoundError(f"shard {shard} is missing from every source {[str(d) for d in dirs]}")
            sizes = {sz for _d, sz in found}
            if len(sizes) > 1 and require_identical:
                where = ", ".join(f"{d}={sz}" for d, sz in found)
                raise ValueError(f"shard {shard} differs between sources ({where}): "
                                 "refusing to mix two checkpoints")
            self._shard_dir[shard] = found[0][0]
        # Every directory given must actually contribute shards. A stray path (a typo,
        # or the wrong drive) would otherwise be accepted silently and the user would
        # believe the model is striped across disks when it is not.
        contributed = {d for d in self._shard_dir.values()}
        idle = [d for d in dirs if d not in contributed]
        if idle and len(dirs) > 1 and require_identical:
            raise ValueError(
                "these model directories supplied no shards: "
                + ", ".join(str(d) for d in idle)
                + " (a source must either provide an index that matches the others "
                  "or hold shards the others do not have)"
            )
        self._headers: dict[str, dict[str, Any]] = {}
        self._data_start: dict[str, int] = {}
        self._blob_cache: dict[tuple[int, int], ExpertBlob] = {}
        self._lock = threading.Lock()

    def shard_path(self, shard: str) -> Path:
        d = self._shard_dir.get(shard)
        return (d / shard) if d is not None else (self.model_dir / shard)

    def _header(self, shard: str) -> dict[str, Any]:
        header = self._headers.get(shard)
        if header is not None:
            return header
        with self._lock:
            header = self._headers.get(shard)
            if header is not None:
                return header
            with self.shard_path(shard).open("rb") as f:
                prefix = f.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"truncated safetensors header: {shard}")
                size = struct.unpack("<Q", prefix)[0]
                if size > 200 * 1024 * 1024:
                    raise ValueError(f"implausible safetensors header size {size}: {shard}")
                raw = f.read(size)
                if len(raw) != size:
                    raise ValueError(f"truncated safetensors header: {shard}")
            header = json.loads(raw)
            self._headers[shard] = header
            self._data_start[shard] = 8 + size
            return header

    def region(self, name: str) -> TensorRegion:
        shard = self.weight_map[name]
        header = self._header(shard)
        entry = header[name]
        start = entry["data_offsets"][0]
        nbytes = entry["data_offsets"][1] - start
        return TensorRegion(
            name=name,
            shard=shard,
            file_offset=self._data_start[shard] + start,
            nbytes=nbytes,
            shape=tuple(entry["shape"]),
            dtype=str(entry["dtype"]),
        )

    @staticmethod
    def expert_tensor_names(layer: int, expert: int) -> tuple[str, ...]:
        prefix = f"layers.{layer}.ffn.experts.{expert}."
        return tuple(prefix + f"{w}.{kind}" for w in ("w1", "w2", "w3") for kind in ("weight", "scale"))

    def expert_blob(self, layer: int, expert: int, *, merge_gap: int = 0) -> ExpertBlob:
        key = (layer, expert)
        cached = self._blob_cache.get(key)
        if cached is not None:
            return cached
        regions = [self.region(n) for n in self.expert_tensor_names(layer, expert)]
        regions.sort(key=lambda r: r.file_offset)
        groups: list[list[TensorRegion]] = []
        for r in regions:
            if groups and groups[-1][-1].shard == r.shard and r.file_offset <= groups[-1][-1].end + merge_gap:
                groups[-1].append(r)
            else:
                groups.append([r])
        tensors: dict[str, TensorSlot] = {}
        segments: list[ReadSegment] = []
        dest = 0
        prefix = f"layers.{layer}.ffn.experts.{expert}."
        for group in groups:
            start = group[0].file_offset
            end = max(r.end for r in group)
            segments.append(ReadSegment(shard=group[0].shard, file_offset=start,
                                        nbytes=end - start, dest_offset=dest))
            for r in group:
                # short name ("w1.weight") is what moe.py asks for
                short = r.name[len(prefix):] if r.name.startswith(prefix) else r.name
                tensors[short] = TensorSlot(
                    name=short,
                    dest_offset=dest + (r.file_offset - start),
                    nbytes=r.nbytes,
                    shape=r.shape,
                    dtype=r.dtype,
                )
            dest += end - start
        blob = ExpertBlob(layer=layer, expert=expert, total_bytes=dest,
                          segments=tuple(segments), tensors=tensors)
        self._blob_cache[key] = blob
        return blob

    def expert_blob_bytes(self, layer: int = 0, expert: int = 0) -> int:
        try:
            return self.expert_blob(layer, expert).total_bytes
        except Exception:
            return ModelGeometry().expert_bytes()

    def layer_of_shard(self, shard: str) -> int | None:
        for name in self.weight_map:
            if ".ffn.experts.0.w1.weight" in name and self.weight_map[name] == shard:
                return int(name.split(".")[1])
        return None


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

class _WindowsDirect:
    """Synchronous FILE_FLAG_NO_BUFFERING reads (Windows O_DIRECT analogue).

    Used for the expert stream: the checkpoint is 510 GB, far larger than the
    OS page cache, so read-ahead caching would only evict the non-expert set.
    Every call reads a 4 KiB-aligned superset and copies the wanted bytes out.
    """

    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 1
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x20000000
    FILE_ATTRIBUTE_NORMAL = 0x80

    def __init__(self, max_scratch: int = 32 << 20):
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateFileW.restype = ctypes.c_void_p
        self._k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        self._k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                       ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        self._k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                               ctypes.c_void_p, ctypes.c_uint32]
        self._k32.VirtualAlloc.restype = ctypes.c_void_p
        self._k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
        self._k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
        self._k32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._k32.GetFileSizeEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong)]
        self._k32.GetFileSizeEx.restype = ctypes.c_int
        self._scratch = self._k32.VirtualAlloc(None, max_scratch, 0x1000 | 0x2000, 0x04)
        self._scratch_size = max_scratch
        self._handles: dict[str, int] = {}
        self._sizes: dict[str, int] = {}
        self.bytes_read = 0

    def _handle(self, path: str) -> int:
        h = self._handles.get(path)
        if h is None:
            h = self._k32.CreateFileW(path, self.GENERIC_READ, self.FILE_SHARE_READ, None,
                                      self.OPEN_EXISTING, self.FILE_FLAG_NO_BUFFERING, None)
            if h in (None, -1, 0xFFFFFFFFFFFFFFFF):
                raise OSError(ctypes.get_last_error(), f"CreateFileW({path})")
            self._handles[path] = h
        return h

    def _file_size(self, path: str) -> int:
        size = self._sizes.get(path)
        if size is None:
            v = ctypes.c_longlong(0)
            self._k32.GetFileSizeEx(self._handle(path), ctypes.byref(v))
            size = int(v.value)
            self._sizes[path] = size
        return size

    def read(self, path: str, offset: int, nbytes: int, dest: memoryview) -> None:
        start = offset & ~(PAGE_BYTES - 1)
        end = (offset + nbytes + PAGE_BYTES - 1) & ~(PAGE_BYTES - 1)
        # NO_BUFFERING requires a whole number of sectors AND the requested range has to
        # stay inside the file; the last page of a shard cannot be read this way, so that
        # tail falls back to a cached read (only ever the final tensor of a shard).
        size = self._file_size(path)
        end = min(end, size) & ~(PAGE_BYTES - 1)
        if end < offset + nbytes:
            with open(path, "rb") as f:
                f.seek(offset)
                dest[:nbytes] = f.read(nbytes)
            self.bytes_read += nbytes
            return
        span = end - start
        if span > self._scratch_size:
            raise ValueError(f"direct read of {span} exceeds scratch {self._scratch_size}")
        if self._k32.SetFilePointerEx(self._handle(path), start, None, 0) == 0:
            raise OSError("SetFilePointerEx")
        got = ctypes.c_uint32(0)
        if self._k32.ReadFile(self._handle(path), self._scratch, span, ctypes.byref(got), None) == 0:
            raise OSError(ctypes.get_last_error(), f"ReadFile({path})")
        if got.value != span:
            raise OSError(f"short direct read {got.value}/{span}")
        head = offset - start
        # memmove, not a ctypes slice: slicing a ctypes array builds a Python
        # list of ints and is ~100x slower than the copy itself.
        ctypes.memmove((ctypes.c_char * nbytes).from_buffer(dest), self._scratch + head, nbytes)
        self.bytes_read += nbytes

    def close(self) -> None:
        for h in self._handles.values():
            self._k32.CloseHandle(h)
        self._handles.clear()
        if self._scratch:
            self._k32.VirtualFree(self._scratch, 0, 0x8000)
            self._scratch = None


class ShardReader:
    """Parallel byte-range reader over the HF shards.

    Per-thread file handles, so no lock is held during reads.  direct=True
    uses unbuffered reads on Windows (recommended for the expert stream).
    """

    def __init__(self, index: ShardIndex, threads: int = 6, direct: bool = False):
        self.index = index
        self.threads = max(1, int(threads))
        self.direct = bool(direct)
        self._local = threading.local()
        self._directs: list[_WindowsDirect] = []
        self._direct_lock = threading.Lock()
        self._closed = False
        self._stats_lock = threading.Lock()
        self.bytes_read = 0
        self.read_calls = 0
        self.read_seconds = 0.0
        self._executor: ThreadPoolExecutor | None = None

    def _file(self, shard: str):
        files = getattr(self._local, "files", None)
        if files is None:
            files = self._local.files = {}
        f = files.get(shard)
        if f is None:
            f = files[shard] = self.index.shard_path(shard).open("rb", buffering=0)
        return f

    def _direct_reader(self) -> _WindowsDirect:
        # one buffered reader per thread: a shared 32 MB scratch buffer would
        # serialize the reads and corrupt interleaved copies.
        direct = getattr(self._local, "direct", None)
        if direct is None:
            direct = self._local.direct = _WindowsDirect(max_scratch=32 << 20)
            with self._direct_lock:
                self._directs.append(direct)
        return direct

    def read_into(self, segments: Sequence[ReadSegment], dest: Any, *, offset: int = 0) -> int:
        """Read segments into dest (a writable buffer); returns bytes read."""
        started = time.perf_counter()
        view = memoryview(dest)
        total = 0
        for seg in sorted(segments, key=lambda s: s.file_offset):
            n = seg.nbytes
            if n == 0:
                continue
            d = seg.dest_offset + offset
            if self.direct and os.name == "nt":
                self._direct_reader().read(str(self.index.shard_path(seg.shard)),
                                           seg.file_offset, n, view[d:d + n])
            else:
                f = self._file(seg.shard)
                f.seek(seg.file_offset)
                filled = 0
                while filled < n:
                    got = f.readinto(view[d + filled:d + n])
                    if not got:
                        raise EOFError(f"short read {seg.shard}:{seg.file_offset}+{filled}/{n}")
                    filled += got
            total += n
        with self._stats_lock:
            self.bytes_read += total
            self.read_calls += len(segments)
            self.read_seconds += time.perf_counter() - started
        return total

    def read_blob(self, blob: ExpertBlob) -> bytearray:
        buf = bytearray(blob.total_bytes)
        self.read_into(blob.segments, buf)
        return buf

    def read_many_pinned(self, blobs: Sequence[ExpertBlob],
                         max_workers: int | None = None) -> list["torch.Tensor"]:
        """Read expert blobs straight into pinned host memory.

        Measured: a pageable 190MB read is 3.2GB/s, a pinned readinto is 7.4GB/s, and
        the following H2D is 13.3GB/s pinned vs 8.3 pageable. Keeping the bytes pinned
        from disk all the way to the slot removes the extra pageable bounce.
        """
        import torch

        from concurrent.futures import ThreadPoolExecutor

        if not blobs:
            return []

        def one(blob: ExpertBlob, ex: ThreadPoolExecutor | None) -> "torch.Tensor":
            buf = torch.empty(blob.total_bytes, dtype=torch.uint8, pin_memory=True)
            segs = list(blob.segments)
            if ex is None or len(segs) <= 1:
                self.read_into(segs, buf.numpy())
                return buf
            # One task per segment. An expert blob is a small scale span plus the big
            # weight span, hundreds of MB to GB apart in the shard; reading them in a
            # row keeps QD at (#experts) instead of (#segments), which measured
            # 3.1GB/s against 7.4GB/s for the fully parallel form.
            view = buf.numpy()
            futures = [ex.submit(self.read_into, [seg], view) for seg in segs]
            for f in futures:
                f.result()
            return buf

        workers = max_workers or min(self.threads, len(blobs))
        if workers <= 1 or len(blobs) == 1:
            return [one(b, None) for b in blobs]
        executor = self._executor
        if executor is None:
            executor = self._executor = ThreadPoolExecutor(max_workers=max(self.threads, 2 * len(blobs)),
                                                           thread_name_prefix="tinyinfer-io")
        return [f.result() for f in [executor.submit(one, b, executor) for b in blobs]]

    def io_executor(self):
        """Shared, dedicated read pool (never the ExpertCache prefetch pool: a fetch task
        already runs on that one, so nesting would deadlock)."""
        from concurrent.futures import ThreadPoolExecutor

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max(2, self.threads),
                                                thread_name_prefix="tinyinfer-io")
        return self._executor

    def read_many(self, blobs: Sequence[ExpertBlob], max_workers: int | None = None) -> list[bytearray]:
        if not blobs:
            return []
        workers = max_workers or min(self.threads, len(blobs))
        if workers <= 1 or len(blobs) == 1:
            return [self.read_blob(b) for b in blobs]
        executor = self._executor
        if executor is None:
            executor = self._executor = ThreadPoolExecutor(max_workers=self.threads,
                                                           thread_name_prefix="tinyinfer-io")
        futures = [executor.submit(self.read_blob, b) for b in blobs]
        return [f.result() for f in futures]

    def stats(self) -> dict[str, float]:
        return {
            "bytes_read": float(self.bytes_read),
            "read_calls": float(self.read_calls),
            "read_seconds": self.read_seconds,
            "gbps": (self.bytes_read / 1e9 / self.read_seconds) if self.read_seconds else 0.0,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        with self._direct_lock:
            direct_readers, self._directs = self._directs, []
        for direct in direct_readers:
            direct.close()
        files = getattr(self._local, "files", None)
        if files:
            for f in files.values():
                f.close()
            files.clear()


# ---------------------------------------------------------------------------
# Replacement policies
# ---------------------------------------------------------------------------

class _Policy:
    """Eviction policy interface.  Keys are (layer, expert) tuples."""

    name = "base"

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))

    def on_hit(self, key: tuple[int, int]) -> None:
        raise NotImplementedError

    def on_insert(self, key: tuple[int, int]) -> None:
        raise NotImplementedError

    def on_remove(self, key: tuple[int, int]) -> None:
        raise NotImplementedError

    def evict_one(self, is_pinned: Callable[[tuple[int, int]], bool]) -> tuple[int, int] | None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class LRUPolicy(_Policy):
    name = "lru"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.order: OrderedDict[tuple[int, int], None] = OrderedDict()

    def on_hit(self, key):
        if key in self.order:
            self.order.move_to_end(key)

    def on_insert(self, key):
        self.order[key] = None
        self.order.move_to_end(key)

    def on_remove(self, key):
        self.order.pop(key, None)

    def evict_one(self, is_pinned):
        for key in list(self.order):
            if not is_pinned(key):
                del self.order[key]
                return key
        return None

    def __len__(self):
        return len(self.order)


class ARCPolicy(_Policy):
    """Adaptive Replacement Cache with ghost lists (recency/frequency blend)."""

    name = "arc"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.t1: OrderedDict = OrderedDict()
        self.t2: OrderedDict = OrderedDict()
        self.b1: OrderedDict = OrderedDict()
        self.b2: OrderedDict = OrderedDict()
        self.p = 0.0

    def _trim_ghost(self):
        while len(self.b1) + len(self.b2) > self.capacity:
            if len(self.b1) > len(self.b2):
                self.b1.popitem(last=False)
            else:
                self.b2.popitem(last=False)

    def on_hit(self, key):
        if key in self.t1:
            self.t1.pop(key)
            self.t2[key] = None
        elif key in self.t2:
            self.t2.move_to_end(key)

    def on_insert(self, key):
        if key in self.b1:
            self.b1.pop(key)
            self.p = min(float(self.capacity), self.p + max(1.0, len(self.b2) / max(1, len(self.b1))))
            self.t2[key] = None
        elif key in self.b2:
            self.b2.pop(key)
            self.p = max(0.0, self.p - max(1.0, len(self.b1) / max(1, len(self.b2))))
            self.t2[key] = None
        else:
            self.t1[key] = None
        while len(self.t1) + len(self.t2) > self.capacity:
            # keep the same layer affinity here: this is the other eviction site ARC
            # uses on insert, and leaving it global re-introduced cross-layer flushing
            victim = self.evict_one(lambda k: False, prefer_layer=key[0])
            if victim is None:
                break

    def on_remove(self, key):
        self.t1.pop(key, None)
        self.t2.pop(key, None)

    def evict_one(self, is_pinned, prefer_layer=None):
        # Layer-affine victim first: the routing working set is per layer (each layer has
        # its own ~384 experts), so a global LRU turns every layer into a scan that
        # flushes the others. Measured 21.5% hit rate globally vs 48% for a per-layer
        # cache of the same size, which is the difference between 1.1s and 0.5s of disk
        # per token. The global ARC order still decides which layer-agnostic key goes.
        if prefer_layer is not None:
            for src, ghost in ((self.t1, self.b1), (self.t2, self.b2)):
                victim = next((k for k in src if k[0] == prefer_layer and not is_pinned(k)), None)
                if victim is not None:
                    src.pop(victim)
                    ghost[victim] = None
                    self._trim_ghost()
                    return victim
        # linear scan for the first un-pinned key; no full-dict copies (O(n) lists)
        # inside the cache lock -- misses hit this path on every insert
        if self.t1 and len(self.t1) > self.p:
            victim = next((k for k in self.t1 if not is_pinned(k)), None)
            if victim is not None:
                self.t1.pop(victim)
                self.b1[victim] = None
                self._trim_ghost()
                return victim
        victim = next((k for k in self.t2 if not is_pinned(k)), None)
        if victim is not None:
            self.t2.pop(victim)
            self.b2[victim] = None
            self._trim_ghost()
            return victim
        victim = next((k for k in self.t1 if not is_pinned(k)), None)
        if victim is not None:
            self.t1.pop(victim)
            self.b1[victim] = None
            self._trim_ghost()
            return victim
        return None

    def __len__(self):
        return len(self.t1) + len(self.t2)


class S3FIFOPolicy(_Policy):
    """S3-FIFO: 10% small FIFO + 90% main FIFO with a 2-bit frequency count."""

    name = "s3fifo"

    def __init__(self, capacity: int, small_ratio: float = 0.1):
        super().__init__(capacity)
        self.small_cap = max(1, int(self.capacity * small_ratio))
        self.small: OrderedDict = OrderedDict()
        self.main: OrderedDict = OrderedDict()
        self.freq: dict[tuple[int, int], int] = {}

    def on_hit(self, key):
        self.freq[key] = min(3, self.freq.get(key, 0) + 1)
        if key in self.main:
            self.main.move_to_end(key)

    def on_insert(self, key):
        self.small[key] = None
        self.small.move_to_end(key)
        self.freq.setdefault(key, 0)
        for _ in range(len(self.small) + 2):
            if len(self.small) <= self.small_cap:
                break
            if self._small_step(lambda k: False) is not None:
                break

    def _small_step(self, is_pinned) -> tuple[int, int] | None:
        """Pop the oldest unpinned small entry: promote it or drop it.

        Returns the key only when it leaves the policy (a real victim); a
        promotion returns None and must be retried.
        """
        for key in list(self.small):
            if is_pinned(key):
                continue
            self.small.pop(key)
            if self.freq.get(key, 0) > 0 and len(self.main) < self.capacity:
                self.main[key] = None
                self.main.move_to_end(key)
                return None
            self.freq.pop(key, None)
            return key
        return None

    def _main_step(self, is_pinned) -> tuple[int, int] | None:
        for key in list(self.main):
            if is_pinned(key):
                continue
            if self.freq.get(key, 0) > 0:
                self.freq[key] -= 1
                self.main.move_to_end(key)
                continue
            self.main.pop(key)
            self.freq.pop(key, None)
            return key
        return None

    def on_remove(self, key):
        self.small.pop(key, None)
        self.main.pop(key, None)
        self.freq.pop(key, None)

    def evict_one(self, is_pinned):
        for _ in range(len(self.small) + len(self.main) + 4):
            if len(self.small) > self.small_cap:
                victim = self._small_step(is_pinned)
                if victim is not None:
                    return victim
                continue
            victim = self._main_step(is_pinned)
            if victim is not None:
                return victim
            victim = self._small_step(is_pinned)
            if victim is not None:
                return victim
            if not self.small and not self.main:
                break
        return None

    def __len__(self):
        return len(self.small) + len(self.main)


def make_policy(name: str, capacity: int) -> _Policy:
    name = name.lower()
    if name == "lru":
        return LRUPolicy(capacity)
    if name == "arc":
        return ARCPolicy(capacity)
    if name in ("s3fifo", "s3-fifo", "s3"):
        return S3FIFOPolicy(capacity)
    raise ValueError(f"unknown cache policy {name!r} (use lru|arc|s3fifo)")


# ---------------------------------------------------------------------------
# Expert caches
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    disk_reads: int = 0
    disk_bytes: int = 0
    evictions: int = 0
    loads: int = 0
    prefetch_issued: int = 0
    prefetch_hits: int = 0
    load_seconds: float = 0.0
    fallbacks: int = 0          # reads that could not use a host-pool slot

    def as_dict(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "disk_reads": self.disk_reads,
            "disk_gb": self.disk_bytes / 1e9,
            "evictions": self.evictions,
            "load_seconds": self.load_seconds,
            "fallbacks": self.fallbacks,
        }


class BlobPool:
    """A pre-committed slab of host memory that cached expert blobs are copied into.

    The planner/engine claims this memory once at start-up (so it is really owned and
    cannot be paged out); the cache then slices its entries out of it instead of
    allocating a second copy of the same size.
    """

    def __init__(self, buffer, nbytes: int):
        # ctypes buffers expose a '<c' format that memoryview cannot slice; cast to bytes
        view = memoryview(buffer)
        try:
            view = view.cast("B")
        except (TypeError, ValueError):
            pass
        self._view = view[: int(nbytes)]
        self._cap = int(nbytes)
        self._off = 0
        self._free: list[tuple[int, int]] = []
        self._lock = threading.Lock()
        # byte address of the slab start, so an evicted slice can be turned back into an
        # (offset, size) range and reused instead of leaking the claim
        try:
            self._base = ctypes.addressof(ctypes.c_char.from_buffer(self._view))
        except Exception:
            self._base = -1

    @property
    def capacity(self) -> int:
        return self._cap

    def take(self, n: int) -> memoryview | None:
        n = int(n)
        with self._lock:
            # Recycle an evicted entry's range first: the cache churns ~2GB per decode
            # step, and a pure bump allocator runs the claim out after a dozen tokens
            # (measured: the read path then fell back to pageable reads at 0.55GB/s).
            for idx, (off, size) in enumerate(self._free):
                if size < n:
                    continue
                self._free.pop(idx)
                if size > n + (1 << 16):
                    self._free.append((off + n, size - n))
                return self._view[off:off + n]
            if self._off + n > self._cap:
                return None
            slot = self._view[self._off:self._off + n]
            self._off += n
            return slot

    def release(self, slot) -> bool:
        """Return a slice taken from this pool (best effort, no header to verify)."""
        try:
            off = ctypes.addressof(ctypes.c_char.from_buffer(slot)) - self._base
            size = len(slot)
        except Exception:
            return False
        if off < 0 or off + size > self._cap:
            return False
        with self._lock:
            self._free.append((int(off), int(size)))
        return True

    def reset(self) -> None:
        with self._lock:
            self._off = 0
            self._free.clear()


class ExpertCache:
    """RAM cache of raw fp4 expert blobs (18.8 MB each), shared by all sessions.

    The bytes are kept exactly as stored on disk (no dequantization, no dtype
    conversion): dequantization happens on the device that consumes the blob.
    """

    def __init__(
        self,
        index: ShardIndex,
        reader: ShardReader,
        capacity_bytes: int,
        policy: str = "arc",
        *,
        max_pinned_bytes: int | None = None,
        executor_threads: int | None = None,
        pin_budget_bytes: int | None = None,
        pool=None,
    ):
        self.index = index
        self._pool = pool
        self.reader = reader
        self.blob_bytes = index.expert_blob_bytes()
        self.capacity_entries = max(1, int(capacity_bytes) // max(1, self.blob_bytes))
        self.policy = make_policy(policy, self.capacity_entries)
        self.max_pinned = (self.capacity_entries if max_pinned_bytes is None
                           else max(0, int(max_pinned_bytes) // max(1, self.blob_bytes)))
        self._entries: dict[tuple[int, int], bytearray] = {}
        self._pinned: set[tuple[int, int]] = set()
        self._refs: dict[tuple[int, int], int] = defaultdict(int)
        self._freq: dict[tuple[int, int], int] = defaultdict(int)
        self._lock = threading.RLock()
        self._stats = CacheStats()
        # Pinned host budget for cached blobs. Every cached expert that is pinned can be
        # fed to the GPU as a direct async DMA; pageable ones serialise through the one
        # staging buffer. Pinning the whole cache (12GB) plus the layer slab (16GB) took
        # the box past its free RAM and crashed, so the budget is explicit.
        self.pin_budget = int(pin_budget_bytes) if pin_budget_bytes else 0
        self._pinned_bytes = 0
        # Pre-touch a pool of pinned blobs. Reading into a freshly cudaHostAlloc'd
        # 18.8MB block pays first-touch page faults (14ms in-engine vs 3.2ms for an
        # already-touched block); the pool is filled once, up front, so steady-state
        # misses reuse warm memory instead of faulting.
        self._pin_pool_size = 0
        # Recycled pinned buffers. Reading into a freshly cudaHostAlloc'd 18.8MB buffer
        # pays page faults on first touch: measured 14ms per expert in-engine against
        # 3.2ms for the identical read into an already-touched buffer (5.95GB/s). A
        # free list of evicted pinned blobs removes that cost entirely.
        self._pin_free: list[Any] = []
        self._executor = ThreadPoolExecutor(max_workers=executor_threads or reader.threads,
                                            thread_name_prefix="tinyinfer-prefetch")
        self._closed = False

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def pinned_entries(self) -> int:
        return len(self._pinned)

    def _is_pinned(self, key: tuple[int, int]) -> bool:
        return key in self._pinned

    def _evict_locked(self, prefer_layer: int | None = None) -> bool:
        if prefer_layer is not None and getattr(self.policy, "name", "") == "arc":
            victim = self.policy.evict_one(self._is_pinned, prefer_layer)
        else:
            victim = self.policy.evict_one(self._is_pinned)
        if victim is None:
            return False
        data = self._entries.pop(victim, None)
        self._refs.pop(victim, None)
        self._freq.pop(victim, None)
        self._stats.evictions += 1
        if data is not None:
            try:
                if getattr(data, "is_pinned", None) and data.is_pinned():
                    self._pinned_bytes = max(0, self._pinned_bytes - self.blob_bytes)
                    if len(self._pin_free) < max(8, self.capacity_entries // 4):
                        self._pin_free.append(data)
            except Exception:
                pass
            # Return the pool range so a long run cannot exhaust the claim.
            pool = getattr(self, "_pool", None)
            release = getattr(pool, "release", None) if pool is not None else None
            if release is not None:
                try:
                    release(data)
                except Exception:
                    pass
        return True

    def _insert_locked(self, key: tuple[int, int], data: bytearray, *,
                       pinned: bool = False, admit: bool = True) -> None:
        old = self._entries.get(key)
        if old is not None:
            self._stats.hits += 1
            self.policy.on_hit(key)
            return
        if not admit:
            return  # prefill streaming: read once, keep the cache for decode/hot experts
        while len(self._entries) >= self.capacity_entries:
            if not self._evict_locked(prefer_layer=key[0]):
                break
        # Copy into the pre-committed slab when we have one: the cache must live inside
        # the memory claimed at start-up, not in a second allocation of the same size
        # (that would defeat the claim and put the box back on the page file).
        pool = getattr(self, "_pool", None)
        # memoryview means _read_pinned already wrote it into a pool slot; copying it
        # into a second slot would need twice the pool for the same data.
        if pool is not None and not isinstance(data, memoryview) and isinstance(data, (bytearray, bytes)):
            slot = pool.take(len(data))
            if slot is not None:
                slot[:] = data
                data = slot
        self._entries[key] = data
        try:
            if getattr(data, "is_pinned", None) and data.is_pinned():
                self._pinned_bytes += self.blob_bytes
        except Exception:
            pass
        self.policy.on_insert(key)
        if pinned:
            self.pin(key)
        self._stats.loads += 1

    def _can_pin(self, nbytes: int) -> bool:
        # pin_budget == 0 means "do not pin": the old "<= 0 -> always" reading made every
        # read allocate a fresh 18.8MB pinned tensor OUTSIDE the host pool, and the cache
        # then held a different buffer than the first read had.
        return self.pin_budget > 0 and (self._pinned_bytes + nbytes) <= self.pin_budget

    def _read_many_pooled(self, blobs: Sequence[ExpertBlob]) -> list:
        """Read several missed blobs into host-pool slots in parallel.

        Everything about the misses is independent: separate files, separate byte ranges,
        and ShardReader keeps one file handle + one scratch buffer per thread. Reading
        them one after another on the fetch thread measured 1.1GB/s (118 experts = 2.1s of
        a 3.6s decode step); the SSD reaches 4.6GB/s once a few reads are in flight.
        """
        pool = getattr(self, "_pool", None)
        if pool is None or len(blobs) <= 1:
            return [self._read_pinned(b) for b in blobs]
        slots = []
        for blob in blobs:
            try:
                slots.append(pool.take(blob.total_bytes))
            except Exception:
                slots.append(None)
        target = int(__import__("os").environ.get("TI_READ_THREADS", "0")) or len(blobs)

        def one(blob, slot):
            if slot is None:
                return self._read_pinned(blob)
            try:
                self.reader.read_into(blob.segments, slot)
                return slot
            except Exception:
                return self.reader.read_blob(blob)

        ex = getattr(self.reader, "io_executor", None)
        if ex is None:
            return [one(b, s) for b, s in zip(blobs, slots)]
        pool_exec = ex()
        if target < len(blobs):
            # tuned depth (TI_READ_THREADS): bounded queue keeps other IO responsive
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=target, thread_name_prefix="tinyinfer-io") as tmp:
                return [f.result() for f in [tmp.submit(one, b, s) for b, s in zip(blobs, slots)]]
        return [f.result() for f in [pool_exec.submit(one, b, s) for b, s in zip(blobs, slots)]]

    def _fill_pin_pool(self) -> None:
        """Allocate the reusable pinned pool once (bounded by the pinned budget)."""
        if self._pin_pool_size:
            return
        if getattr(self, "_pool", None) is not None:
            # The cache itself lives in a claimed pinned pool, and reads go straight into
            # a slot of that pool (_read_pinned). A second pinned pool on top (up to
            # 256 x 18.8MB = 4.8GB) put the process past what Windows lets one process
            # pin and every later cudaMalloc failed -- the fresh-allocation path below is
            # therefore not used when a pool exists.
            self._pin_pool_size = 1
            return
        target = min(max(8, self.capacity_entries // 2), 256)
        self._pin_pool_size = 1
        try:
            import torch

            for _ in range(target):
                if self.pin_budget and (self._pinned_bytes + self.blob_bytes) > self.pin_budget:
                    break
                self._pin_free.append(torch.empty(self.blob_bytes, dtype=torch.uint8,
                                                  pin_memory=True))
                self._pinned_bytes += self.blob_bytes
        except Exception:
            pass

    def _take_pinned(self, nbytes: int):
        """A recycled pinned buffer of at least nbytes, or None."""
        if not self._pin_pool_size:
            self._fill_pin_pool()
        while self._pin_free:
            buf = self._pin_free.pop()
            try:
                if int(buf.numel()) >= int(nbytes):
                    return buf[:nbytes]
            except Exception:
                continue
        return None

    def _read_pinned(self, blob: ExpertBlob):
        """Read a blob straight into pinned host memory.

        Pageable bytes serialise every H2D through the single 18.8MB staging buffer
        (one synchronous 18.8MB memcpy per expert under a lock: measured ~11ms per
        layer of pure stall). A pinned source is a direct async DMA at 26.5GB/s.
        Falls back to a pageable bytearray if the pinned pool refuses the allocation.
        """
        pool = getattr(self, "_pool", None)
        if pool is not None:
            # One copy, not two: read the blob into the slot it will live in. Handing
            # back a plain bytearray here made _insert_locked copy it a second time and
            # the extra pinned pool was the only place to stage from.
            slot = pool.take(blob.total_bytes)
            if slot is not None:
                try:
                    # read_into wraps dest in memoryview(dest) and writes through it, so
                    # the writable pool slice must be passed as-is; np.frombuffer gives a
                    # READ-ONLY array, the write threw, the except swallowed it and the
                    # cache kept empty slots (garbage experts, garbled answers).
                    self.reader.read_into(blob.segments, slot)
                    return slot
                except Exception:
                    pass
        self._stats.fallbacks += 1
        if not self._can_pin(blob.total_bytes):
            return self.reader.read_blob(blob)
        buf = self._take_pinned(blob.total_bytes)
        reuse = buf is not None
        try:
            if buf is None:
                import torch

                buf = torch.empty(blob.total_bytes, dtype=torch.uint8, pin_memory=True)
                self._pinned_bytes += int(blob.total_bytes)
            self.reader.read_into(blob.segments, buf.numpy())
            return buf
        except Exception:
            if reuse:
                self._pin_free.append(buf)
            return self.reader.read_blob(blob)

    def get(self, layer: int, expert: int, *, blocking: bool = True,
            admit: bool = True) -> bytearray | None:
        key = (layer, expert)
        with self._lock:
            data = self._entries.get(key)
            if data is not None:
                self._stats.hits += 1
                self._freq[key] += 1
                self.policy.on_hit(key)
                return data
        if not blocking:
            return None
        blob = self.index.expert_blob(layer, expert)
        started = time.perf_counter()
        data = self._read_pinned(blob)
        with self._lock:
            self._stats.misses += 1
            self._stats.disk_reads += 1
            self._stats.disk_bytes += len(data)
            self._stats.load_seconds += time.perf_counter() - started
            self._insert_locked(key, data, admit=admit)
            self._freq[key] += 1
        return data

    def prefetch(self, layer: int, experts: Iterable[int], *, admit: bool = True) -> Future:
        """Read a set of experts into RAM in the background (deduplicated)."""

        def work() -> int:
            loaded = 0
            for expert in experts:
                key = (layer, expert)
                with self._lock:
                    if key in self._entries:
                        continue
                blob = self.index.expert_blob(layer, expert)
                data = self._read_pinned(blob)
                with self._lock:
                    if key not in self._entries:
                        self._stats.disk_reads += 1
                        self._stats.disk_bytes += len(data)
                        self._insert_locked(key, data, admit=admit)
                        loaded += 1
                    self._stats.prefetch_issued += 1
            return loaded

        if self._closed:
            raise RuntimeError("cache closed")
        return self._executor.submit(work)

    def contains(self, layer: int, expert: int) -> bool:
        with self._lock:
            return (layer, expert) in self._entries

    def admit(self, layer: int, expert: int, data: bytearray) -> bytearray:
        """Insert a pre-read expert (locked insert only; the disk read happened outside)."""
        key = (layer, expert)
        with self._lock:
            if __import__("os").environ.get("TI_CACHE_TRACE") == "1" and len(self._entries) < 3:
                print(f"[admit] {key} present={key in self._entries} len={len(data)} "
                      f"type={type(data).__name__}", flush=True)
            if key not in self._entries:
                self._stats.disk_reads += 1
                self._stats.disk_bytes += len(data)
                self._insert_locked(key, data)
        return data

    def pin(self, layer: int, expert: int) -> bool:
        key = (layer, expert)
        with self._lock:
            if key in self._entries:
                self._pinned.add(key)
                return True
            return False

    def unpin(self, layer: int, expert: int) -> None:
        with self._lock:
            self._pinned.discard((layer, expert))

    def pin_hot(self, per_layer: int = 20, *, skip_layers: Iterable[int] = ()) -> int:
        """Pin the most frequently routed experts of each layer (weight selection)."""
        skip = set(skip_layers)
        by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
        with self._lock:
            for key, count in self._freq.items():
                if key[0] in skip:
                    continue
                by_layer[key[0]].append((count, key[1]))
            pinned = 0
            for layer, rows in by_layer.items():
                rows.sort(reverse=True)
                for count, expert in rows[:per_layer]:
                    key = (layer, expert)
                    if key in self._entries and key not in self._pinned and pinned < self.max_pinned:
                        self._pinned.add(key)
                        pinned += 1
            return pinned

    def hot_experts(self, per_layer: int = 8) -> dict[int, list[int]]:
        with self._lock:
            by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for key, count in self._freq.items():
                by_layer[key[0]].append((count, key[1]))
            return {layer: [e for _, e in sorted(rows, reverse=True)[:per_layer]]
                    for layer, rows in by_layer.items()}

    def stats(self) -> dict[str, float]:
        with self._lock:
            out = self._stats.as_dict()
            out.update({"entries": len(self._entries), "pinned": len(self._pinned),
                        "capacity": self.capacity_entries, "policy": self.policy.name,
                        "blob_bytes": float(self.blob_bytes)})
            return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        self.reader.close()


class VramSlotCache:
    """Explicit-state VRAM expert slots: FREE -> RESERVED -> COPYING -> STAGED -> IN_USE.

    Concurrent-safe by construction: a single lock guards all state transitions,
    per-slot CUDA events signal copy completion (the compute stream waits on the
    event instead of polling), and a condition variable lets acquire() block until
    a prefetch finishes. Eviction only ever touches slots with zero references.
    """

    FREE, RESERVED, COPYING, STAGED, IN_USE = 0, 1, 2, 3, 4
    _NAMES = {0: "free", 1: "reserved", 2: "copying", 3: "staged", 4: "in_use"}

    def __init__(self, slots: int, blob_bytes: int, device: str = "cuda",
                 dtype: Any = None, pin_staging: bool = True, slot_arena: bool = False):
        self.slot_arena = bool(slot_arena)
        self.slots = max(0, int(slots))
        self.blob_bytes = int(blob_bytes)
        self.device = device
        self.pin_staging = pin_staging
        self._dtype = dtype
        self._torch = None
        self._buffers: list[Any] = []
        self._staging: Any = None
        self._staging_b: Any = None       # second staging buffer for the prefetch thread
        self._copy_stream: Any = None
        self._copy_events: list[Any] = []
        self._state = [self.FREE] * self.slots
        self._key: list[tuple[int, int] | None] = [None] * self.slots
        self._refs = [0] * self.slots
        self._event: list[Any] = [None] * self.slots
        self._by_key: dict[tuple[int, int], int] = {}
        self._lru: list[int] = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._staging_lock = threading.Lock()
        self._staging_lock_b = threading.Lock()
        self.hits = 0
        self.loads = 0
        self.evictions = 0
        # set while a prefill wave owns the card: the pool must not re-create itself
        # behind its back (it is 1.2GB at 64 slots and the prefill workspace OOMs)
        self._paused = False

    # -- infrastructure ---------------------------------------------------
    def _ensure(self):
        if self._buffers or self.slots == 0 or getattr(self, "_paused", False):
            return
        import torch
        self._torch = torch
        dtype = self._dtype or torch.uint8
        dev = torch.device(self.device)
        try:
            if self.slot_arena:
                # One arena, sliced into slots: a single CUDA allocation instead of one
                # per slot, and contiguous addresses so a batch of experts can later be
                # run by one kernel (Flare's grouped MoE relies on the same contiguity).
                self._arena = torch.empty((self.slots, self.blob_bytes), dtype=dtype, device=dev)
                self._buffers = [self._arena[i] for i in range(self.slots)]
            else:
                self._arena = None
                self._buffers = [torch.empty(self.blob_bytes, dtype=dtype, device=dev)
                                 for _ in range(self.slots)]
        except Exception:
            # never let the staging pool be the allocation that takes the run down:
            # without it experts are served straight from host memory (slower, correct)
            self._buffers = []
            self.slots = 0
            return
        if self.pin_staging and dev.type == "cuda":
            self._staging = torch.empty(self.blob_bytes, dtype=dtype, pin_memory=True)
            self._staging_b = torch.empty(self.blob_bytes, dtype=dtype, pin_memory=True)
            self._copy_stream = torch.cuda.Stream(device=dev)
            self._copy_events = [torch.cuda.Event() for _ in range(self.slots)]

    def _copy_blocking(self, idx: int, data: Any, event: Any = None):
        """Copy data bytes into slot idx; returns the completion CUDA event (or None)."""
        import torch
        if isinstance(data, (bytearray, memoryview)):
            # memoryview comes from the claimed pool: wrap it, never copy it. bytearray
            # wrapping was zero-copy already; routing memoryview through bytearray() made
            # every expert a pageable copy (lost the async DMA path entirely).
            cpu = torch.frombuffer(data, dtype=torch.uint8)
        elif isinstance(data, bytes):
            cpu = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        else:
            cpu = data
        buf = self._buffers[idx]
        if cpu.dtype != buf.dtype:
            cpu = cpu.view(buf.dtype)
        # prefetch thread uses the second staging buffer so it never blocks a fetch copy
        use_b = threading.current_thread().name.startswith("vram-prefetch")
        lock = self._staging_lock_b if use_b else self._staging_lock
        staging = self._staging_b if use_b and self._staging_b is not None else self._staging
        # A host-pool slice IS pinned (cudaHostAlloc) but torch.frombuffer() does not say
        # so, which sent it down the pageable path: an 18.8MB host memcpy on the fetch
        # thread (measured 18ms/layer) and then an async DMA out of the SHARED staging
        # buffer. The next expert's memcpy can overwrite a transfer that is still in
        # flight -- exactly the disagreement two identical greedy runs showed. Issue the
        # DMA against the pool slice itself: no staging, no race, 13.3GB/s.
        # Direct DMA from the pool slice (1.08s/token vs 1.27s through staging). The
        # staging path also produced " 2023-1-1 20:" for a capital question with the same
        # slots: one shared staging buffer is reused while an earlier async DMA out of it
        # may still be in flight.
        if isinstance(data, memoryview) and self._copy_stream is not None:
            ev = event if event is not None else torch.cuda.Event()
            with torch.cuda.stream(self._copy_stream):
                if _async_h2d(buf, data, int(buf.numel())):
                    ev.record(self._copy_stream)
                    return ev
        # Pinned sources go straight to the device as an async DMA (26GB/s, six in
        # parallel). Routing them through the single shared staging buffer serialised
        # them through one 18.8MB host memcpy (measured 18ms/layer). Pageable sources
        # cannot be async, so they copy synchronously.
        if cpu.is_pinned():
            if self._copy_stream is None:
                buf.copy_(cpu, non_blocking=True)
                return None
            # Copy on the dedicated copy stream. A pinned H2D queued on the compute
            # stream makes every later host sync (the per-layer routing bincount) wait
            # for the whole 18.8MB transfer: measured 12.7ms/layer of pure stall.
            ev = event if event is not None else torch.cuda.Event()
            with torch.cuda.stream(self._copy_stream):
                buf.copy_(cpu, non_blocking=True)
                ev.record(self._copy_stream)
            return ev
        if staging is None or self._copy_stream is None:
            buf.copy_(cpu, non_blocking=False)
            return None
        with lock:
            ev = event if event is not None else torch.cuda.Event()
            staging.copy_(cpu)
            with torch.cuda.stream(self._copy_stream):
                buf.copy_(staging, non_blocking=True)
                ev.record(self._copy_stream)
            return ev

    def _find_victim_locked(self) -> int | None:
        for idx in range(self.slots):
            if self._state[idx] == self.FREE:
                return idx
        for idx in self._lru:
            if self._state[idx] in (self.STAGED, self.IN_USE) and self._refs[idx] == 0:
                return idx
        return None

    def _evict_locked(self, idx: int) -> None:
        key = self._key[idx]
        if key is not None:
            self._by_key.pop(key, None)
            self._key[idx] = None
        if idx in self._lru:
            self._lru.remove(idx)
        self._state[idx] = self.FREE
        self._refs[idx] = 0
        self._event[idx] = None
        self.evictions += 1

    def _reserve_locked(self, key: tuple[int, int]) -> int | None:
        if key in self._by_key:
            return self._by_key[key]
        idx = self._find_victim_locked()
        if idx is None:
            return None
        if self._state[idx] != self.FREE:
            self._evict_locked(idx)
        self._key[idx] = key
        self._by_key[key] = idx
        self._state[idx] = self.RESERVED
        self._lru.append(idx)
        self.loads += 1
        return idx

    # -- prefetch path ----------------------------------------------------
    def preload(self, layer: int, expert: int, data: Any) -> bool:
        """Reserve, copy and publish an expert for a later acquire() (background safe)."""
        self._ensure()
        key = (layer, expert)
        with self._lock:
            if key in self._by_key:
                return False
            idx = self._reserve_locked(key)
            if idx is None:
                return False
            self._state[idx] = self.COPYING
        import torch
        ev = torch.cuda.Event()
        with self._lock:
            self._event[idx] = ev  # published before the copy so acquire() can wait on it
        try:
            self._copy_blocking(idx, data, event=ev)
        except BaseException:
            with self._lock:
                self._evict_locked(idx)
                self._cond.notify_all()
            raise
        with self._lock:
            if self._state[idx] == self.COPYING:
                self._state[idx] = self.STAGED
            self._cond.notify_all()
        return True

    # -- fetch path -------------------------------------------------------
    def acquire(self, layer: int, expert: int) -> tuple[int, bool]:
        """Return (slot, ready). ready=False means the caller must copy_in + publish."""
        self._ensure()
        key = (layer, expert)
        with self._cond:
            while True:
                idx = self._by_key.get(key)
                if idx is not None:
                    st = self._state[idx]
                    if st in (self.STAGED, self.IN_USE):
                        self._refs[idx] += 1
                        self.hits += 1
                        self._state[idx] = self.IN_USE
                        return idx, True
                    if st == self.COPYING:
                        # consume immediately; the compute stream waits on the slot event
                        self._refs[idx] += 1
                        self.hits += 1
                        self._state[idx] = self.IN_USE
                        return idx, True
                    # RESERVED: prefetch has not started the copy yet; wait for publish
                    if not self._cond.wait(timeout=5.0):
                        raise RuntimeError(f"VRAM slot prefetch stalled for {key} state={self._NAMES[st]}")
                    continue
                idx = self._reserve_locked(key)
                if idx is None:
                    # no evictable slot: degrade to the RAM record instead of blocking/crashing
                    return -1, False
                self._refs[idx] = 1
                self._state[idx] = self.IN_USE
                return idx, False

    def acquire_many(self, layer: int, experts: list[int]) -> list[tuple[int, bool] | None]:
        """Batch acquire: one lock for the whole expert list (decode fetches 6/ layer).

        Returns per-expert (slot, ready) or None when the caller should fall back to
        the single acquire() path (RESERVED slots that a prefetch is still filling).
        """
        self._ensure()
        out: list[tuple[int, bool] | None] = []
        with self._cond:
            for expert in experts:
                key = (layer, expert)
                idx = self._by_key.get(key)
                if idx is not None:
                    st = self._state[idx]
                    if st == self.RESERVED:
                        out.append(None)  # prefetch in flight: let acquire() wait
                        continue
                    self._refs[idx] += 1
                    self.hits += 1
                    self._state[idx] = self.IN_USE
                    out.append((idx, True))
                    continue
                idx = self._reserve_locked(key)
                if idx is None:
                    out.append((-1, False))
                    continue
                self._refs[idx] = 1
                self._state[idx] = self.IN_USE
                out.append((idx, False))
        return out

    def copy_in(self, slot: int, data: Any):
        return self._copy_blocking(slot, data)

    def publish(self, slot: int, event: Any = None) -> None:
        with self._lock:
            if event is not None:
                self._event[slot] = event
            if self._state[slot] == self.COPYING:
                self._state[slot] = self.STAGED
            self._cond.notify_all()

    def event(self, slot: int):
        """The DMA event for a slot (None once its data is resident)."""
        return self._event[slot]

    def wait_event(self, slot: int) -> None:
        ev = self._event[slot]
        if ev is not None:
            self._torch.cuda.current_stream().wait_event(ev)

    def release(self, layer: int, expert: int) -> None:
        key = (layer, expert)
        with self._lock:
            idx = self._by_key.get(key)
            if idx is None:
                return
            if self._refs[idx] > 0:
                self._refs[idx] -= 1
            if self._refs[idx] == 0 and self._state[idx] == self.IN_USE:
                self._state[idx] = self.STAGED
            self._cond.notify_all()

    def has(self, layer: int, expert: int) -> bool:
        with self._lock:
            return (layer, expert) in self._by_key

    def is_staged(self, layer: int, expert: int) -> bool:
        with self._lock:
            idx = self._by_key.get((layer, expert))
            return idx is not None and self._state[idx] == self.STAGED

    @property
    def staged_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._state if s == self.STAGED)

    @property
    def buffer(self):
        def get(slot: int):
            self._ensure()
            return self._buffers[slot]
        return get

    def stats(self) -> dict[str, float]:
        with self._lock:
            return {"slots": self.slots, "blob_bytes": self.blob_bytes,
                    "hits": self.hits, "loads": self.loads, "evictions": self.evictions,
                    "staged": sum(1 for s in self._state if s == self.STAGED),
                    "copying": sum(1 for s in self._state if s == self.COPYING),
                    "gb": self.slots * self.blob_bytes / 1e9}




# ---------------------------------------------------------------------------
# Expert store (facade used by tinyinfer.moe)
# ---------------------------------------------------------------------------

_DTYPES = {"I8": "int8", "U8": "uint8", "F8_E4M3": "float8_e4m3fn",
           "F8_E8M0": "float8_e8m0fnu", "BF16": "bfloat16", "F16": "float16",
           "F32": "float32", "I32": "int32", "I64": "int64", "BOOL": "bool"}


def _async_h2d(dst, src, nbytes: int) -> bool:
    """cudaMemcpyAsync from a host-pool slice straight into a device tensor.

    torch.frombuffer() over cudaHostAlloc memory reports is_pinned() == False (measured),
    so copy_() stages the bytes through a private pinned buffer and synchronises. The
    source really is pinned, so the DMA is issued directly -- same stream as the caller,
    so the kernels that consume it are ordered after it.
    """
    import torch

    lib = getattr(_async_h2d, "lib", None)
    if lib is None:
        try:
            from .budget import _cudart

            lib = _cudart()
            lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_size_t, ctypes.c_int,
                                            ctypes.c_void_p]
            lib.cudaMemcpyAsync.restype = ctypes.c_int
            _async_h2d.lib = lib
        except Exception:
            return False
    try:
        if torch.is_tensor(src):
            addr = int(src.data_ptr())
        else:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(src))
        rc = lib.cudaMemcpyAsync(
            ctypes.c_void_p(int(dst.data_ptr())), ctypes.c_void_p(int(addr)),
            ctypes.c_size_t(int(nbytes)), 1,
            ctypes.c_void_p(int(torch.cuda.current_stream().cuda_stream)),
        )
        return rc == 0
    except Exception:
        return False


class ExpertRecord:
    """One materialized expert; tensor(name) returns a zero-copy view."""

    __slots__ = ("key", "buffer", "tensors", "device", "event")

    def __init__(self, key: tuple[int, int], buffer: Any, tensors: Mapping[str, TensorSlot],
                 device: str, event: Any = None):
        self.key = key
        self.buffer = buffer  # torch.uint8 tensor (CPU or CUDA) or bytearray
        self.tensors = tensors
        self.device = device
        # Pending H2D for this slot, to be waited on right before the tiles are read.
        # fetch() never waits itself: a wait queued on the compute stream here made the
        # next routing bincount stall on the whole transfer (12.7ms per layer).
        self.event = event

    def tensor(self, name: str):
        import torch
        slot = self.tensors[name]
        torch_dtype = getattr(torch, _DTYPES.get(slot.dtype, "uint8"))
        if not torch.is_tensor(self.buffer):
            buf = torch.frombuffer(self.buffer, dtype=torch.uint8)
        else:
            buf = self.buffer
        view = buf.narrow(0, slot.dest_offset, slot.nbytes).view(torch_dtype)
        return view.view(slot.shape) if len(slot.shape) != 1 else view.view(slot.shape[0])

    def tensors_on(self, device) -> dict:
        """Every tensor of this blob on `device` from ONE transfer.

        Two things go wrong when torch copies this, and both were measured: the six
        tensors are six separate copy_() calls (1440 submissions per decode step), and
        torch.frombuffer() over cudaHostAlloc memory reports is_pinned() == False, so
        every one of them takes the pageable route -- stage 18.8MB into a private pinned
        buffer, then DMA, synchronously. The bytes ARE pinned; the runtime just has to be
        told, so this issues the DMA itself against the pool slice.
        """
        import torch

        def view_of(flat):
            out = {}
            for name, slot in self.tensors.items():
                torch_dtype = getattr(torch, _DTYPES.get(slot.dtype, "uint8"))
                view = flat.narrow(0, slot.dest_offset, slot.nbytes).view(torch_dtype)
                out[name] = (view.view(slot.shape) if len(slot.shape) != 1
                             else view.view(slot.shape[0]))
            return out

        if torch.is_tensor(self.buffer) and self.buffer.device.type == "cuda":
            return view_of(self.buffer)
        if torch.is_tensor(self.buffer):
            buf = self.buffer
            nbytes = int(buf.numel())
            dev = torch.empty(nbytes, dtype=torch.uint8, device=device)
            if not _async_h2d(dev, buf, nbytes):
                dev.copy_(buf, non_blocking=True)
            return view_of(dev)
        nbytes = len(self.buffer)
        dev = torch.empty(nbytes, dtype=torch.uint8, device=device)
        if not _async_h2d(dev, self.buffer, nbytes):
            dev.copy_(torch.frombuffer(self.buffer, dtype=torch.uint8), non_blocking=True)
        return view_of(dev)

    def names(self) -> tuple[str, ...]:
        return tuple(self.tensors)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ExpertRecord(layer={self.key[0]}, expert={self.key[1]}, device={self.device})"


class ExpertStore:
    """Glue between ShardStore-style weights and the tiered caches.

    Typical use from moe.py:

        with store.experts(layer_id, expert_ids) as records:
            for rec in records:
                w1 = rec.tensor("w1.weight")      # uint8 [2304, 2560]
                s1 = rec.tensor("w1.scale")       # float8_e8m0fnu [2304, 160]
                expert = Expert(dequant_fp4_block(w1, s1), ...)

    If vram_slots > 0 and CUDA is available, records point at device buffers;
    otherwise they point directly at the RAM cache (zero-copy).
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        cache_bytes: int = 16 << 30,
        policy: str = "arc",
        vram_slots: int = 0,
        device: str = "cuda",
        reader_threads: int = 6,  # measured optimum: 4-8 threads ~5.3GB/s (buffered IO ceiling)
        direct: bool = False,
        max_pinned_bytes: int | None = None,
        pool=None,
        index: ShardIndex | None = None,
        slot_arena: bool = False,
    ):
        self.index = index or ShardIndex(model_dir)
        self.reader = ShardReader(self.index, threads=reader_threads, direct=direct)
        # per-expert access counts: a chat workload revisits a small set of experts and
        # pinning them (hot_prefetch) turns later reads into RAM hits
        import collections as _collections
        self._freq: _collections.Counter = _collections.Counter()
        # PIN_FRACTION of the expert cache stays pinned: pinned blobs are the only ones
        # that become a direct async DMA instead of a serialised staging memcpy, but the
        # layer slab already holds ~16GB on an 8GB/V47GB box, so pinning all of a 12GB
        # cache pushed the machine over its free RAM and crashed.
        if max_pinned_bytes is None:
            from .budget import budget as _hb
            # Pinned blobs skip the staging memcpy; they also take RAM the layer slab
            # wants. Take whatever the host budget allows (0 = all pageable, still
            # correct, just slower), never a per-machine constant.
            max_pinned_bytes = _hb().grant(int(cache_bytes * 0.5))
        self.ram = ExpertCache(self.index, self.reader, cache_bytes, policy,
                               max_pinned_bytes=max_pinned_bytes,
                               pin_budget_bytes=max_pinned_bytes,
                               pool=pool)
        self.slots = (VramSlotCache(vram_slots, self.index.expert_blob_bytes(), device=device,
                                    slot_arena=slot_arena) if vram_slots else None)
        self.device = device
        self._open: dict[tuple[int, int], int] = defaultdict(int)
        self._stream_reads = 0
        self._stream_bytes = 0
        self._lock = threading.RLock()

    @classmethod
    def from_store(cls, store: Any, **kwargs) -> "ExpertStore":
        """Build from a tinyinfer.weights.ShardStore (duck-typed)."""
        model_dir = getattr(store, "model_dir", None)
        if model_dir is None:
            raise TypeError("store must expose model_dir (tinyinfer.weights.ShardStore does)")
        index = ShardIndex(model_dir)
        weight_map = getattr(store, "weight_map", None)
        if weight_map:
            index.weight_map = weight_map
        return cls(model_dir, index=index, **kwargs)

    def _materialize(self, layer: int, expert: int, *, cache: bool = True,
                     slot: int | None = None, ready: bool = False,
                     data: Any = None) -> ExpertRecord:
        blob = self.index.expert_blob(layer, expert)
        if data is not None:
            # the batch read already produced these bytes; looking them up in the RAM
            # cache again used to miss after the ARC eviction that the admit itself
            # caused, so the same 18.8MB blob was read twice (81 read_blob/step)
            pass
        elif cache:
            data = self.ram.get(layer, expert)
        else:
            # prefill streams each expert once per layer; do not evict the
            # decode/hot working set from the RAM cache for one-off bytes.
            data = self.ram.get(layer, expert, admit=False)
            with self._lock:
                self._stream_reads += 1
                self._stream_bytes += int(getattr(data, "nbytes", len(data)))
        if self.slots is None or self.slots.slots == 0:
            return ExpertRecord((layer, expert), data, blob.tensors, "cpu")
        if slot is None:
            slot, ready = self.slots.acquire(layer, expert)
        if slot < 0:  # no VRAM slot available: serve straight from the RAM cache
            return ExpertRecord((layer, expert), data, blob.tensors, "cpu")
        if ready:
            # a prefetch already owns this slot: hand its event to the caller
            return ExpertRecord((layer, expert), self.slots.buffer(slot), blob.tensors,
                                self.device, self.slots.event(slot))
        ev = self.slots.copy_in(slot, data)
        self.slots.publish(slot, ev)
        # copy_in is a non_blocking DMA on the copy stream. The compute stream has to
        # wait for it before reading the tiles, but only right before the read: doing it
        # here serialised the next layer's routing sync behind the transfer.
        return ExpertRecord((layer, expert), self.slots.buffer(slot), blob.tensors, self.device, ev)

    def _iocp_reader(self):
        if getattr(self, "_iocp", None) is None:
            from .win_iocp import IocpShardReader
            self._iocp = IocpShardReader(depth=8)
            self._iocp_lock = threading.Lock()
        return self._iocp

    def _read_misses_iocp(self, layer: int, misses: list[int]) -> list[Any]:
        """Read every missed expert through one IOCP batch (QD = 2 x misses).

        Measured: 6 threads doing one synchronous read each reach 2.5-4.6GB/s, while
        eight overlapped reads in flight from one thread reach 7.3GB/s -- the SSD needs
        queue depth, not more threads.
        """
        blobs = [self.index.expert_blob(layer, e) for e in misses]
        spans = []
        for b in blobs:
            for seg in b.segments:
                spans.append((str(self.index.model_dir / seg.shard), seg.file_offset, seg.nbytes))
        reader = self._iocp_reader()
        with self._iocp_lock:
            parts = reader.read_many(spans)
        out = []
        k = 0
        for b in blobs:
            buf = bytearray(b.total_bytes)
            for seg in b.segments:
                data = parts[k]
                k += 1
                buf[seg.dest_offset:seg.dest_offset + len(data)] = data[:seg.nbytes]
            out.append(buf)
        return out

    def _pin_cap(self) -> int:
        """Pinned bytes admitted to the RAM cache (TI_PIN_CACHE_GB overrides).

        DEFAULT 0: the separate pinned-slot pool recycles buffers through _pin_free, and
        an evicted buffer could be handed back while a fetch() record still referenced it
        -- measured as expert 0 returning expert 2's bytes on the second forward, which
        made the same layer give a different answer after reset(). With no pin pool every
        expert lives in its own host-pool slot (bump allocation, never reused).
        """
        cap = getattr(self, "_pin_cap_bytes", None)
        if cap is None:
            env = __import__("os").environ.get("TI_PIN_CACHE_GB")
            cap = int(float(env) * (1 << 30)) if env else 0
            self._pin_cap_bytes = cap
        return cap

    def hot_experts(self, n: int = 256) -> list[tuple[int, int]]:
        """Most requested (layer, expert) pairs so far -- the ones worth pinning."""
        return [k for k, _ in self._freq.most_common(n)]

    def hot_prefetch(self, n: int = 256) -> int:
        """Read the hottest experts into RAM ahead of the next token.

        Called between requests (never inside a decode step, where the extra IO would
        only compete with the current fetch). Returns the number of distinct pairs read.
        """
        hot = self.hot_experts(n)
        by_layer: dict[int, list[int]] = {}
        for layer, expert in hot:
            by_layer.setdefault(layer, []).append(expert)
        read = 0
        for layer, experts in by_layer.items():
            try:
                recs = self.fetch(layer, experts, cache=True)
            except Exception:
                continue
            self.release(recs)
            read += len(recs)
        return read

    def fetch(self, layer: int, experts: Iterable[int], *, cache: bool = True) -> list[ExpertRecord]:
        import time as _t
        _t0 = _t.perf_counter()
        try:
            return self._fetch_inner(layer, experts, cache=cache)
        finally:
            self._fetch_ms = getattr(self, "_fetch_ms", 0.0) + (_t.perf_counter() - _t0) * 1e3
            self._fetch_calls = getattr(self, "_fetch_calls", 0) + 1

    def _fetch_inner(self, layer: int, experts: Iterable[int], *, cache: bool = True) -> list[ExpertRecord]:
        experts = [int(e) for e in experts]
        for _e in experts:
            self._freq[(int(layer), _e)] += 1
        blobs: list[Any] = []
        datas: list[Any] | None = None
        # batch the disk misses through read_many (parallel pread, lock-free), then
        # materialise sequentially so the slot state machine stays single-threaded
        if len(experts) > 1:
            # single-expert fetches go to the bytearray reader; the pinned path costs
            # an 18.8MB cudaHostAlloc per blob and measured 2x slower for that case
            misses = [e for e in experts if not self.ram.contains(layer, e)]
            if __import__("os").environ.get("TI_CACHE_TRACE") == "1" and layer == 0:
                print(f"[fetch] layer0 experts={experts} misses={misses}", flush=True)
            if misses:
                # a prefetch for this very expert may already be in flight: wait for it
                # instead of issuing a second read of the same blob on this thread
                misses = [e for (_, e) in self._settle_pending([(layer, e) for e in misses])]
                if __import__("os").environ.get("TI_CACHE_TRACE") == "1" and layer == 0:
                    print(f"[fetch] layer0 settled misses={misses}", flush=True)
            if misses:
                blobs = [self.index.expert_blob(layer, e) for e in misses]
                datas = None
                if __import__("os").environ.get("TI_IOCP") == "1":
                    try:
                        datas = self._read_misses_iocp(layer, misses)
                    except Exception:
                        datas = None
                if datas is not None:
                    for b, d in zip(blobs, datas):
                        self.ram.admit(layer, b.expert, d)
                    blobs = []
                try:
                    # pinned reads: 7.4GB/s vs 3.2 pageable, and the bytes stay pinned
                    # for the H2D-to-slot copy (13.3GB/s vs 8.3)
                    _d0 = __import__("time").perf_counter()
                    # read straight into the host pool: pinned tensors allocated here would
                    # live OUTSIDE the pool (47 experts x 18.8MB x 40 layers ~= 35GB of extra
                    # commit charge, the observed "82GB committed" in the task manager)
                    # Only decode-sized fetches (one token routes to <=8 experts) are
                    # admitted: a prefill batch may touch all 384 experts, and filling the
                    # cache with those evicts the decode working set while making the
                    # stored buffer a different one from the read that produced the output.
                    _admit_ok = cache and len(experts) <= 8
                    if _admit_ok:
                        datas = self.ram._read_many_pooled(blobs) if blobs else []
                    else:
                        # Prefill streams each expert once and never admits it. A pool read
                        # here (15.1s vs 23.9s for 64 tokens) fragments the claim -- decode
                        # then falls back to pageable (1.20 -> 1.03 tok/s). A private pinned
                        # staging buffer was tried next and produced wrong answers (the wave
                        # reuse raced the in-flight DMAs); prefill stays on the bytearray
                        # reader until that is understood.
                        datas = [self.reader.read_blob(b) for b in blobs] if blobs else []
                    self._disk_ms = getattr(self, "_disk_ms", 0.0) + (__import__("time").perf_counter() - _d0) * 1e3
                    for b, d in zip(blobs, datas):
                        if _admit_ok and d is not None:
                            self.ram.admit(layer, b.key[1], d)
                except Exception as exc:
                    if __import__("os").environ.get("TI_CACHE_TRACE") == "1":
                        print(f"[fetch] admit path failed: {exc!r}", flush=True)
        # batch the slot reservations under one lock (decode fetches 6 experts/layer)
        acquired = None
        if self.slots is not None and self.slots.slots > 0 and len(experts) > 1 and cache:
            acquired = self.slots.acquire_many(layer, experts)
        by_expert = {b.key[1]: d for b, d in zip(blobs, datas)} if blobs and datas else {}
        records = []
        for i, expert in enumerate(experts):
            have = by_expert.get(expert)
            if acquired is not None and acquired[i] is not None:
                slot, ready = acquired[i]
                records.append(self._materialize(layer, expert, cache=cache, slot=slot,
                                                 ready=ready, data=have))
            else:
                records.append(self._materialize(layer, expert, cache=cache, data=have))
            with self._lock:
                self._open[(layer, expert)] += 1
        return records

    def prefetch_to_vram(self, layer: int, experts: Iterable[int], *, cache: bool = True) -> int:
        """Stage experts in VRAM slots without taking references (prefill waves).

        Runs on a background thread: the RAM fetch and the copy-stream DMA proceed while
        the main thread keeps computing (acquire() waits on the per-slot event).
        """
        if self.slots is None or self.slots.slots == 0:
            return 0
        experts = [int(e) for e in experts]
        if getattr(self, "_vram_executor", None) is None:
            import concurrent.futures
            # one worker serialised every blob behind a single thread: 186 blobs/token
            # at ~1.2GB/s could never keep up with the compute thread. Six matches the
            # reader's measured scaling (8 threads = 10.9GB/s on cached blobs).
            self._vram_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6,
                                                                        thread_name_prefix="vram-prefetch")

        def one(expert: int) -> int:
            if self.slots.is_staged(layer, expert):
                return 0
            data = self.ram.get(layer, expert, admit=cache)
            return 1 if self.slots.preload(layer, expert, data) else 0

        if getattr(self, "_pending", None) is None:
            self._pending = {}
        # one future per blob: the compute thread can wait for the read it needs
        # instead of issuing a second read of the same 18.8MB (fetch -> _settle_pending)
        for expert in experts:
            fut = self._vram_executor.submit(one, expert)
            self._pending[(int(layer), int(expert))] = fut
        return 0

    def fetch_async(self, layer: int, experts: Iterable[int], *, cache: bool = True) -> Future:
        """Start a disk read on a private pool so the caller can compute meanwhile."""
        if getattr(self, "_fetch_pool", None) is None:
            import concurrent.futures
            self._fetch_pool = concurrent.futures.ThreadPoolExecutor(
                # 12 concurrent blobs (M=2 verify) drove the SSD into a 0.4GB/s regime;
                # four keeps the queue inside what this device likes
                max_workers=4, thread_name_prefix="expert-fetch")
        ids = [int(e) for e in experts]

        def run():
            # inference_mode is thread-local: the slot buffers were allocated under it
            # on the calling thread, so the copy must happen under it here too
            import torch as _t
            with _t.inference_mode():
                return self.fetch(layer, ids, cache=cache)

        return self._fetch_pool.submit(run)

    def drop_slots(self) -> None:
        """Release the VRAM slot pool; _ensure() re-creates it on the next acquire.

        A prefill wave allocates a tilelang workspace that alone needs more than the slot
        pool leaves free on an 8GB card (measured: 64 slots = 1.2GB -> CUDA OOM inside
        the first chunk). Decode is what the slots are for, so prefill hands the VRAM back
        and decode gets it again lazily.
        """
        slots = self.slots
        if slots is None:
            return
        slots._paused = True
        if not getattr(slots, "_buffers", None):
            return
        with slots._lock:
            for idx in range(slots.slots):
                slots._state[idx] = slots.FREE
                slots._key[idx] = None
                slots._refs[idx] = 0
                slots._event[idx] = None
            slots._by_key.clear()
            try:
                slots._lru.clear()
            except Exception:
                pass
            slots._buffers = []
            slots._arena = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def resume_slots(self) -> None:
        """Let the slot pool allocate itself again (the next acquire does it lazily)."""
        if self.slots is not None:
            self.slots._paused = False

    def release(self, records: Iterable[ExpertRecord]) -> None:
        for rec in records:
            layer, expert = rec.key
            with self._lock:
                self._open[(layer, expert)] = max(0, self._open[(layer, expert)] - 1)
            if self.slots is not None:
                self.slots.release(layer, expert)

    @contextmanager
    def experts(self, layer: int, experts: Iterable[int], *,
                cache: bool = True) -> Iterator[list[ExpertRecord]]:
        records = []
        try:
            fetched = self.fetch(layer, experts, cache=cache)
            records.extend(fetched)
            yield fetched
        finally:
            if records:
                self.release(records)

    def prefetch(self, layer: int, experts: Iterable[int]) -> Future:
        """Issue RAM reads ahead of use and remember them so fetch() never reads twice."""
        fut = self.ram.prefetch(layer, experts)
        if getattr(self, "_pending", None) is None:
            self._pending = {}
        for e in experts:
            self._pending[(int(layer), int(e))] = fut
        return fut

    def _settle_pending(self, misses: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Wait for in-flight prefetches of these experts; return the ones still missing.

        Without this, the predict-then-fetch pair reads the same 18.8MB blob twice: the
        prefetch thread has not finished, fetch() sees a cache miss and re-reads it on
        the calling thread -- measured 1.5x the ideal disk bytes per token.
        """
        pending = getattr(self, "_pending", None)
        if not pending:
            return misses
        still: list[tuple[int, int]] = []
        for key in misses:
            fut = pending.pop(key, None)
            if fut is None:
                still.append(key)
                continue
            try:
                # a queued-but-not-started prefetch must not stall the compute thread:
                # past this budget we just read the blob ourselves
                fut.result(timeout=0.5)
            except Exception:
                still.append(key)
                continue
            if not self.ram.contains(*key):
                still.append(key)
        return still

    def pin(self, layer: int, expert: int) -> bool:
        return self.ram.pin(layer, expert)

    def pin_hot(self, per_layer: int = 20) -> int:
        return self.ram.pin_hot(per_layer=per_layer)

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ram": self.ram.stats(), "reader": self.reader.stats(),
                               "stream_reads": self._stream_reads,
                               "stream_gb": self._stream_bytes / 1e9}
        if self.slots is not None:
            out["vram"] = self.slots.stats()
        return out

    def close(self) -> None:
        self.ram.close()


# ---------------------------------------------------------------------------
# Router prediction (branch prediction)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    layer: int
    experts: tuple[int, ...]
    confidence: float = 0.0

    def __len__(self) -> int:
        return len(self.experts)


@dataclass
class PredictionContext:
    session: str = "default"
    step: int = 0
    layer: int = 0
    token_ids: tuple[int, ...] = ()
    hidden: Any = None                 # CPU tensor/array for lookahead predictors
    last_routes: Mapping[int, Sequence[int]] = field(default_factory=dict)  # previous token
    routes_so_far: Mapping[int, Sequence[int]] = field(default_factory=dict)
    phase: str = "decode"


class Predictor(Protocol):
    def predict(self, ctx: PredictionContext, target_layer: int) -> Prediction:
        ...

    def observe(self, ctx: PredictionContext, layer: int, experts: Sequence[int]) -> None:
        ...


class TemporalPredictor:
    """Previous token's route at the same layer (the classic baseline)."""

    name = "temporal"

    def __init__(self, n_experts: int, keep: int = 1):
        self.n_experts = n_experts
        self.keep = keep
        self._last: dict[str, dict[int, tuple[int, ...]]] = defaultdict(dict)
        self._hits: dict[int, int] = defaultdict(int)
        self._seen: dict[int, int] = defaultdict(int)

    def observe(self, ctx: PredictionContext, layer: int, experts: Sequence[int]) -> None:
        self._last[ctx.session][layer] = tuple(int(e) for e in experts)

    def predict(self, ctx: PredictionContext, target_layer: int) -> Prediction:
        routes = self._last.get(ctx.session, {})
        experts = routes.get(target_layer)
        if not experts:
            return Prediction(target_layer, ())
        rate = self._hits.get(target_layer, 0) / max(1, self._seen.get(target_layer, 0))
        return Prediction(target_layer, tuple(experts), confidence=max(0.15, rate))

    def feedback(self, layer: int, hits: int) -> None:
        self._hits[layer] += hits
        self._seen[layer] += 1


class NgramRouterPredictor:
    """Route from the last n token ids: repetitive agent loops reuse routes.

    This is the router-side analogue of an n-gram draft model: token ids are
    known before the layer runs, so this predictor can be evaluated on CPU
    without any hidden state.
    """

    name = "ngram"

    def __init__(self, n: int = 4, max_entries: int = 200_000):
        self.n = n
        self.max_entries = max_entries
        self._table: OrderedDict[tuple[int, ...], dict[int, tuple[int, ...]]] = OrderedDict()
        self._last: dict[str, dict[int, tuple[int, ...]]] = defaultdict(dict)

    @staticmethod
    def context_key(token_ids: Sequence[int], n: int) -> tuple[int, ...] | None:
        if len(token_ids) < n:
            return None
        return tuple(int(t) for t in token_ids[-n:])

    def observe(self, ctx: PredictionContext, layer: int, experts: Sequence[int]) -> None:
        self._last[ctx.session][layer] = tuple(int(e) for e in experts)
        key = self.context_key(ctx.token_ids, self.n)
        if key is None:
            return
        entry = self._table.get(key)
        if entry is None:
            entry = self._table[key] = {}
            if len(self._table) > self.max_entries:
                self._table.popitem(last=False)
        entry[layer] = tuple(int(e) for e in experts)
        self._table.move_to_end(key)

    def predict(self, ctx: PredictionContext, target_layer: int) -> Prediction:
        key = self.context_key(ctx.token_ids, self.n)
        if key is not None:
            entry = self._table.get(key)
            if entry and target_layer in entry:
                return Prediction(target_layer, entry[target_layer], confidence=0.8)
        experts = self._last.get(ctx.session, {}).get(target_layer)
        if experts:
            return Prediction(target_layer, tuple(experts), confidence=0.25)
        return Prediction(target_layer, ())


class LookaheadRouterPredictor:
    """CPU lookahead router: linear map hidden -> gate logits for a target layer.

    The released router is sqrt(softplus(W_gate @ h)) with a small top-k bias.
    We cannot know h of a future layer exactly, but the current sublayer hidden
    state is available on CPU, and a ridge-regressed linear map (5120 -> 384
    per target layer, 2M params = 8 MB fp32) is enough to run on the CPU while
    the GPU works.  fit() consumes logged pairs; the prototype script in
    scripts/prototypes/loading_research contains the collection/training loop.
    """

    name = "lookahead"

    def __init__(self, weights: Mapping[int, Any] | None = None, top_k: int = 8, bias: Any = None):
        self.weights: dict[int, Any] = dict(weights or {})
        self.bias: dict[int, Any] = dict(bias or {})
        self.top_k = top_k

    @staticmethod
    def fit(pairs: Sequence[tuple[Any, Any]], ridge: float = 1e-3) -> tuple[Any, Any]:
        """Ridge fit on CPU: pairs of (hidden [dim], target logits [n_experts])."""
        import numpy as np
        X = np.stack([np.asarray(h, dtype=np.float32) for h, _ in pairs])
        Y = np.stack([np.asarray(y, dtype=np.float32) for _, y in pairs])
        X = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
        A = X.T @ X + ridge * len(X) * np.eye(X.shape[1], dtype=np.float32)
        W = np.linalg.solve(A, X.T @ Y).astype(np.float32)
        return W[:-1].T.copy(), W[-1].copy()

    def load_npz(self, path: str | Path) -> None:
        import numpy as np
        with np.load(path) as data:
            for key in data.files:
                if key.startswith("w"):
                    self.weights[int(key[1:])] = data[key]
                elif key.startswith("b"):
                    self.bias[int(key[1:])] = data[key]

    def save_npz(self, path: str | Path) -> None:
        import numpy as np
        payload = {f"w{k}": v for k, v in self.weights.items()}
        payload.update({f"b{k}": v for k, v in self.bias.items()})
        np.savez(path, **payload)

    def predict(self, ctx: PredictionContext, target_layer: int) -> Prediction:
        W = self.weights.get(target_layer)
        if W is None or ctx.hidden is None:
            return Prediction(target_layer, ())
        import numpy as np
        h = np.asarray(ctx.hidden, dtype=np.float32).reshape(-1)
        logits = W @ h + self.bias.get(target_layer, 0.0)
        k = min(self.top_k, logits.shape[0])
        idx = np.argpartition(-logits, k - 1)[:k]
        idx = idx[np.argsort(-logits[idx])]
        p = np.exp(logits[idx] - logits[idx].max())
        p = p / p.sum()
        return Prediction(target_layer, tuple(int(i) for i in idx), confidence=float(p[0]))

    def observe(self, ctx: PredictionContext, layer: int, experts: Sequence[int]) -> None:
        return None


class OraclePredictor:
    """Ground-truth routes for tests/simulations (upper bound)."""

    name = "oracle"

    def __init__(self, routes: Mapping[tuple[str, int], Sequence[int]] | None = None):
        self.routes = dict(routes or {})

    def set_routes(self, key: tuple[str, int], experts: Sequence[int]) -> None:
        self.routes[key] = tuple(int(e) for e in experts)

    def predict(self, ctx: PredictionContext, target_layer: int) -> Prediction:
        experts = self.routes.get((ctx.session, target_layer))
        return Prediction(target_layer, tuple(experts or ()), confidence=1.0)

    def observe(self, ctx: PredictionContext, layer: int, experts: Sequence[int]) -> None:
        return None


class EnsemblePredictor:
    def __init__(self, members: Sequence[tuple[Predictor, float]]):
        self.members = list(members)

    def predict(self, ctx: PredictionContext, target_layer: int) -> Prediction:
        score: dict[int, float] = {}
        for predictor, weight in self.members:
            pred = predictor.predict(ctx, target_layer)
            for rank, expert in enumerate(pred.experts):
                score[expert] = max(score.get(expert, 0.0), weight * pred.confidence * (1.0 / (1 + rank)))
        if not score:
            return Prediction(target_layer, ())
        ranked = sorted(score.items(), key=lambda kv: -kv[1])
        return Prediction(target_layer, tuple(e for e, _ in ranked), confidence=ranked[0][1])

    def observe(self, ctx: PredictionContext, layer: int, experts: Sequence[int]) -> None:
        for predictor, _ in self.members:
            observe = getattr(predictor, "observe", None)
            if observe:
                observe(ctx, layer, experts)


# ---------------------------------------------------------------------------
# Prefetcher
# ---------------------------------------------------------------------------

@dataclass
class PrefetchConfig:
    lead_layers: int = 1                 # predict this many layers ahead on CPU
    max_experts_per_layer: int = 2       # only high-value predictions
    min_confidence: float = 0.35
    free_compute_ms: float = 5.0         # GPU time available to hide reads
    bandwidth_gbs: float = 8.8
    enabled: bool = True
    gate: bool = True                    # decode gate: off when not paying
    gate_window: int = 16
    gate_min_useful: float = 0.10        # fraction of prefetched experts later used
    gate_min_saved_ms: float = 0.5
    prefetch_next_token: bool = False
    next_token_max_experts: int = 8
    prefetch_to_vram: bool = False

    def free_bytes(self) -> int:
        return max(0, int(self.free_compute_ms / 1e3 * self.bandwidth_gbs * 1e9))

    def max_experts(self, blob_bytes: int) -> int:
        by_bw = self.free_bytes() // max(1, blob_bytes)
        return int(max(0, min(self.max_experts_per_layer, by_bw)))


class Prefetcher:
    """Predicts future experts on CPU and warms the RAM cache during GPU work.

    Prediction never reduces bytes: a wrong prediction is wasted bandwidth.
    It only helps when the predicted reads fit in otherwise idle NVMe time
    (the GPU is busy with attention/FFN) and the prediction is precise.  The
    gate measures precision/usage over a window and switches prefetch off when
    it does not pay, the same lesson as Strata decode gate and flash-moe's
    negative temporal-prediction result.
    """

    def __init__(self, store: ExpertStore, predictor: Predictor, cfg: PrefetchConfig | None = None):
        self.store = store
        self.predictor = predictor
        self.cfg = cfg or PrefetchConfig()
        self.blob_bytes = store.index.expert_blob_bytes()
        self._pending: dict[tuple[str, int], tuple[tuple[int, ...], int]] = {}
        self._futures: list[Future] = []
        self._lock = threading.RLock()
        self._enabled = self.cfg.enabled
        self._window_hits = 0
        self._window_pred = 0
        self._window_wasted_bytes = 0
        self._window = 0
        self.total_issued = 0
        self.total_hits = 0
        self.total_wasted_bytes = 0
        self.total_predicted = 0
        self.gate_trips = 0

    def _budget_experts(self) -> int:
        return self.cfg.max_experts(self.blob_bytes)

    def _predict(self, ctx: PredictionContext, target_layer: int) -> tuple[int, ...]:
        pred = self.predictor.predict(ctx, target_layer)
        if not pred.experts:
            return ()
        k = self._budget_experts()
        experts = pred.experts[:k]
        if pred.confidence < self.cfg.min_confidence and len(experts) > 1:
            return ()
        return tuple(int(e) for e in experts)

    def note_layer(self, ctx: PredictionContext, layer: int, experts: Sequence[int]) -> None:
        """Call after layer layer routed; schedules a lookahead prefetch."""
        actual = tuple(int(e) for e in experts)
        self._score_pending(ctx.session, layer, actual)
        observe = getattr(self.predictor, "observe", None)
        if observe:
            observe(ctx, layer, actual)
        if not self._enabled or self.cfg.lead_layers <= 0:
            return
        target = layer + self.cfg.lead_layers
        if target >= 40:
            return
        predicted = self._predict(ctx, target)
        if not predicted:
            return
        if all(self.store.ram.contains(target, e) for e in predicted):
            return
        with self._lock:
            self._pending[(ctx.session, target)] = (predicted, len(predicted) * self.blob_bytes)
        self._futures.append(self.store.prefetch(target, predicted))
        self._futures = [f for f in self._futures if not f.done()]
        self.total_issued += len(predicted)
        self.total_predicted += 1

    def note_step_start(self, ctx: PredictionContext) -> None:
        """Optional temporal warm-up across tokens (disabled by default)."""
        if not self.cfg.prefetch_next_token or not self._enabled:
            return
        for target in range(0, 40):
            predicted = self._predict(ctx, target)
            if predicted:
                self.store.prefetch(target, predicted)

    def _score_pending(self, session: str, layer: int, actual: Sequence[int]) -> None:
        with self._lock:
            item = self._pending.pop((session, layer), None)
        if item is None:
            return
        predicted, issued_bytes = item
        hits = len(set(predicted) & set(actual))
        wasted = (len(predicted) - hits) * self.blob_bytes
        self.total_hits += hits
        self.total_wasted_bytes += wasted
        self._window_hits += hits
        self._window_pred += len(predicted)
        self._window_wasted_bytes += wasted
        self._window += 1
        if self.cfg.gate and self._window >= self.cfg.gate_window:
            used = self._window_hits / max(1, self._window_pred)
            if used < self.cfg.gate_min_useful:
                self.gate_trips += 1
                self._enabled = False
            self._window = self._window_hits = self._window_pred = self._window_wasted_bytes = 0

    def wait(self, timeout: float | None = None) -> None:
        for fut in list(self._futures):
            fut.result(timeout=timeout)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "issued_experts": self.total_issued,
                "predictions": self.total_predicted,
                "hits": self.total_hits,
                "precision": self.total_hits / max(1, self.total_issued),
                "wasted_gb": self.total_wasted_bytes / 1e9,
                "gate_trips": self.gate_trips,
            }


# ---------------------------------------------------------------------------
# Multi-session scheduler
# ---------------------------------------------------------------------------

@dataclass
class LayerPlan:
    layer: int
    experts: tuple[int, ...]
    per_session: dict[str, tuple[int, ...]]
    nbytes: int
    waves: tuple[tuple[int, ...], ...] = ()

    @property
    def sharing(self) -> float:
        requested = sum(len(v) for v in self.per_session.values())
        return requested / max(1, len(self.experts))


@dataclass
class StepPlan:
    phase: str
    layers: list[LayerPlan]
    total_bytes: int
    dedupe_ratio: float

    def table(self) -> str:
        rows = [f"phase={self.phase} layers={len(self.layers)} bytes={self.total_bytes / 1e9:.3f} GB "
                f"dedupe={self.dedupe_ratio:.2f}"]
        for lp in self.layers[:8]:
            rows.append(f"  L{lp.layer:02d} experts={len(lp.experts):3d} "
                        f"bytes={lp.nbytes / 1e6:7.2f} MB sharing={lp.sharing:.2f}"
                        + (f" waves={len(lp.waves)}" if lp.waves else ""))
        if len(self.layers) > 8:
            rows.append(f"  ... {len(self.layers) - 8} more layers")
        return "\n".join(rows)


class SessionScheduler:
    """Plans which experts to load per decode/prefill step for N sessions.

    Decode: one token per session, union of the 6 experts per layer is loaded
    once and shared by all sessions (dedupe is free throughput).
    Prefill: a chunk touches most of a layer's 384 experts; the plan splits the
    expert set into waves that fit VRAM slots (Wave-Partitional Prefill) so each
    expert is read once per chunk instead of once per token.
    """

    def __init__(self, geo: ModelGeometry = ModelGeometry(), *, vram_slots: int = 24,
                 max_step_bytes: int | None = None):
        self.geo = geo
        self.vram_slots = max(1, vram_slots)
        self.max_step_bytes = max_step_bytes

    def plan(self, sessions: Mapping[str, Mapping[int, Sequence[int]]],
             *, phase: str = "decode") -> StepPlan:
        layers = sorted({layer for routes in sessions.values() for layer in routes})
        plans: list[LayerPlan] = []
        requested = 0
        for layer in layers:
            per_session: dict[str, tuple[int, ...]] = {}
            counts: dict[int, int] = defaultdict(int)
            for session, routes in sessions.items():
                ids = tuple(int(e) for e in routes.get(layer, ()))
                per_session[session] = ids
                requested += len(ids)
                for e in ids:
                    counts[e] += 1
            union = tuple(sorted(counts, key=lambda e: (-counts[e], e)))
            waves: tuple[tuple[int, ...], ...] = ()
            if phase == "prefill" and len(union) > self.vram_slots:
                wave = max(1, (self.vram_slots - self.geo.n_activated) // 2)
                waves = tuple(union[i:i + wave] for i in range(0, len(union), wave))
            plans.append(LayerPlan(layer=layer, experts=union, per_session=per_session,
                                   nbytes=len(union) * self.geo.expert_bytes(), waves=waves))
        total = sum(p.nbytes for p in plans)
        if self.max_step_bytes and total > self.max_step_bytes:
            plans = plans[: max(1, len(plans) // 2)]
            total = sum(p.nbytes for p in plans)
        return StepPlan(phase=phase, layers=plans, total_bytes=total,
                        dedupe_ratio=requested / max(1, sum(len(p.experts) for p in plans)))


# ---------------------------------------------------------------------------
# Simulation (no weights required)
# ---------------------------------------------------------------------------

def simulate_workload(
    steps: int = 48,
    n_layers: int = 40,
    n_experts: int = 384,
    k: int = 6,
    *,
    zipf_s: float = 0.9,
    topic_concentration: float = 0.35,
    topic_reuse: float = 0.5,
    temporal_reuse: float = 0.25,
    seed: int = 0,
) -> list[list[tuple[int, ...]]]:
    """Synthetic per-token routes with Zipf + topic + temporal structure.

    Returns tokens[token][layer] -> expert ids.  The parameters are priors
    taken from published MoE routing traces (heavy-tailed recency/frequency,
    ~25% same-layer temporal overlap); they are not a measurement of this
    model -- collect_router_trace.py replaces them with real routes.
    """
    rng = random.Random(seed)
    import numpy as np
    prng = np.random.default_rng(seed)
    base = 1.0 / np.arange(1, n_experts + 1) ** zipf_s
    base /= base.sum()
    topic = prng.dirichlet(np.full(n_experts, topic_concentration))
    prev: list[tuple[int, ...]] = [() for _ in range(n_layers)]
    out: list[list[tuple[int, ...]]] = []
    for t in range(steps):
        if t % 8 == 0:
            topic = prng.dirichlet(np.full(n_experts, topic_concentration))
        dist = topic_reuse * topic + (1.0 - topic_reuse) * base
        dist = dist / dist.sum()
        token: list[tuple[int, ...]] = []
        for layer in range(n_layers):
            chosen: list[int] = []
            if prev[layer] and rng.random() < temporal_reuse:
                n_copy = rng.choice([1, 2])
                chosen.extend(rng.sample(prev[layer], min(n_copy, len(prev[layer]))))
            while len(chosen) < k:
                e = int(prng.choice(n_experts, p=dist))
                if e not in chosen:
                    chosen.append(e)
            token.append(tuple(chosen))
        prev = token
        out.append(token)
    return out


def simulate_cache(routes: Sequence[Sequence[Sequence[int]]], capacity_entries: int,
                   policy: str = "lru", *, pin_per_layer: int = 0) -> dict[str, float]:
    """Replay routes through a cache; returns hit rate and disk bytes."""
    n_experts = max((e for tok in routes for layer in tok for e in layer), default=0) + 1
    cap = max(1, capacity_entries)
    pol = make_policy(policy, cap)
    store: dict[tuple[int, int], None] = {}
    freq: dict[tuple[int, int], int] = defaultdict(int)
    pinned: set[tuple[int, int]] = set()
    hits = misses = 0
    bytes_read = 0
    warmup = max(1, len(routes) // 4)
    blob = ModelGeometry().expert_bytes()
    for t, token in enumerate(routes):
        for layer, experts in enumerate(token):
            for e in experts:
                key = (layer, e)
                freq[key] += 1
                if key in store:
                    hits += 1
                    pol.on_hit(key)
                else:
                    misses += 1
                    bytes_read += blob
                    while len(store) >= cap:
                        victim = pol.evict_one(lambda k: k in pinned)
                        if victim is None:
                            break
                        store.pop(victim, None)
                    store[key] = None
                    pol.on_insert(key)
        if pin_per_layer and t == warmup:
            by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for (layer, e), c in freq.items():
                by_layer[layer].append((c, e))
            for layer, rows in by_layer.items():
                for _, e in sorted(rows, reverse=True)[:pin_per_layer]:
                    key = (layer, e)
                    if key in store:
                        pinned.add(key)
    total = hits + misses
    return {
        "policy": policy,
        "capacity": cap,
        "hit_rate": hits / total if total else 0.0,
        "disk_gb": bytes_read / 1e9,
        "disk_gb_per_token": bytes_read / 1e9 / max(1, len(routes)),
        "pinned": len(pinned),
    }


def simulate_prefetch(
    routes: Sequence[Sequence[Sequence[int]]],
    *,
    precision: float = 0.6,
    recall: float = 0.5,
    k_prefetch: int = 2,
    read_gbs: float = 8.8,
    compute_ms: float = 5.0,
    blob_bytes: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Layer-time model for CPU lookahead prefetch.

    Each layer reads k_prefetch experts during the previous layer's compute_ms
    of GPU work, then the remaining 6 - hits experts on the critical path.
    A prefetched miss costs bandwidth later (wasted), but the model charges it
    only when it exceeds the free window.
    """
    blob = blob_bytes or ModelGeometry().expert_bytes()
    n_layers_total = len(routes) * (len(routes[0]) if routes else 0)
    read_ms_per_expert = blob / (read_gbs * 1e9) * 1e3
    baseline_ms = n_layers_total * (compute_ms + ModelGeometry().n_activated * read_ms_per_expert)
    free_bytes = compute_ms / 1e3 * read_gbs * 1e9  # NVMe idle while the GPU computes
    total_ms = 0.0
    hits = predicted = 0
    wasted = 0
    for token in routes:
        for experts in token:
            n_pred = min(k_prefetch, len(experts))
            # hits = min(k * precision, K * recall): precision caps false positives,
            # recall caps how much of the 6-expert set a k-sized prediction can cover.
            hits_this = min(float(n_pred) * precision, float(len(experts)) * recall)
            hits_this = max(0.0, min(float(n_pred), hits_this))
            hits += hits_this
            predicted += n_pred
            wasted += (n_pred - hits_this) * blob
            critical_bytes = (len(experts) - hits_this) * blob
            prefetch_bytes = n_pred * blob
            # Prefetched bytes beyond the compute window extend the layer.
            overflow = max(0.0, prefetch_bytes - free_bytes)
            total_ms += compute_ms + (critical_bytes + overflow) / (read_gbs * 1e9) * 1e3
    return {
        "precision": hits / max(1, predicted),
        "k_prefetch": k_prefetch,
        "baseline_ms_per_layer": baseline_ms / max(1, n_layers_total),
        "prefetch_ms_per_layer": total_ms / max(1, n_layers_total),
        "speedup": baseline_ms / total_ms if total_ms else 0.0,
        "wasted_gb": wasted / 1e9,
        "total_gb_per_token": (sum(len(e) for tok in routes for e in tok) * blob / max(1, len(routes))
                               + wasted / max(1, len(routes))) / 1e9,
    }


def run_simulation_suite(
    *,
    steps: int = 64,
    capacities_gb: Sequence[float] = (4, 8, 16, 24, 32),
    policies: Sequence[str] = ("lru", "arc", "s3fifo"),
    seed: int = 0,
) -> list[dict[str, Any]]:
    routes = simulate_workload(steps=steps, seed=seed)
    blob = ModelGeometry().expert_bytes()
    rows: list[dict[str, Any]] = []
    for gb in capacities_gb:
        cap = int(gb * 1e9 // blob)
        for policy in policies:
            row = simulate_cache(routes, cap, policy)
            row.update({"capacity_gb": gb, "tokens": steps})
            rows.append(row)
        row = simulate_cache(routes, cap, "arc", pin_per_layer=20)
        row["policy"] = "arc+pin20"
        row.update({"capacity_gb": gb, "tokens": steps})
        rows.append(row)
    for precision in (0.4, 0.5, 0.6, 0.7):
        row = simulate_prefetch(routes, precision=precision, recall=0.5, k_prefetch=2)
        rows.append({"policy": f"prefetch@{precision:.1f}", "capacity_gb": 0.0,
                     "hit_rate": row["precision"], "disk_gb": row["wasted_gb"],
                     "disk_gb_per_token": row["wasted_gb"] / max(1, steps),
                     "speedup": row["speedup"]})
    return rows


def format_sim_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"{'policy':<16s} {'cache GB':>8s} {'hit rate':>9s} {'disk GB/token':>14s} {'speedup':>8s}"]
    for row in rows:
        lines.append(f"{str(row.get('policy', '')):<16s} {row.get('capacity_gb', 0):>8.1f} "
                     f"{row.get('hit_rate', 0):>9.3f} {row.get('disk_gb_per_token', 0):>14.3f} "
                     f"{row.get('speedup', 0):>8.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self test / CLI
# ---------------------------------------------------------------------------

def _write_fake_model(root: Path, *, layers: int = 1, experts: int = 6) -> None:
    """Create a tiny safetensors checkpoint with the real expert naming scheme."""
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(0)
    all_tensors: list[tuple[str, str, Any]] = []
    weight_map: dict[str, str] = {}
    for layer in range(layers):
        shard = f"model-{layer + 3:05d}-of-00048.safetensors"
        names: dict[str, Any] = {}
        names["layers.%d.ffn.gate.weight" % layer] = torch.zeros(experts, 8)
        for expert in range(experts):
            p = f"layers.{layer}.ffn.experts.{expert}."
            names[p + "w1.weight"] = torch.randint(0, 255, (8, 4), dtype=torch.uint8)
            names[p + "w1.scale"] = torch.ones(8, 1, dtype=torch.uint8)
            names[p + "w2.weight"] = torch.randint(0, 255, (8, 4), dtype=torch.uint8)
            names[p + "w2.scale"] = torch.ones(8, 1, dtype=torch.uint8)
            names[p + "w3.weight"] = torch.randint(0, 255, (8, 4), dtype=torch.uint8)
            names[p + "w3.scale"] = torch.ones(8, 1, dtype=torch.uint8)
        save_file(names, str(root / shard))
        for name in names:
            weight_map[name] = shard
    (root / ShardIndex.INDEX_FILE).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")


def selftest(tmpdir: str | Path | None = None) -> bool:
    import tempfile

    ctx = tempfile.TemporaryDirectory() if tmpdir is None else None
    root = Path(ctx.name if ctx else tmpdir)
    opened: list[Any] = []
    try:
        _selftest_body(root, opened)
        return True
    finally:
        for obj in reversed(opened):
            try:
                obj.close()
            except Exception:
                pass
        if ctx is not None:
            import gc
            gc.collect()
            ctx.cleanup()


def _selftest_body(root: Path, opened: list[Any]) -> None:
    import numpy as np

    _write_fake_model(root, layers=2, experts=6)
    index = ShardIndex(root)
    blob = index.expert_blob(0, 0)
    assert blob.total_bytes == 6 * 20, blob.total_bytes
    reader = ShardReader(index, threads=2)
    opened.append(reader)
    data = reader.read_blob(blob)
    assert isinstance(data, bytearray) and len(data) == blob.total_bytes

    store = ExpertStore.from_store(type("S", (), {"model_dir": root})(),
                                   cache_bytes=4 * blob.total_bytes, policy="lru",
                                   vram_slots=2, device="cpu", reader_threads=2)
    opened.append(store)
    recs = store.fetch(0, [0, 1])
    assert len(recs) == 2
    w1 = recs[0].tensor("w1.weight")
    assert tuple(w1.shape) == (8, 4), w1.shape
    store.release(recs)
    before = store.ram.stats()["disk_reads"]
    recs = store.fetch(0, [0, 1])
    store.release(recs)
    assert store.ram.stats()["disk_reads"] == before, store.ram.stats()

    tiny = ExpertCache(index, ShardReader(index, threads=1),
                       capacity_bytes=2 * blob.total_bytes, policy="s3fifo")
    opened.append(tiny)
    for e in range(6):
        tiny.get(0, e)
    assert len(tiny) <= 2, len(tiny)

    for name in ("lru", "arc", "s3fifo"):
        pol = make_policy(name, 4)
        for e in range(8):
            pol.on_insert((0, e))
            while len(pol) > 4:
                pol.evict_one(lambda k: False)
        assert len(pol) <= 4, (name, len(pol))

    sched = SessionScheduler(vram_slots=8)
    sessions = {"a": {0: (1, 2, 3, 4, 5, 6)}, "b": {0: (1, 2, 3, 9, 10, 11)}}
    plan = sched.plan(sessions, phase="decode")
    assert len(plan.layers[0].experts) == 9, plan.layers[0].experts
    assert plan.dedupe_ratio > 1.0
    big = {"a": {0: tuple(range(40))}}
    plan = sched.plan(big, phase="prefill")
    assert plan.layers[0].waves and len(plan.layers[0].waves) > 1

    ctxp = PredictionContext(session="a", step=1, layer=3, token_ids=(1, 2, 3, 4),
                             hidden=np.zeros(8, dtype=np.float32))
    trend = TemporalPredictor(384)
    trend.observe(ctxp, 4, (1, 2))
    assert trend.predict(ctxp, 4).experts == (1, 2)
    ng = NgramRouterPredictor(n=4)
    ng.observe(ctxp, 4, (5, 6))
    assert ng.predict(ctxp, 4).experts == (5, 6)
    W, b = LookaheadRouterPredictor.fit([(np.zeros(8), np.arange(6, dtype=np.float32)),
                                        (np.ones(8), np.arange(6, dtype=np.float32)[::-1])])
    look = LookaheadRouterPredictor({4: W}, bias={4: b})
    assert look.predict(ctxp, 4).experts

    store2 = ExpertStore.from_store(type("S", (), {"model_dir": root})(),
                                    cache_bytes=8 * blob.total_bytes, policy="arc", device="cpu")
    opened.append(store2)
    oracle = OraclePredictor({("a", 1): (0, 1)})
    prefetcher = Prefetcher(store2, oracle, PrefetchConfig(lead_layers=1, gate=False))
    for _ in range(3):
        prefetcher.note_layer(ctxp, 0, (0, 1))
    prefetcher.wait()
    assert prefetcher.total_issued > 0, prefetcher.stats()

    # --- layer-major prefill planner + residency plans ---
    slot_plan = ExpertSlotPlan.plan(ModelGeometry(), 24)
    assert (slot_plan.prefill_compute, slot_plan.prefill_stage) == (12, 12), slot_plan
    assert slot_plan.decode_current == 6
    vram_plan = VramResidencyPlan.plan()
    assert vram_plan.feasible and len(vram_plan.attn.resident_layers) >= 1
    planner = PrefillPlanner(ModelGeometry(), 512, slots=24, block=256, attn_resident_layers=34)
    projected = planner.plan()
    assert projected.projected and projected.expert_bytes > 0 and projected.time_seconds > 0
    route_map = {layer: np.tile(np.arange(6, dtype=np.int16), (512, 1)) for layer in range(2)}
    actual = planner.plan(route_map)
    assert not actual.projected and len(actual.layers) == 2 and actual.layers[0].waves
    assert actual.layers[0].n_unique == 6
    print(f"prefill(projected 512): {projected.expert_bytes / 1e9:.1f} GB, "
          f"{projected.time_seconds:.1f} s, {projected.tokens_per_second:.1f} tok/s; "
          f"VRAM plan {vram_plan.used_bytes / 1e9:.2f}/{vram_plan.available_bytes / 1e9:.2f} GB")

    # --- VRAM staging (double buffer) + cache-less streaming ---
    store3 = ExpertStore.from_store(type("S", (), {"model_dir": root})(),
                                    cache_bytes=2 * blob.total_bytes, policy="arc",
                                    vram_slots=4, device="cpu")
    opened.append(store3)
    staged = store3.prefetch_to_vram(0, [0, 1])
    assert staged == 2, staged
    assert store3.slots.staged_count == 2
    entries_before = len(store3.ram)
    recs = store3.fetch(0, [2], cache=False)
    assert len(recs) == 1 and store3.stats()["stream_reads"] == 1
    assert len(store3.ram) == entries_before, "cache=False must not admit"
    store3.release(recs)
    with store3.experts(0, [0]) as recs:
        assert store3.slots.is_staged(0, 0) is False
    assert not store3.slots.has(0, 0) or store3.slots.staged_count == 1

    budget = TierBudget.plan()
    assert budget.ram_expert_entries > 0
    io = IoBudget.compute(ModelGeometry(), hit_rate=0.3)
    assert io.tokens_per_second > 0
    rows = run_simulation_suite(steps=16, capacities_gb=(4, 16), policies=("lru", "arc"))
    assert rows
    print("budget plan:\n" + budget.table())
    print("\nIO scenarios (8.8 GB/s):")
    for b in budget.io_scenarios():
        print(f"  hit={b.hit_rate:4.1%} read={b.read_bytes_per_token/1e9:5.2f} GB "
              f"nvme={b.nvme_ms:6.0f} ms pcie={b.pcie_ms:5.0f} ms ~{b.tokens_per_second:4.2f} tok/s")
    print("\nsimulation:\n" + format_sim_table(rows))
    print("\nSELFTEST PASSED")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="tinyinfer tiered loading utilities")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("selftest")
    p_sim = sub.add_parser("sim")
    p_sim.add_argument("--steps", type=int, default=64)
    p_budget = sub.add_parser("budget")
    p_budget.add_argument("--vram", type=float, default=8.0)
    p_budget.add_argument("--ram", type=float, default=47.2)
    p_budget.add_argument("--ram-available", type=float, default=None)
    p_budget.add_argument("--mtp", action="store_true", help="include the 8 GB MTP/DSpark set")
    p_budget.add_argument("--expert-cache-gb", type=float, default=20.0,
                          help="recommended default expert-cache size (0 = computed max)")
    p_prefill = sub.add_parser("prefill")
    p_prefill.add_argument("--tokens", default="1024,8192,32768",
                           help="comma separated prompt lengths")
    p_prefill.add_argument("--slots", type=int, default=24)
    p_prefill.add_argument("--block", type=int, default=1024)
    p_prefill.add_argument("--attn-resident", type=int, default=34)
    p_prefill.add_argument("--bandwidth", type=float, default=8.8)
    p_vram = sub.add_parser("vram")
    p_vram.add_argument("--vram", type=float, default=8.0)
    p_vram.add_argument("--slots", type=int, default=24)
    p_vram.add_argument("--attn-mode", default="auto")
    p_vram.add_argument("--head-gb", type=float, default=0.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "sim":
        print(format_sim_table(run_simulation_suite(steps=args.steps)))
        return 0
    if args.cmd == "prefill":
        for tok in [int(x) for x in args.tokens.split(",") if x.strip()]:
            planner = PrefillPlanner(ModelGeometry(), tok, slots=args.slots, block=args.block,
                                     attn_resident_layers=args.attn_resident,
                                     bandwidth_gbs=args.bandwidth)
            print(f"=== layer-major prefill: {tok} tokens, slots={args.slots}, block={args.block} ===")
            print(planner.plan().table())
            print()
        return 0
    if args.cmd == "vram":
        plan = VramResidencyPlan.plan(vram_gb=args.vram, slots=args.slots,
                                      attn_mode=args.attn_mode, head_gb=args.head_gb)
        print(plan.table())
        print()
        print(plan.attn.table())
        print(plan.slots.table())
        return 0
    if args.cmd == "budget":
        cap = args.expert_cache_gb if args.expert_cache_gb > 0 else None
        b = TierBudget.plan(vram_gb=args.vram, ram_gb=args.ram,
                            ram_available_gb=args.ram_available, include_mtp=args.mtp,
                            target_expert_cache_gb=cap)
        print(b.table())
        print()
        for io in b.io_scenarios():
            print(f"hit={io.hit_rate:4.1%}  read/token={io.read_bytes_per_token/1e9:5.2f} GB  "
                  f"NVMe={io.nvme_ms:6.0f} ms  PCIe={io.pcie_ms:5.0f} ms  ~{io.tokens_per_second:4.2f} tok/s")
        return 0
    return 0 if selftest() else 1


if __name__ == "__main__":
    raise SystemExit(main())
