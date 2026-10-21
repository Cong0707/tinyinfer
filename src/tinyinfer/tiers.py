"""Optimization tiers (0..3), each a coherent bundle of switches.

Positioning (user-defined):
  0 CONSERVATIVE  plain PyTorch reference path -- framework-average speed/requirements,
                  bf16 dequant, no tilelang, no fp8/fp4 kernels, no speculation.
  1 BALANCED      production-grade mechanisms already proven upstream, ported in:
                  official tilelang sparse_attn + fp8/fp4 block GEMM, and the NVIDIA
                  Flare mechanisms (CUDA graph capture, prepared capacity catalog,
                  DSpark feedback-window accumulation, async expert fetch, batched
                  scatter, contiguous slot arena, immutable prefix checkpoints, resume
                  sampling, tensor-parallel sharding). Everything here has a working
                  reference implementation upstream; we reproduce it, we do not invent.
  2 AGGRESSIVE    OUR engine's own features (the main working target): tiered layer
                  streaming, expert RAM/VRAM caches, gated prefetch, MTP scheduling,
                  hardware-adaptive resource planning, allocator tuning.
  3 EXTREME       approximations to chase the last drop of speed: top-k reduction,
                  low-rank output projection, fast math, KV4, layer skipping. Each item
                  must be validated; stable ones get promoted down into level 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class OptLevel(IntEnum):
    CONSERVATIVE = 0
    BALANCED = 1
    AGGRESSIVE = 2
    EXTREME = 3

@dataclass
class TierConfig:
    level: OptLevel
    use_tilelang_attn: bool
    use_fp8_gemm: bool
    use_fp4_gemm: bool
    keep_weights_quantized: bool
    expert_cache_gb: float
    kv_bits: int
    prefetch: bool
    prefetch_k: int
    mtp: bool
    head_on_cpu: bool
    expert_topk: int = 0  # 0 = checkpoint top-k (6); lower values trade quality for IO
    # ---- mechanisms ported from NVIDIA Flare (docs/flare_analysis.md) ---------------
    # Each is a coherent execution choice, not a debug switch: it is on for the tiers
    # where it has been validated and off below them.
    cuda_graph: bool = False          # capture the fully-resident decode step (tier 2+)
    graph_catalog: tuple = ()         # prepared prefill capacities / decode batches (tier 2+)
    mtp_feedback_window: bool = False  # accumulate the draft's 128-token KV window (tier 2+)
    async_expert_fetch: bool = False  # issue disk reads before serving cached experts
    batched_scatter: bool = False     # one index_add per expert batch instead of per expert
    slot_arena: bool = False          # one contiguous VRAM slot arena
    prefix_checkpoint: bool = False   # publish/restore immutable prefix state (disk or RAM)
    resume_sampling: bool = False     # re-sample from a saved hidden vector
    official_gemv: bool = False       # official block-scaled GEMV port (b12x algorithm)
    tp_ranks: int = 1                 # tensor-parallel ranks (same-process, tier 2+ on NVLink)
    approximations: dict = field(default_factory=dict)
    notes: str = ""


def plan(level: OptLevel = OptLevel.BALANCED, vram_gb: float = 8.0, ram_gb: float = 47.0) -> TierConfig:
    """Build a tier config. Expert cache leaves ~10-12GB RAM for non-expert weights/KV/OS."""
    cache = max(2.0, min(28.0, ram_gb - 24.0)) if level >= OptLevel.BALANCED else 4.0
    if level == OptLevel.CONSERVATIVE:
        # reference: bf16 weights, plain torch ops, no speculation -- correctness baseline
        return TierConfig(
            level=level, use_tilelang_attn=False, use_fp8_gemm=False, use_fp4_gemm=False,
            keep_weights_quantized=False, expert_cache_gb=cache, kv_bits=16,
            prefetch=False, prefetch_k=0, mtp=False, head_on_cpu=False,
            approximations={}, notes="reference: bf16 weights + plain torch ops (framework-average)")
        # tier 0 keeps every engine mechanism off: it is the correctness baseline
    if level == OptLevel.BALANCED:
        # official optimizations as shipped by the reference stack (tilelang + fp8/fp4)
        return TierConfig(
            level=level, use_tilelang_attn=True, use_fp8_gemm=True, use_fp4_gemm=True,
            keep_weights_quantized=True, expert_cache_gb=cache, kv_bits=16,
            prefetch=False, prefetch_k=0, mtp=False, head_on_cpu=False,
            # level 1 = production-proven mechanisms ported in (official kernels + Flare)
            cuda_graph=True, graph_catalog=(128, 256, 512, 1024, 2048),
            mtp_feedback_window=True, async_expert_fetch=True, batched_scatter=True,
            slot_arena=True, prefix_checkpoint=True, resume_sampling=True, tp_ranks=1,
            approximations={},
            notes="production-proven ports: official tilelang/fp8/fp4 + Flare graph, "
                  "window, checkpoints, slot arena")
    if level == OptLevel.AGGRESSIVE:
        # OUR engine optimizations (main production target): streaming + caches + prefetch + MTP
        return TierConfig(
            level=level, use_tilelang_attn=True, use_fp8_gemm=True, use_fp4_gemm=True,
            keep_weights_quantized=True, expert_cache_gb=cache, kv_bits=8,
            prefetch=True, prefetch_k=2, mtp=True, head_on_cpu=True,
            # level 2 = our engine's features on top of the ported mechanisms
            cuda_graph=True, graph_catalog=(128, 256, 512, 1024, 2048),
            mtp_feedback_window=True, async_expert_fetch=True, batched_scatter=True,
            slot_arena=True, prefix_checkpoint=True, resume_sampling=True,
            approximations={"allocator_expandable": True},
            notes="engine features: tiered streaming, expert caches, gated prefetch, MTP")
    # EXTREME: lossy approximations for maximum speed; promote validated items down to tier 2
    return TierConfig(
        level=level, use_tilelang_attn=True, use_fp8_gemm=True, use_fp4_gemm=True,
        keep_weights_quantized=True, expert_cache_gb=cache, kv_bits=4,
        prefetch=True, prefetch_k=3, mtp=True, head_on_cpu=True,
        expert_topk=4,
        cuda_graph=True, graph_catalog=(128, 256, 512, 1024, 2048),
        mtp_feedback_window=True, async_expert_fetch=True, batched_scatter=True,
        slot_arena=True, prefix_checkpoint=True, resume_sampling=True,
        approximations={
            "allocator_expandable": True,
            "wo_low_rank": 0.75,
            "skip_layer_pattern": "every8",
            "fast_math": ["exp", "silu", "rsqrt", "sin", "cos"],
            "fp4_experts": True,
        },
        notes="approximations: top-4 experts, low-rank wo, KV4, fast math; promote stable items to tier 2")