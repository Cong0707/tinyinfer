"""Optimization tiers (0..3), each a coherent bundle of switches.

Positioning (user-defined):
  0 CONSERVATIVE  plain PyTorch reference path -- framework-average speed/requirements,
                  bf16 dequant, no tilelang, no fp8/fp4 kernels, no speculation.
  1 BALANCED      official optimizations proven at production scale: tilelang
                  sparse_attn (official kernel.py), fp8/fp4 block GEMM, quantized weights.
  2 AGGRESSIVE    OUR engine's optimizations (the main working target): tiered streaming,
                  expert RAM/VRAM caches, gated prefetch, MTP speculation, hardware-adaptive
                  resource planning, allocator tuning. Everything here is validated on this
                  laptop; keep it production-stable.
  3 EXTREME       approximations to chase the last drop of speed: top-k reduction, low-rank
                  output projection, fast math, KV4, layer skipping. Each item must be
                  validated; stable ones get promoted down into tier 2.
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
    if level == OptLevel.BALANCED:
        # official optimizations as shipped by the reference stack (tilelang + fp8/fp4)
        return TierConfig(
            level=level, use_tilelang_attn=True, use_fp8_gemm=True, use_fp4_gemm=True,
            keep_weights_quantized=True, expert_cache_gb=cache, kv_bits=16,
            prefetch=False, prefetch_k=0, mtp=False, head_on_cpu=False,
            approximations={}, notes="official optimizations: tilelang sparse_attn + fp8/fp4 GEMM")
    if level == OptLevel.AGGRESSIVE:
        # OUR engine optimizations (main production target): streaming + caches + prefetch + MTP
        return TierConfig(
            level=level, use_tilelang_attn=True, use_fp8_gemm=True, use_fp4_gemm=True,
            keep_weights_quantized=True, expert_cache_gb=cache, kv_bits=8,
            prefetch=True, prefetch_k=2, mtp=True, head_on_cpu=True,
            approximations={"allocator_expandable": True},
            notes="engine optimizations: tiered streaming, expert caches, gated prefetch, MTP")
    # EXTREME: lossy approximations for maximum speed; promote validated items down to tier 2
    return TierConfig(
        level=level, use_tilelang_attn=True, use_fp8_gemm=True, use_fp4_gemm=True,
        keep_weights_quantized=True, expert_cache_gb=cache, kv_bits=4,
        prefetch=True, prefetch_k=3, mtp=True, head_on_cpu=True,
        expert_topk=4,
        approximations={
            "allocator_expandable": True,
            "wo_low_rank": 0.75,
            "skip_layer_pattern": "every8",
            "fast_math": ["exp", "silu", "rsqrt", "sin", "cos"],
            "fp4_experts": True,
        },
        notes="approximations: top-4 experts, low-rank wo, KV4, fast math; promote stable items to tier 2")
