"""Optimization tiers: conservative -> balanced -> aggressive -> extreme.

The numbers come from our own measurements (docs/kernels.md, docs/loading.md).
Each tier is a coherent set of switches, not a menu: tiers[1] includes tiers[0].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class OptLevel(IntEnum):
    CONSERVATIVE = 0  # plain PyTorch reference; bf16 dequant; no tilelang, no speculation
    BALANCED = 1      # tilelang sparse_attn; ExpertStore streaming; head on CPU
    AGGRESSIVE = 2    # fp8/fp4 direct GEMM; KV quantization; gated prefetch; MTP speculation
    EXTREME = 3       # approximation/fitting: low-rank MMA, layer skipping, FFT-based linear attn


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
    approximations: dict = field(default_factory=dict)
    notes: str = ""


def plan(level: OptLevel = OptLevel.BALANCED, vram_gb: float = 8.0, ram_gb: float = 47.0) -> TierConfig:
    """Build a tier config. Expert cache leaves ~10-12GB RAM for non-expert weights/KV/OS."""
    cache = max(2.0, min(28.0, ram_gb - 24.0)) if level >= OptLevel.BALANCED else 4.0
    if level <= OptLevel.CONSERVATIVE:
        return TierConfig(
            level=level, use_tilelang_attn=False, use_fp8_gemm=False, use_fp4_gemm=False,
            keep_weights_quantized=False, expert_cache_gb=cache, kv_bits=16,
            prefetch=False, prefetch_k=0, mtp=False, head_on_cpu=False,
            approximations={}, notes="reference correctness path; slow but bit-comparable")
    if level == OptLevel.BALANCED:
        return TierConfig(
            level=level, use_tilelang_attn=True, use_fp8_gemm=False, use_fp4_gemm=False,
            keep_weights_quantized=True, expert_cache_gb=cache, kv_bits=16,
            prefetch=False, prefetch_k=0, mtp=False, head_on_cpu=True,
            approximations={}, notes="tilelang attention + tiered expert storage; no speculation")
    if level == OptLevel.AGGRESSIVE:
        return TierConfig(
            level=level, use_tilelang_attn=True, use_fp8_gemm=True, use_fp4_gemm=True,
            keep_weights_quantized=True, expert_cache_gb=cache, kv_bits=8,
            prefetch=True, prefetch_k=2, mtp=True, head_on_cpu=True,
            approximations={"kv_hadamard": True}, notes="fp8/fp4 kernels, KV8, gated prefetch, MTP")
    return TierConfig(
        level=level, use_tilelang_attn=True, use_fp8_gemm=True, use_fp4_gemm=True,
        keep_weights_quantized=True, expert_cache_gb=cache, kv_bits=4,
        prefetch=True, prefetch_k=3, mtp=True, head_on_cpu=True,
        approximations={
            "kv_hadamard": True,
            "wo_low_rank": 0.75,       # SVD-truncate grouped output projection
            "skip_layer_pattern": "every8",  # drop a layer's residual contribution periodically
            "fast_math": ["exp", "silu", "rsqrt", "sin", "cos"],  # curve fits for SFU ops
            "fft_linear_attn": False,  # experiment only: FFT token mixing (measured unusable)
            "fp4_experts": True,
        },
        notes="approximation/fitting regime: low-rank output proj, layer skipping, KV4; "
              "quality must be re-measured per prompt class")
