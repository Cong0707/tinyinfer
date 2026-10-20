"""Reference dequantizers for the HF checkpoint's fp8/fp4 block formats.

The released weights store:
  - dense/attention weights as FP8 E4M3 with one E8M0 (ue8m0) scale per 32x32 block;
  - routed experts as FP4 E2M1 packed two-per-byte with one E8M0 scale per 32
    elements along K (one scale row per expert output row).
This module turns them back into plain bf16/fp32 tensors for PyTorch fallback paths.
"""
from __future__ import annotations

import torch

# E2M1 magnitudes by 3-bit code (0..7); sign is the high nibble bit.
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_TABLE = torch.cat([E2M1_VALUES, -E2M1_VALUES])  # 16 entries, code 8..15 == negatives


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 scales (stored as uint8 bit patterns) to float32 powers of two."""
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.to(torch.float32)
    bits = scale.view(torch.uint8).to(torch.int32)
    # E8M0: value = 2^(exp - 127); 0xFF is NaN, 0x00 is 2^-127.
    return torch.pow(2.0, (bits - 127).to(torch.float32))


def fp8_block_scale_to_dense(scale: torch.Tensor, out_features: int, in_features: int, block: int = 32) -> torch.Tensor:
    """Expand [out/32, in/32] scales without materializing a repeat_interleave."""
    s = e8m0_to_float(scale)  # [bo, bi]
    bo, bi = s.shape
    dense = s[:, None, :, None].expand(bo, block, bi, block)
    dense = dense.reshape(bo * block, bi * block)
    return dense[:out_features, :in_features]


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP8 E4M3 weight with 32x32 block scales -> bf16 tensor [out, in].

    On CUDA this dispatches to a single fused tilelang kernel (about 4x faster than the
    torch elementwise path), which matters because wo_a is dequantized every token.
    """
    if weight.is_cuda and block == 32:
        try:
            from .dequant_kernel import dequant_fp8
            return dequant_fp8(weight, scale)
        except Exception:
            pass
    out_features, in_features = weight.shape
    bo, bi = out_features // block, in_features // block
    s = e8m0_to_float(scale)[:bo, :bi]
    w4 = weight.to(torch.float32).view(bo, block, bi, block)
    out = (w4 * s[:, None, :, None]).reshape(out_features, in_features)
    return out.to(torch.bfloat16)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """[out, in//2] uint8 (two E2M1 values per byte) -> [out*2?] float32 in column order.

    HF stores each byte as (even column in low nibble, odd column in high nibble); the
    returned tensor is laid out [out, in].
    """
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)
    table = E2M1_TABLE.to(packed.device)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=torch.float32, device=packed.device)
    out[..., 0::2] = table[lo.long()]
    out[..., 1::2] = table[hi.long()]
    return out


def dequant_fp4_block(packed: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP4 E2M1 experts with 1x32 (K) block scales -> bf16 tensor [out, in]."""
    values = unpack_fp4(packed)
    out_features, in_features = values.shape
    bi = in_features // block
    s = e8m0_to_float(scale)[:, :bi]
    v4 = values.view(out_features, bi, block)
    out = (v4 * s[:, :, None]).reshape(out_features, in_features)
    return out.to(torch.bfloat16)
