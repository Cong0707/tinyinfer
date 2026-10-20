"""Sanity tests for fp8/fp4 block dequantizers."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.quant import E2M1_TABLE, dequant_fp4_block, dequant_fp8_block, e8m0_to_float, unpack_fp4


def test_e8m0():
    # E8M0: value = 2^(exp-127); byte 127 -> 1.0, 128 -> 2.0, 126 -> 0.5
    bits = torch.tensor([126, 127, 128, 130], dtype=torch.uint8)
    decoded = e8m0_to_float(bits)
    assert torch.allclose(decoded, torch.tensor([0.5, 1.0, 2.0, 8.0])), decoded
    print("e8m0 ok")


def test_fp8_block():
    out_features, in_features = 64, 96
    w = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
    s = torch.tensor([[127, 128], [129, 127]], dtype=torch.uint8)  # ceil(64/32)=2, ceil(96/32)=3 -> should be 2x3
    # make scale 2 x 3 with known powers
    s = torch.tensor([[127, 128, 126], [127, 127, 127]], dtype=torch.uint8)
    out = dequant_fp8_block(w, s)
    assert out.shape == (out_features, in_features)
    # first 32x32 block scaled by 1.0, second column-block by 2.0, third by 0.5
    assert torch.allclose(out[:32, :32].float(), w[:32, :32].float())
    assert torch.allclose(out[:32, 32:64].float(), w[:32, 32:64].float() * 2.0)
    assert torch.allclose(out[:32, 64:96].float(), w[:32, 64:96].float() * 0.5)
    print("fp8 block dequant ok")


def test_fp4_pack_order():
    # packed byte: low nibble = even column, high nibble = odd column
    # codes: 0 -> +0, 1 -> +0.5, 2 -> +1, 9 (0x9) -> -0.5, 10 -> -1
    packed = torch.tensor([[0x10 | 0x2, 0x9 | 0x0]], dtype=torch.uint8)  # [0x12, 0x09]
    vals = unpack_fp4(packed)
    expect = torch.tensor([[1.0, 0.5, 0.0, 0.0]])  # low=0x2 -> 1.0? code2 = 1.0; high=0x1 -> 0.5; code 0x9 -> -0.5, code0 -> 0
    # note: 0x9 low nibble = 9 -> -0.5; 0x0 high nibble = 0 -> 0
    expect = torch.tensor([[1.0, 0.5, -0.5, 0.0]])
    assert torch.allclose(vals, expect), vals
    print("fp4 pack order ok")


def test_fp4_block_scale():
    # one output row, K=64 -> packed 32 bytes, scale per 32 -> 2 blocks
    packed = torch.full((2, 32), 0x11, dtype=torch.uint8)  # every value = 0.5
    scale = torch.tensor([[127, 128], [127, 127]], dtype=torch.uint8)  # [2,2]: (1.0,2.0),(1.0,1.0)
    out = dequant_fp4_block(packed, scale)
    assert out.shape == (2, 64)
    assert torch.allclose(out[0, :32].float(), torch.full((32,), 0.5))
    assert torch.allclose(out[0, 32:].float(), torch.full((32,), 1.0))
    assert torch.allclose(out[1].float(), torch.full((64,), 0.5))
    print("fp4 block dequant ok")


if __name__ == "__main__":
    test_e8m0()
    test_fp8_block()
    test_fp4_pack_order()
    test_fp4_block_scale()
    print("ALL OK")
