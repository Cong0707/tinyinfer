"""Tensor-parallel sharding invariants (no distributed process needed).

The interesting part is that a shard of the output dimension must not straddle a
quantisation block: fp8 blocks are 32 (dequant) and fp4 blocks are 32, so a rank's slice
of the weights and its slice of the scales have to line up.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.quant import dequant_fp4_block, dequant_fp8_block  # noqa: E402
from tinyinfer.tp import Shard, shard_linear  # noqa: E402


def test_columns_cover_the_whole_dimension():
    assert Shard(0, 2).columns(1280) == (0, 640)
    assert Shard(1, 2).columns(1280) == (640, 1280)
    ranges = [Shard(i, 3).columns(1280) for i in range(3)]
    assert ranges[0][0] == 0 and ranges[-1][1] == 1280
    for (lo, hi), (nlo, _) in zip(ranges, ranges[1:]):
        assert hi == nlo
        assert (hi - lo) % 128 == 0


def test_fp8_out_shard_keeps_blocks_aligned():
    torch.manual_seed(0)
    out, inn = 1280, 512
    w = torch.randn(out, inn).to(torch.float8_e4m3fn)
    s = torch.full((out // 32, inn // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    lin = SimpleNamespace(weight=w, scale=s)
    full = dequant_fp8_block(w, s)
    for rank in range(3):
        sh = shard_linear(lin, Shard(rank, 3), axis="out")
        lo, hi = Shard(rank, 3).columns(out)
        assert sh.weight.shape[0] == hi - lo
        # (hi - lo) must be a multiple of the 32-row block or the scales would shift
        assert (hi - lo) % 32 == 0
        part = dequant_fp8_block(sh.weight, sh.scale)
        assert torch.allclose(part, full[lo:hi], atol=1e-3)


def test_fp4_out_shard_keeps_blocks_aligned():
    torch.manual_seed(1)
    out, inn = 256, 128
    # packed e2m1: two values per byte
    packed = torch.randint(0, 255, (out, inn // 2), dtype=torch.uint8)
    w = packed.view(torch.float4_e2m1fn_x2)
    s = torch.full((out, inn // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    lin = SimpleNamespace(weight=w, scale=s)
    for rank in range(2):
        sh = shard_linear(lin, Shard(rank, 2), axis="out")
        lo, hi = Shard(rank, 2).columns(out)
        assert (hi - lo) % 32 == 0
        part = dequant_fp4_block(sh.weight.contiguous(), sh.scale)
        full = dequant_fp4_block(w.contiguous(), s)
        assert torch.allclose(part, full[lo:hi])


def test_all_reduce_is_identity_without_distributed():
    from tinyinfer.tp import all_reduce_sum

    x = torch.arange(6, dtype=torch.float32)
    assert torch.equal(all_reduce_sum(x.clone()), x)
