"""Tensor-parallel sharding (port of Flare's same-process TP).

Flare runs each tensor-parallel rank as a thread of one process with its own CUDA
context and a VMM arena it maps into every rank; peer buffers keep identical
addresses from preparation through serving. tinyinfer shards with plain torch tensors
and one process, so this module provides the two pieces that matter for a first
working TP path:

* column/row shard helpers for the quantized projections we already hold
  (`QuantLinear` fp8 blocks, fp4 expert weights), which keep the per-block scale
  layout intact because a shard of the output dimension shards whole blocks;
* an all-reduce that uses `torch.distributed` (NCCL/NCCL-on-PCIe) when it is
  initialized and falls back to a plain sum over device-local copies otherwise, so a
  single-rank run behaves exactly like today.

Nothing here is on by default: `TierConfig.tp_ranks == 1` keeps every code path
unchanged, and a rank count above one is only accepted when `torch.distributed` is
available, since a silent single-rank fallback would produce wrong numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# fp8 GEMM blocks are 128x128 and the fp4 expert kernels use 32-wide row segments, so a
# shard boundary that is not a multiple of 128 makes one rank own half a block and the
# block scale it reads no longer belongs to its rows. A ceil() split of 1280 over three
# ranks gives 427 rows, i.e. 13.34 blocks: that is exactly the bug this alignment removes.
SHARD_BLOCK = 128


@dataclass(frozen=True)
class Shard:
    rank: int
    world: int

    def _range(self, n: int) -> tuple[int, int]:
        per = (n + self.world - 1) // self.world
        per = ((per + SHARD_BLOCK - 1) // SHARD_BLOCK) * SHARD_BLOCK
        lo = min(self.rank * per, n)
        hi = n if self.rank == self.world - 1 else min(n, lo + per)
        return lo, hi

    def columns(self, out: int) -> tuple[int, int]:
        """Output-dimension range held by this rank (row-parallel linear)."""
        return self._range(out)

    def rows(self, inn: int) -> tuple[int, int]:
        """Input-dimension range held by this rank (column-parallel linear)."""
        return self._range(inn)


def shard_linear(linear, shard: Shard, *, axis: str = "out"):
    """Slice a quantized linear's weights/scales for one rank.

    `axis='out'` shards the output dimension (weights [out, in] -> rows), which is what
    a row-parallel projection needs; the block scales follow because both fp8 (128x128)
    and fp4 (32x32) blocks never straddle the shard boundary when the dimension is a
    multiple of the block size -- callers must ensure that and we assert it.
    """
    w = getattr(linear, "weight", None)
    s = getattr(linear, "scale", None)
    if w is None:
        return linear
    dim = 0 if axis == "out" else 1
    n = w.shape[dim]
    lo, hi = shard.columns(n) if axis == "out" else shard.rows(n)
    out = type(linear).__new__(type(linear))
    out.__dict__.update(getattr(linear, "__dict__", {}))
    # Scale layouts differ between the two quantisations and a blind row slice is
    # wrong for both of the block-scaled cases:
    #   fp8 non-expert: scale [out/32, in/32]  (one scalar per 32x32 block)
    #   fp4 expert:     scale [out,    in/32]  (one scalar per 32-wide row segment)
    # Sharding the output dimension therefore moves the scale slice by out/32 for the
    # former and by out for the latter; sharding the input dimension moves both by /32.
    if axis == "out":
        out.weight = w[lo:hi].contiguous()
        if s is not None:
            if s.shape[0] == w.shape[0]:
                out.scale = s[lo:hi].contiguous()
            elif s.shape[0] > 0 and s.shape[0] * 32 == w.shape[0]:
                out.scale = s[lo // 32:(hi + 31) // 32].contiguous()
            else:
                out.scale = s[lo:hi].contiguous()
    else:
        out.weight = w[:, lo:hi].contiguous()
        if s is not None:
            if s.shape[1] * 32 == w.shape[1]:
                out.scale = s[:, lo // 32:(hi + 31) // 32].contiguous()
            else:
                out.scale = s[:, lo:hi].contiguous()
    return out


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across ranks when distributed is live; identity otherwise."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
    except Exception:
        pass
    return x


def available() -> bool:
    try:
        import torch.distributed as dist

        return bool(dist.is_available())
    except Exception:
        return False
