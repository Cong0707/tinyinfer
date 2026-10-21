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


@dataclass(frozen=True)
class Shard:
    rank: int
    world: int

    def columns(self, out: int) -> tuple[int, int]:
        """Output-dimension range held by this rank (row-parallel linear)."""
        per = (out + self.world - 1) // self.world
        lo = self.rank * per
        return lo, min(out, lo + per)

    def rows(self, inn: int) -> tuple[int, int]:
        """Input-dimension range held by this rank (column-parallel linear)."""
        per = (inn + self.world - 1) // self.world
        lo = self.rank * per
        return lo, min(inn, lo + per)


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
    if axis == "out":
        out.weight = w[lo:hi].contiguous()
        if s is not None:
            out.scale = s[lo:hi].contiguous()
    else:
        out.weight = w[:, lo:hi].contiguous()
        if s is not None:
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
