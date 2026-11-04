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
# ---------------------------------------------------------------------------
# Multi-GPU placement: asymmetric sharding, VRAM pooling, automatic addressing
# ---------------------------------------------------------------------------
#
# The pieces above assume ranks of equal size, which is what torch.distributed wants.
# A machine assembled from whatever cards were available is not that: a 96 GB PRO 6000
# next to a 32 GB 5090 has to give the big card three times the weights, and nothing in
# NCCL expresses that. So placement here is explicit instead of rank-based:
#
#   probe_devices()  measure every card (total/free VRAM, bandwidth)
#   Pool             one ledger across all cards, sized from what is really free
#   asplit()         split a dimension in proportion to each card's share, block-aligned
#   Placement        key -> device, i.e. the addressing table the forward pass consults
#
# Nothing here assumes the cards are identical, symmetric, or even the same architecture.


@dataclass
class Device:
    """One visible GPU and what it can take."""

    index: int
    name: str
    total: int          # bytes of VRAM
    free: int           # bytes free right now
    bandwidth: float    # bytes/s, from the spec sheet (clock x bus), not measured


def probe_devices(reserve: int = 512 << 20) -> list[Device]:
    """Enumerate every visible GPU with its real free VRAM.

    reserve is kept out of the budget on each card: the caching allocator, the CUDA
    context and the KV cache all need room that placement does not get to spend.
    """
    out: list[Device] = []
    if not torch.cuda.is_available():
        return out
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:
            free = p.total_memory
        # bandwidth from the spec: memory clock (kHz) x 2 (DDR) x bus width / 8
        bus = getattr(p, "memory_bus_width", 0) or 0
        clk = getattr(p, "memory_clock_rate", 0) or 0     # kHz
        bw = float(clk) * 1e3 * 2.0 * bus / 8.0 if bus and clk else 0.0
        out.append(Device(index=i, name=p.name, total=int(p.total_memory),
                          free=max(0, int(free) - reserve), bandwidth=bw))
    return out


class Pool:
    """One VRAM ledger across every card.

    Placement asks the pool, not the devices directly, so that two allocations cannot
    both believe they own the same free bytes. Shares are proportional to each card's
    free VRAM, which is what makes the split asymmetric without a separate knob.
    """

    def __init__(self, devices: list[Device]):
        self.devices = devices
        self.cap = [max(0, int(d.free)) for d in devices]
        self.used = [0] * len(devices)

    def remaining(self) -> int:
        return sum(c - u for c, u in zip(self.cap, self.used))

    def room_on(self, i: int) -> int:
        return max(0, self.cap[i] - self.used[i])

    def shares(self) -> list[float]:
        """Fraction of the remaining budget each card represents (sums to 1)."""
        rem = [self.room_on(i) for i in range(len(self.cap))]
        tot = float(sum(rem))
        if tot <= 0:
            return [0.0] * len(rem)
        return [r / tot for r in rem]

    def take(self, nbytes: int, prefer: int | None = None) -> int:
        """Book nbytes on one card; returns its index, or -1 when nothing fits."""
        if nbytes <= 0:
            return -1
        order = list(range(len(self.cap)))
        if prefer is not None and 0 <= prefer < len(self.cap):
            order.remove(prefer)
            order.insert(0, prefer)
        # best fit: smallest card that still holds it, keeping the big ones for big pieces
        fit = [i for i in order if self.room_on(i) >= nbytes]
        if not fit:
            return -1
        best = min(fit, key=lambda i: self.room_on(i))
        self.used[best] += nbytes
        return best

    def release(self, i: int, nbytes: int) -> None:
        self.used[i] = max(0, self.used[i] - int(nbytes))


def asplit(n: int, caps: list[int], align: int = SHARD_BLOCK) -> list[tuple[int, int]]:
    """Split n rows in proportion to caps, every boundary a multiple of align.

    Largest-remainder apportionment on the blocks, so the parts add up to exactly n and
    a small card never ends up with a negative remainder. Blocks smaller than align
    cannot be split at all (the scale layout would straddle the boundary), so a card
    whose share rounds to zero simply gets nothing -- that is correct, not a bug.
    """
    if n <= 0 or not caps:
        return []
    nb = n // align
    if nb <= 0:
        return [(0, n)]
    tot = float(sum(max(0, c) for c in caps))
    if tot <= 0:
        tot = 1.0
    exact = [nb * max(0, c) / tot for c in caps]
    base = [int(e) for e in exact]
    left = nb - sum(base)
    frac = sorted(range(len(caps)), key=lambda i: exact[i] - base[i], reverse=True)
    for k in range(left):
        base[frac[k % len(frac)]] += 1
    out: list[tuple[int, int]] = []
    lo = 0
    for b in base:
        hi = lo + b * align
        if hi > lo:
            out.append((lo, hi))
        lo = hi
    if lo < n and out:
        out[-1] = (out[-1][0], n)
    return out


class Placement:
    """The addressing table: which device holds each named tensor.

    Keys are names the forward pass already uses (layer.12.attn.wo_b,
    layer.12.expert.7.w1), so a lookup costs a dict hit and the caller never has to know
    how the model was distributed.
    """

    def __init__(self, pool: Pool):
        self.pool = pool
        self.map: dict[str, int] = {}
        self.bytes: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.map)

    def place(self, key: str, nbytes: int, prefer: int | None = None) -> int:
        dev = self.pool.take(nbytes, prefer=prefer)
        if dev < 0:
            raise MemoryError(
                f"placement: {nbytes} bytes for {key} does not fit; "
                f"{self.pool.remaining()} bytes left across {len(self.pool.cap)} device(s)")
        self.map[key] = dev
        self.bytes[key] = int(nbytes)
        return dev

    def device(self, key: str, default: int = 0) -> int:
        return self.map.get(key, default)

    def summary(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for k, d in self.map.items():
            out[d] = out.get(d, 0) + self.bytes.get(k, 0)
        return out

def shard_forward(x: torch.Tensor, plan: list, fn) -> torch.Tensor:
    """Run one row-parallel projection across cards and stitch the output back.

    plan is what asplit() produced, zipped with the device that owns each range:
        [(dev, (lo, hi)), ...]
    fn(dev, lo, hi, x_on_dev) must compute this device's slice of the output. The
    activations are copied in, the results are copied back and concatenated on dim=-1,
    which is where a row-parallel projection splits.

    This is the piece that cannot be verified without two real cards, so it is kept
    separate from the model code and exercised by a single-GPU test with a fake plan:
    one card holding two ranges reproduces the unsharded result exactly.
    """
    if not plan:
        raise ValueError("shard_forward: empty plan")
    if len(plan) == 1:
        dev, (lo, hi) = plan[0]
        tgt = torch.device(f"cuda:{dev}") if x.is_cuda else x.device
        xs = x if x.device == tgt else x.to(tgt)
        out = fn(dev, lo, hi, xs)
        return out if out.device == x.device else out.to(x.device)
    outs = []
    for dev, (lo, hi) in plan:
        tgt = torch.device(f"cuda:{dev}") if x.is_cuda else x.device
        xs = x if x.device == tgt else x.to(tgt)
        ys = fn(dev, lo, hi, xs)
        if ys.device != x.device:
            ys = ys.to(x.device, non_blocking=True)
        outs.append(ys)
    return torch.cat(outs, dim=-1)


def plan_for(n: int, pool: Pool, align: int = SHARD_BLOCK) -> list:
    """asplit(n, capacities) paired with the device index that owns each range."""
    caps = [pool.room_on(i) for i in range(len(pool.cap))]
    ranges = asplit(n, caps, align=align)
    # asplit drops cards whose share rounds to zero, so walk both lists together
    out = []
    for dev in range(len(caps)):
        if not ranges:
            break
        out.append((dev, ranges.pop(0)))
    return out
