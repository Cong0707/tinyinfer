"""Expert parallelism: put different experts on different cards.

Tensor parallelism splits each weight matrix; expert parallelism splits the *set* of
experts and leaves every expert whole. For a mixture-of-experts model that is the
difference that matters: on this checkpoint a routed expert is 18.8 MB and there are
384 of them per layer, so a token touches 6 tensors it will never share with a
different token. TP keeps that read count identical on every card (each card still
reads its slice of all six), while EP divides it: with two ranks a card holds 192
experts and only reads the 3 of them that this token happened to route to.

Two pieces:

  route(experts, world)   which rank owns each expert id (contiguous blocks, so an
                          expert's blob stays with its neighbours on disk)
  reduce_partials(...)    every rank computed a disjoint subset of the experts, so the
                          layer output is the SUM of the ranks' partial outputs

The second piece is where a naive implementation goes wrong, and it went wrong here
before: moe.forward filtered `active` to the rank's own experts and returned what it
had. Nothing summed the ranks, so with world > 1 the answer was missing every expert
the other rank owned -- correct-looking, silently wrong.

Single-process, several devices: the ranks are CUDA devices in this process, not
torch.distributed processes, so the reduction is a device-to-device add through host
memory (or peer access when available). The interface is the same either way, which is
what lets one code path serve one card and eight.
"""
from __future__ import annotations

import torch


class ExpertRoute:
    """Which device owns which expert, for every layer."""

    def __init__(self, n_experts: int, devices: list[str], caps: list[int] | None = None):
        self.n_experts = max(0, int(n_experts))
        self.devices = list(devices) or ["cuda:0"]
        caps = list(caps) if caps else [1] * len(self.devices)
        self.world = len(self.devices)
        # Contiguous blocks, proportional to capacity: expert ids are laid out on disk in
        # order, so a contiguous range keeps each rank's reads sequential where a
        # round-robin assignment would make every rank skip every other blob.
        tot = float(sum(caps)) or 1.0
        counts = [int(round(self.n_experts * c / tot)) for c in caps]
        left = self.n_experts - sum(counts)
        if left:
            big = max(range(len(caps)), key=lambda i: caps[i])
            counts[big] += left
        self.owner: list[int] = []
        self.block: list[tuple[int, int]] = []
        lo = 0
        for rank, n in enumerate(counts):
            self.block.append((lo, lo + n))
            self.owner.extend([rank] * n)
            lo += n

    def rank_of(self, expert: int) -> int:
        if 0 <= expert < len(self.owner):
            return self.owner[expert]
        return 0

    def mine(self, experts, rank: int) -> list[int]:
        """The subset of `experts` this rank owns, preserving order."""
        return [e for e in experts if self.rank_of(e) == rank]

    def device_of(self, expert: int) -> str:
        return self.devices[self.rank_of(expert)]

    def summary(self) -> str:
        parts = [f"rank{r} {d}: experts [{lo}, {hi})"
                 for r, ((lo, hi), d) in enumerate(zip(self.block, self.devices))]
        return f"{self.n_experts} experts over {self.world} rank(s): " + "; ".join(parts)


def reduce_partials(partials: list, target_device=None) -> torch.Tensor:
    """Sum the ranks' partial layer outputs into one tensor.

    Each rank computed a disjoint set of experts, so their contributions to the same
    output rows have to be added -- concatenation would be wrong (it would double the
    row count) and taking any single one would silently drop the others.

    A single-element list is returned as-is: on one card there is nothing to add, and
    addition would be a wasted pass over the activation.
    """
    if not partials:
        raise ValueError("reduce_partials: no partials")
    if len(partials) == 1:
        p = partials[0]
        if target_device is not None and p.device != target_device:
            return p.to(target_device, non_blocking=True)
        return p
    acc = partials[0]
    for p in partials[1:]:
        if p.device != acc.device:
            p = p.to(acc.device, non_blocking=True)
        acc = acc + p
    if target_device is not None and acc.device != target_device:
        acc = acc.to(target_device, non_blocking=True)
    return acc


def load_balance(route: ExpertRoute, counts: list[int]) -> list[int]:
    """Re-assign experts to ranks to even out their observed load.

    Contiguous blocks keep disk reads sequential but can leave one rank with the hot
    experts. `counts` is how often each expert was actually routed; this returns a
    rank per expert minimising the busiest rank, using first-fit-decreasing over the
    block sizes as the capacity of each rank. Ties keep the original order, so a
    workload without a hot spot is left alone.
    """
    n = len(counts)
    if n == 0 or route.world <= 1:
        return [0] * n
    target = max(1, int(round(sum(counts) / float(route.world))))
    order = sorted(range(n), key=lambda i: -int(counts[i]))
    load = [0] * route.world
    out = [0] * n
    for e in order:
        best = min(range(route.world), key=lambda r: (load[r], r))
        out[e] = best
        load[best] += int(counts[e])
    return out
