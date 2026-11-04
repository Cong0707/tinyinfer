"""Sequences: the unit of KV ownership and of admission into a batch.

The engine's own generation loop assumes one conversation exists at a time: every
attention layer holds exactly one window buffer and one compressed-KV cache, keyed by
nothing. That is the single reason concurrency is impossible today -- two requests
would write into the same tensors and each would decode the other's history.

This module makes the KV a first-class, owned resource:

  Sequence       one conversation: tokens so far, its own attention state, its slot
  SequenceGroup  the sequences sharing one admission decision (speculative decoding
                 will put a target and its drafts here; plain serving has one member)
  KVPool         hands out slots and reclaims them, so a finished conversation frees
                 VRAM for the next one instead of leaking it until restart

Sizing, from the checkpoint (sliding_window 128, head_dim 512, 40 layers):

  window KV   128 x 512 x 2 B = 128 KB per layer  -> 5 MB per sequence for all layers
  compressed  one group per `ratio` tokens, and a group is head_dim wide too; at
              ratio 1 a 4K-token conversation needs 4096 x 512 x 2 B = 4 MB per
              compressed layer, 38 of them = 152 MB

So the window part is negligible and the compressed part is what the pool has to
budget. Nothing here allocates for max_position_embeddings (1M): that would be 1 GB
per layer per sequence, which is why the engine caps the run at a few thousand groups
and grows on demand.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch


_IDS = itertools.count()


@dataclass
class Sequence:
    """One conversation and the attention state it owns."""

    prompt: list[int]
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stop_ids: tuple = ()
    id: int = field(default_factory=lambda: next(_IDS))
    # produced tokens, in order
    out: list[int] = field(default_factory=list)
    # KV slot handed out by the pool; None until admitted
    slot: int | None = None
    done: bool = False
    # what the engine needs to restore this sequence's attention into a shared batch
    state: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Tokens consumed so far (prompt + produced, minus the one not yet fed)."""
        return len(self.prompt) + len(self.out)

    @property
    def next_pos(self) -> int:
        """Absolute position of the next token this sequence needs."""
        return self.length - 1

    @property
    def n_out(self) -> int:
        return len(self.out)


@dataclass
class SequenceGroup:
    """Sequences admitted together (one member for plain serving)."""

    members: list[Sequence] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return all(s.done or s.n_out >= s.max_new_tokens for s in self.members)


class KVPool:
    """Fixed number of sequence slots; each slot owns a full set of KV buffers.

    Slots are reuse-based rather than grown: the engine's buffers are views into the
    slot's storage, so a slot handed to a new sequence must be cleared of the previous
    one's history or the new conversation starts with someone else's tokens in context.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._free = list(range(self.capacity))
        self._owner: dict[int, int] = {}       # slot -> sequence id

    def acquire(self, seq: Sequence) -> int | None:
        if not self._free:
            return None
        slot = self._free.pop(0)
        self._owner[slot] = seq.id
        seq.slot = slot
        seq.state.clear()                   # a recycled slot carries no history
        return slot

    def release(self, seq: Sequence) -> None:
        if seq.slot is None:
            return
        slot = seq.slot
        if self._owner.get(slot) == seq.id:
            del self._owner[slot]
            self._free.append(slot)
        seq.slot = None

    @property
    def free(self) -> int:
        return len(self._free)

    @property
    def used(self) -> int:
        return self.capacity - len(self._free)

    def stats(self) -> dict:
        return {'capacity': self.capacity, 'used': self.used, 'free': self.free}
