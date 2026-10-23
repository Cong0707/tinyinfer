"""The KV/state container contract, and one implementation per architecture.

The engine owns a session's state object (it snapshots it, saves it to .kvcache, and
hands it to the model), but it cannot know what a layer needs to keep: DeepSeek keeps a
compressed latent plus a window, Qwen keeps a KV window on 16 layers and a delta-rule
recurrent state on 48. So the interface is here and the storage is the model's.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class KvState(Protocol):
    """What a model writes its per-step state into."""

    def layer(self, layer_id: int) -> Any:
        """Per-layer state holder; its attributes are the model's business."""

    def update(self, keys: torch.Tensor, values: torch.Tensor, layer_id: int,
               kwargs: dict | None = None):
        """Write this step's K/V at the position in kwargs["cache_position"]."""

    def reset(self) -> None:
        """Drop all content; the object stays usable and keeps its capacity."""

    def snapshot(self) -> dict:
        """Everything needed to restore this state later (host tensors)."""

    def restore(self, snap: dict) -> None:
        """Inverse of snapshot()."""


class _Slot:
    """One layer's keep-state: a KV window and/or a recurrent tensor."""

    __slots__ = ("keys", "values", "recurrent", "conv", "written")

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.recurrent: torch.Tensor | None = None
        self.conv: torch.Tensor | None = None
        self.written = 0


class QwenKv:
    """KV windows for the full-attention layers, recurrent state for the linear ones."""

    def __init__(self, cfg: dict, capacity: int, dtype=torch.bfloat16,
                 device: str = "cuda") -> None:
        self.cfg = cfg
        self.capacity = int(capacity)
        self.dtype = dtype
        self.device = device
        self.kinds = list(cfg["layer_types"])
        self._slots: dict[int, _Slot] = {}

    def layer(self, layer_id: int) -> _Slot:
        slot = self._slots.get(layer_id)
        if slot is None:
            slot = self._slots[layer_id] = _Slot()
        return slot

    def _alloc_kv(self, slot: _Slot, layer_id: int, n: int) -> None:
        nkv = int(self.cfg["num_key_value_heads"])
        hd = int(self.cfg["head_dim"])
        if slot.keys is None:
            slot.keys = torch.zeros(1, nkv, self.capacity, hd, dtype=self.dtype,
                                    device=self.device)
            slot.values = torch.zeros(1, nkv, self.capacity, hd, dtype=self.dtype,
                                      device=self.device)

    def update(self, keys, values, layer_id, kwargs=None):
        slot = self.layer(layer_id)
        pos = 0
        if kwargs and kwargs.get("cache_position") is not None:
            pos = int(kwargs["cache_position"].flatten()[0])
        n = keys.shape[-2]
        if pos + n > self.capacity:
            # grow instead of writing past the end: index_copy_ past capacity trips a
            # device-side assert that takes the whole CUDA context with it
            self.capacity = max(pos + n, self.capacity * 2)
            grown_k = torch.zeros(1, keys.shape[1], self.capacity, keys.shape[-1],
                                  dtype=self.dtype, device=self.device)
            grown_v = torch.zeros_like(grown_k)
            if slot.keys is not None:
                grown_k[:, :, :slot.keys.shape[2]] = slot.keys
                grown_v[:, :, :slot.values.shape[2]] = slot.values
            slot.keys, slot.values = grown_k, grown_v
        elif slot.keys is None:
            self._alloc_kv(slot, layer_id, n)
        slot.keys[:, :, pos:pos + n].copy_(keys)
        slot.values[:, :, pos:pos + n].copy_(values)
        slot.written = max(slot.written, pos + n)
        return slot.keys[:, :, :pos + n], slot.values[:, :, :pos + n]

    # the linear-attention layers keep their own state through these
    def set_recurrent(self, layer_id: int, state: torch.Tensor) -> None:
        slot = self.layer(layer_id)
        slot.recurrent = state

    def reset(self) -> None:
        for slot in self._slots.values():
            slot.keys = slot.values = slot.recurrent = slot.conv = None
            slot.written = 0

    def snapshot(self) -> dict:
        out = {}
        for layer_id, slot in self._slots.items():
            out[layer_id] = {
                "keys": None if slot.keys is None else slot.keys.detach().cpu(),
                "values": None if slot.values is None else slot.values.detach().cpu(),
                "recurrent": (None if slot.recurrent is None
                              else slot.recurrent.detach().cpu()),
                "conv": None if slot.conv is None else slot.conv.detach().cpu(),
                "written": slot.written,
            }
        return out

    def restore(self, snap: dict) -> None:
        self.reset()
        for layer_id, entry in snap.items():
            slot = self.layer(int(layer_id))
            for name in ("keys", "values", "recurrent", "conv"):
                value = entry.get(name)
                if value is not None:
                    setattr(slot, name, value.to(self.device))
            slot.written = int(entry.get("written", 0))