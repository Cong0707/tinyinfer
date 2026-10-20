"""Direct HF safetensors access for tinyinfer.

Opens shards through safetensors' memory mapping and exposes individual tensors,
so weights can be read without conversion or a full resident copy.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

INDEX_FILE = "model.safetensors.index.json"


class ShardStore:
    """Lazy, per-tensor access to a HF checkpoint directory."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / INDEX_FILE
        self.weight_map: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self._handles: dict[str, object] = {}

    # -- discovery ---------------------------------------------------------
    def shard_of(self, name: str) -> str:
        return self.weight_map[name]

    def shard_files(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def missing(self, names) -> list[str]:
        return [n for n in names if n not in self.weight_map]

    # -- tensor access -----------------------------------------------------
    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
            self._handles[shard] = h
        return h

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self.shard_of(name)).get_tensor(name)

    def get_slice_info(self, name: str) -> tuple[tuple[int, ...], str]:
        shard = self.shard_of(name)
        view = self._handle(shard).get_slice(name)
        return tuple(view.get_shape()), view.get_dtype()

    def close(self) -> None:
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
