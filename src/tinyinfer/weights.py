"""Direct HF safetensors access for tinyinfer.

Reads tensor bytes with pread instead of keePing an mmap per shard: on Windows,
one mmap per 6.9GB shard piles up commit charge and multi-layer runs crashed
(after 7-8 layers) with access violations. pread copies into fresh host memory,
so only the tensors a block actually needs are resident.

Headers are parsed lazily and cached; a missing/incomplete shard raises a clear
error instead of faulting.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

INDEX_FILE = "model.safetensors.index.json"

_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}


class ShardStore:
    """Lazy, per-tensor access to a HF checkpoint directory (pread based)."""

    def __init__(self, model_dir: str | Path, *, max_cached_headers: int = 64):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / INDEX_FILE
        self.weight_map: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self._max_headers = max_cached_headers

    # -- discovery ---------------------------------------------------------
    def shard_of(self, name: str) -> str:
        return self.weight_map[name]

    def shard_files(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def missing(self, names) -> list[str]:
        return [n for n in names if n not in self.weight_map]

    # -- header / data access ---------------------------------------------
    def _header(self, shard: str) -> tuple[dict, int]:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        path = self.model_dir / shard
        if not path.exists():
            raise FileNotFoundError(f"shard missing (download incomplete): {shard}")
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors header: {shard}")
            size = struct.unpack("<Q", prefix)[0]
            if size > 200 * 1024 * 1024:
                raise ValueError(f"implausible header size {size} for {shard}")
            raw = f.read(size)
            if len(raw) != size:
                raise ValueError(f"truncated safetensors header: {shard}")
        header = json.loads(raw)
        info = (header, 8 + size)
        self._headers[shard] = info
        if len(self._headers) > self._max_headers:
            self._headers.pop(next(iter(self._headers)))
        return info

    def get(self, name: str) -> torch.Tensor:
        shard = self.shard_of(name)
        header, data_start = self._header(shard)
        entry = header.get(name)
        if entry is None:
            raise KeyError(f"tensor {name} not found in {shard}")
        start, nbytes = entry["data_offsets"][0], entry["data_offsets"][1] - entry["data_offsets"][0]
        dtype = _DTYPES.get(str(entry["dtype"]))
        if dtype is None:
            raise ValueError(f"unsupported dtype {entry['dtype']} for {name}")
        path = self.model_dir / shard
        with path.open("rb") as f:
            f.seek(data_start + start)
            raw = f.read(nbytes)
        if len(raw) != nbytes:
            raise ValueError(f"tensor {name} is truncated in {shard} (download incomplete)")
        flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return flat.view(dtype).reshape(tuple(entry["shape"]))

    def get_slice_info(self, name: str) -> tuple[tuple[int, ...], str]:
        shard = self.shard_of(name)
        header, _ = self._header(shard)
        entry = header[name]
        return tuple(entry["shape"]), str(entry["dtype"])

    def close(self) -> None:
        self._headers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
