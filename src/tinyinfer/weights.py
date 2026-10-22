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

    def __init__(self, model_dir, *, max_cached_headers: int = 64):
        """One checkpoint, possibly split over several directories/disks.

        Same contract as loading.ShardIndex: the sources must describe the same model
        (identical weight_map where an index is present, identical shard sizes where a
        shard appears twice) or this raises. A directory that contributes no shard is
        rejected as a typo instead of being silently ignored.
        """
        raw = ([model_dir] if isinstance(model_dir, (str, Path))
               else [p for chunk in model_dir for p in str(chunk).split(";")])
        dirs: list[Path] = []
        for entry in raw:
            for part in str(entry).split(","):
                part = part.strip().strip('"')
                if part and Path(part).exists() and Path(part) not in dirs:
                    dirs.append(Path(part))
        if not dirs:
            raise FileNotFoundError(f"no model directory exists among {raw!r}")
        self.model_dirs = dirs
        self.model_dir = dirs[0]
        maps: list[tuple[Path, dict[str, str]]] = []
        for d in dirs:
            p = d / INDEX_FILE
            if p.is_file():
                maps.append((d, json.loads(p.read_text(encoding="utf-8"))["weight_map"]))
        if not maps:
            raise FileNotFoundError(f"{INDEX_FILE} not found under any of {[str(d) for d in dirs]}")
        base_dir, base_map = maps[0]
        for d, m in maps[1:]:
            if m != base_map:
                diff = sorted(k for k in set(base_map) | set(m) if base_map.get(k) != m.get(k))
                sample = diff[0] if diff else "?"
                raise ValueError("model sources are not the same checkpoint: "
                                 f"{len(diff)} tensors differ, e.g. {sample} -> "
                                 f"{base_map.get(sample)} vs {m.get(sample)}")
        self.weight_map: dict[str, str] = base_map
        self._shard_dir: dict[str, Path] = {}
        for shard in sorted(set(base_map.values())):
            found = [(d, (d / shard).stat().st_size) for d in dirs if (d / shard).is_file()]
            if not found:
                raise FileNotFoundError(f"shard {shard} missing from every source")
            if len({sz for _d, sz in found}) > 1:
                raise ValueError(f"shard {shard} differs between sources: refusing to mix checkpoints")
            self._shard_dir[shard] = found[0][0]
        idle = [d for d in dirs if d not in set(self._shard_dir.values())]
        if idle and len(dirs) > 1:
            raise ValueError("these model directories supplied no shards: "
                             + ", ".join(str(d) for d in idle))
        self._headers: dict[str, tuple[dict, int]] = {}
        self._max_headers = max_cached_headers

    def shard_path(self, shard: str) -> Path:
        d = self._shard_dir.get(shard)
        return (d / shard) if d is not None else (self.model_dir / shard)

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
        path = self.shard_path(shard)
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
        # When the engine is building its pinned pool, read the weights STRAIGHT into a
        # pool slice: the tensor then lives inside the pool, so the start-up build shows
        # no growth in Task Manager (the pool was already fully committed) instead of
        # adding ~17GB of private transients on top of it.
        pool = getattr(self, "build_pool", None)
        slot = pool.alloc(nbytes) if pool is not None else None
        if slot is not None:
            with path.open("rb") as f:
                f.seek(data_start + start)
                got = f.readinto(slot)
            if got != nbytes:
                raise ValueError(f"tensor {name} is truncated in {shard} (download incomplete)")
            flat = torch.frombuffer(slot, dtype=torch.uint8)
            return flat.view(dtype).reshape(tuple(entry["shape"]))
        with path.open("rb") as f:
            f.seek(data_start + start)
            raw = f.read(nbytes)
        if len(raw) != nbytes:
            raise ValueError(f"tensor {name} is truncated in {shard} (download incomplete)")
        flat = torch.frombuffer(raw, dtype=torch.uint8)
        return flat.view(dtype).reshape(tuple(entry["shape"]))

    def read_many(self, names: list[str], *, threads: int = 8) -> dict[str, torch.Tensor]:
        """Parallel tensor reads (thread pool over pread); warms the OS cache for get().

        Startup is dominated by per-tensor reads (one open+seek+read each); reading a
        layer's weights through this pool saturates the SSD instead of serializing.
        """
        from concurrent.futures import ThreadPoolExecutor

        if not names:
            return {}
        out: dict[str, torch.Tensor] = {}
        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            futures = [pool.submit(self.get, n) for n in names]
            for n, fut in zip(names, futures):
                out[n] = fut.result()
        return out

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
