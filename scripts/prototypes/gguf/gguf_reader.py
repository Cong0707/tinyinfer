"""Minimal, dependency-free GGUF header/index parser (prototype).

The parser reads only the GGUF metadata and tensor directory.  Tensor payloads
stay in the file: a GGUFTensor exposes a zero-copy memoryview for a byte range,
which is what an ExpertStore style loader needs for pread/O_DIRECT streaming.

Format reference: llama.cpp/ggml/include/gguf.h (header comment) and
gguf-py/gguf/gguf_reader.py.  This file is intentionally small and strict; it is
not a replacement for gguf-py.

Usage:
    python gguf_reader.py MODEL.gguf
    python gguf_reader.py MODEL.gguf --tensors
    python gguf_reader.py MODEL.gguf --tensor blk.0.ffn_gate_exps.weight
    python gguf_reader.py MODEL.gguf --dequant blk.0.ffn_gate_exps.weight
    python gguf_reader.py MODEL.gguf --json
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

try:
    from . import quants_ref
except ImportError:  # direct script execution
    import quants_ref  # type: ignore

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGUF_MAX_STRING_LENGTH = 1 << 30
GGUF_MAX_ARRAY_ELEMENTS = 1 << 30
GGML_MAX_DIMS = 4

# gguf_type enum (ggml/include/gguf.h)
VALUE_TYPES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32",
    5: "int32", 6: "float32", 7: "bool", 8: "string", 9: "array",
    10: "uint64", 11: "int64", 12: "float64",
}
_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}

TYPE_NAMES = quants_ref.TYPE_IDS
TENSOR_TYPE_NAMES = {tid: name for tid, name in TYPE_NAMES.items()}


class GGUFError(ValueError):
    """Raised when the file is not a valid GGUF or uses an unknown feature."""


@dataclass
class GGUFTensor:
    name: str
    dims: tuple[int, ...]          # GGUF order: ne0 (innermost) first
    type_id: int
    offset: int                    # relative to the data section
    data_offset: int               # absolute file offset
    n_elements: int
    n_bytes: int
    reader: "GGUFReader" = field(repr=False)

    @property
    def type_name(self) -> str:
        return TENSOR_TYPE_NAMES.get(self.type_id, f"type_{self.type_id}")

    @property
    def shape(self) -> tuple[int, ...]:
        """NumPy/torch logical shape: dims reversed (last dim is ne0)."""
        return tuple(reversed(self.dims))

    def raw(self) -> np.ndarray:
        """Zero-copy uint8 view of the payload."""
        return np.frombuffer(self.reader._mm, dtype=np.uint8,
                             count=self.n_bytes, offset=self.data_offset)

    def dequantize(self) -> np.ndarray:
        if self.type_id not in TENSOR_TYPE_NAMES:
            raise NotImplementedError(f"unknown ggml type id {self.type_id}")
        return quants_ref.dequantize_tensor(self.raw(), self.type_name, list(self.dims))

    def __str__(self) -> str:
        return (f"{self.name}: {self.type_name} dims={list(self.dims)} "
                f"nelements={self.n_elements} bytes={self.n_bytes} "
                f"offset={self.offset} (+{self.data_offset} in file)")


class GGUFReader:
    """Parse a GGUF metadata block and tensor directory.

    Parameters
    ----------
    path:
        GGUF file path.
    load_metadata_arrays:
        When False, arrays are only validated and counted; their values are
        discarded.  Useful for very large vocab arrays.
    """

    def __init__(self, path: str | os.PathLike[str], *, load_metadata_arrays: bool = True):
        self.path = Path(path)
        self._file: BinaryIO | None = open(self.path, "rb")
        self._size = os.fstat(self._file.fileno()).st_size
        if self._size < 24:
            raise GGUFError(f"{self.path}: file too small for a GGUF header")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._off = 0
        self.version = 0
        self.n_tensors = 0
        self.n_kv = 0
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self.data_offset = 0
        self.metadata: dict[str, Any] = {}
        self.metadata_types: dict[str, str] = {}
        self.tensors: list[GGUFTensor] = []
        self.tensors_by_name: dict[str, GGUFTensor] = {}
        self._load(load_metadata_arrays)

    # -- low level ---------------------------------------------------------
    def _need(self, n: int) -> None:
        if self._off + n > self._size:
            raise GGUFError(f"truncated file at offset {self._off} (need {n} bytes)")

    def _u32(self) -> int:
        self._need(4)
        v = struct.unpack_from("<I", self._mm, self._off)[0]
        self._off += 4
        return v

    def _u64(self) -> int:
        self._need(8)
        v = struct.unpack_from("<Q", self._mm, self._off)[0]
        self._off += 8
        return v

    def _string(self) -> str:
        n = self._u64()
        if n > GGUF_MAX_STRING_LENGTH:
            raise GGUFError(f"string length {n} exceeds limit")
        self._need(n)
        raw = self._mm[self._off:self._off + n]
        self._off += n
        return raw.decode("utf-8", errors="replace")

    def _value(self, vtype: int, *, load_arrays: bool) -> Any:
        fmt = _SCALAR_FMT.get(vtype)
        if fmt is not None:
            self._need(struct.calcsize(fmt))
            (v,) = struct.unpack_from(fmt, self._mm, self._off)
            self._off += struct.calcsize(fmt)
            return v
        if vtype == 8:  # string
            return self._string()
        if vtype == 9:  # array
            subtype = self._u32()
            count = self._u64()
            if count > GGUF_MAX_ARRAY_ELEMENTS:
                raise GGUFError(f"array count {count} exceeds limit")
            if subtype == 8:
                # Strings dominate real GGUF metadata (tokenizer vocab).
                values = [self._string() for _ in range(count)] if load_arrays else None
                if values is None:
                    for _ in range(count):
                        self._string()
                return values
            if subtype not in _SCALAR_FMT:
                raise GGUFError(f"unsupported array sub-type {subtype}")
            elem = struct.calcsize(_SCALAR_FMT[subtype])
            self._need(elem * count)
            if load_arrays:
                values = list(struct.unpack_from(f"<{count}{_SCALAR_FMT[subtype][1:]}",
                                                 self._mm, self._off))
            else:
                values = None
            self._off += elem * count
            return values
        raise GGUFError(f"unknown gguf value type {vtype}")

    # -- parse -------------------------------------------------------------
    def _load(self, load_metadata_arrays: bool) -> None:
        magic = self._u32()
        if magic != GGUF_MAGIC:
            raise GGUFError(f"{self.path}: bad magic 0x{magic:08x} (expected GGUF)")
        self.version = self._u32()
        if self.version not in (2, 3):
            raise GGUFError(f"{self.path}: unsupported GGUF version {self.version}")
        self.n_tensors = self._u64()
        self.n_kv = self._u64()
        if self.n_tensors > GGUF_MAX_ARRAY_ELEMENTS or self.n_kv > GGUF_MAX_ARRAY_ELEMENTS:
            raise GGUFError("tensor/KV count exceeds limit")

        for _ in range(self.n_kv):
            key = self._string()
            vtype = self._u32()
            if key in self.metadata:
                raise GGUFError(f"duplicate metadata key {key!r}")
            self.metadata[key] = self._value(vtype, load_arrays=load_metadata_arrays)
            self.metadata_types[key] = VALUE_TYPES.get(vtype, f"type_{vtype}")

        tensor_infos: list[tuple[str, tuple[int, ...], int, int]] = []
        seen_names: set[str] = set()
        for _ in range(self.n_tensors):
            name = self._string()
            if name in seen_names:
                raise GGUFError(f"duplicate tensor name {name!r}")
            seen_names.add(name)
            n_dims = self._u32()
            if n_dims > GGML_MAX_DIMS:
                raise GGUFError(f"{name}: {n_dims} dims exceeds GGML_MAX_DIMS")
            dims = tuple(self._u64() for _ in range(n_dims)) if n_dims else ()
            type_id = self._u32()
            offset = self._u64()
            tensor_infos.append((name, dims, type_id, offset))

        align = self.metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
        if not isinstance(align, int) or align <= 0 or (align & (align - 1)) != 0:
            raise GGUFError(f"invalid general.alignment {align!r}")
        self.alignment = align
        pad = self._off % self.alignment
        if pad:
            self._off += self.alignment - pad
        self.data_offset = self._off

        for name, dims, type_id, offset in tensor_infos:
            if type_id not in TENSOR_TYPE_NAMES:
                raise GGUFError(f"{name}: unknown ggml tensor type {type_id}")
            type_name = TENSOR_TYPE_NAMES[type_id]
            block, type_size = quants_ref.QUANT_SIZES[type_name]
            n_elements = 1
            for d in dims:
                n_elements *= int(d)
            if dims and dims[0] % block != 0:
                raise GGUFError(f"{name}: ne0={dims[0]} not a multiple of {type_name} block {block}")
            n_bytes = (n_elements // block) * type_size if dims else 0
            absolute = self.data_offset + offset
            if absolute + n_bytes > self._size:
                raise GGUFError(f"{name}: payload ends at {absolute + n_bytes} beyond file size {self._size}")
            t = GGUFTensor(name=name, dims=dims, type_id=type_id, offset=offset,
                           data_offset=absolute, n_elements=n_elements,
                           n_bytes=n_bytes, reader=self)
            self.tensors.append(t)
            self.tensors_by_name[name] = t

    # -- helpers -----------------------------------------------------------
    def get(self, name: str) -> GGUFTensor:
        try:
            return self.tensors_by_name[name]
        except KeyError as e:
            raise KeyError(f"tensor not found: {name}") from e

    def __enter__(self) -> "GGUFReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None  # type: ignore[assignment]
        if self._file is not None:
            self._file.close()
            self._file = None

    def summary(self) -> str:
        lines = [
            f"file: {self.path}",
            f"version: {self.version}  tensors: {self.n_tensors}  kv: {self.n_kv}",
            f"alignment: {self.alignment}  data_offset: {self.data_offset}  size: {self._size}",
            "metadata:",
        ]
        for k, v in self.metadata.items():
            if isinstance(v, list):
                preview = v[:8]
                suffix = " ..." if len(v) > 8 else ""
                text = f"{preview}{suffix} (len={len(v)})"
            elif isinstance(v, (bytes, bytearray)):
                text = f"<{len(v)} bytes>"
            else:
                text = repr(v)
            lines.append(f"  {k} ({self.metadata_types[k]}): {text}")
        if self.tensors:
            lines.append("tensors:")
            for t in self.tensors:
                lines.append(f"  {t}")
        return "\n".join(lines)

    def metadata_json(self, max_array: int = 8) -> dict[str, Any]:
        def enc(v: Any) -> Any:
            if isinstance(v, list):
                return [enc(x) for x in v[:max_array]]
            if isinstance(v, (bytes, bytearray)):
                return f"<{len(v)} bytes>"
            return v
        return {
            "path": str(self.path),
            "version": self.version,
            "alignment": self.alignment,
            "data_offset": self.data_offset,
            "n_tensors": self.n_tensors,
            "n_kv": self.n_kv,
            "metadata": {k: enc(v) for k, v in self.metadata.items()},
            "metadata_types": self.metadata_types,
            "tensors": [
                {"name": t.name, "type": t.type_name, "type_id": t.type_id,
                 "dims": list(t.dims), "bytes": t.n_bytes, "offset": t.offset}
                for t in self.tensors
            ],
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parse a GGUF header/index (prototype)")
    ap.add_argument("path", type=Path)
    ap.add_argument("--tensors", action="store_true", help="list tensors")
    ap.add_argument("--tensor", metavar="NAME", help="show one tensor")
    ap.add_argument("--dequant", metavar="NAME", help="dequantize one tensor and print stats")
    ap.add_argument("--json", action="store_true", help="dump metadata/tensor index as JSON")
    args = ap.parse_args(argv)

    with GGUFReader(args.path) as r:
        if args.json:
            print(json.dumps(r.metadata_json(), indent=2, ensure_ascii=False))
        elif args.tensor:
            t = r.get(args.tensor)
            print(t)
            print(f"logical shape: {t.shape}")
        elif args.dequant:
            t = r.get(args.dequant)
            y = t.dequantize()
            print(t)
            print(f"dequantized: shape={y.shape} dtype={y.dtype} "
                  f"min={np.min(y):.6g} max={np.max(y):.6g} "
                  f"mean={np.mean(y):.6g} finite={np.isfinite(y).all()}")
        elif args.tensors:
            for t in r.tensors:
                print(t)
        else:
            print(r.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
