"""Generate a small synthetic GGUF file for parser/dequant tests.

No model weights and no external packages are needed; only NumPy and the
standard library.  The file has the same layout llama.cpp writes: metadata
KVs, tensor directory, 32-byte alignment, then tensor payloads.

Usage:
    python make_sample_gguf.py --out sample.gguf
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

try:
    from . import quants_ref
except ImportError:  # direct script execution
    import quants_ref  # type: ignore

MAGIC = 0x46554747
VERSION = 3
ALIGNMENT = 32

STRING = 8
ARRAY = 9
UINT32 = 4
FLOAT32 = 6
BOOL = 7


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, vtype: int, value) -> bytes:
    out = _s(key) + struct.pack("<I", vtype)
    if vtype == STRING:
        return out + _s(value)
    if vtype == UINT32:
        return out + struct.pack("<I", value)
    if vtype == FLOAT32:
        return out + struct.pack("<f", value)
    if vtype == BOOL:
        return out + struct.pack("<?", value)
    if vtype == ARRAY:
        subtype, items = value
        out += struct.pack("<I", subtype) + struct.pack("<Q", len(items))
        if subtype == STRING:
            return out + b"".join(_s(x) for x in items)
        fmt = {UINT32: "I", FLOAT32: "f"}[subtype]
        return out + struct.pack(f"<{len(items)}{fmt}", *items)
    raise ValueError(f"unsupported vtype {vtype}")


def _half(x: float) -> np.ndarray:
    return np.frombuffer(np.array([x], dtype=np.float16).tobytes(), dtype=np.uint8)


def _blocks_q4_k(rng: np.random.Generator, n: int) -> np.ndarray:
    b = rng.integers(0, 256, size=(n, 144), dtype=np.uint8)
    b[:, 0:2] = _half(1.0)   # d
    b[:, 2:4] = _half(0.0)   # dmin
    return b


def _blocks_q6_k(rng: np.random.Generator, n: int) -> np.ndarray:
    b = rng.integers(0, 256, size=(n, 210), dtype=np.uint8)
    b[:, 208:210] = _half(1.0)
    return b


def _blocks_mxfp4(rng: np.random.Generator, n: int) -> np.ndarray:
    b = rng.integers(0, 256, size=(n, 17), dtype=np.uint8)
    b[:, 0] = 127             # E8M0: 2^0
    return b


def _tensor_info(name: str, dims: tuple[int, ...], type_id: int, offset: int) -> bytes:
    out = _s(name) + struct.pack("<I", len(dims))
    out += b"".join(struct.pack("<Q", d) for d in dims)
    return out + struct.pack("<I", type_id) + struct.pack("<Q", offset)


def build(out_path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)

    metadata = [
        ("general.architecture", STRING, "deepseek4"),
        ("general.name", STRING, "tinyinfer-gguf-proto"),
        ("general.alignment", UINT32, ALIGNMENT),
        ("deepseek4.block_count", UINT32, 2),
        ("deepseek4.embedding_length", UINT32, 256),
        ("deepseek4.expert_count", UINT32, 4),
        ("deepseek4.attention.compress_ratios", ARRAY, (UINT32, [0, 4])),
        ("tinyinfer.proto.float", FLOAT32, 3.5),
        ("tinyinfer.proto.bool", BOOL, True),
        ("tokenizer.ggml.model", STRING, "gpt2"),
        ("tokenizer.ggml.tokens", ARRAY, (STRING, ["a", "b", "c", "d"])),
        ("tokenizer.ggml.merges", ARRAY, (STRING, ["a b"])),
    ]

    # name, dims (GGUF order, ne0 first), ggml type id, payload
    tensors = [
        ("token_embd.weight", (256, 4), quants_ref.TYPE_NAMES["Q4_K"],
         _blocks_q4_k(rng, 4).tobytes()),
        ("blk.0.ffn_down_exps.weight", (256, 8, 2), quants_ref.TYPE_NAMES["Q6_K"],
         _blocks_q6_k(rng, 16).tobytes()),
        ("blk.0.ffn_gate_exps.weight", (32, 16, 4), quants_ref.TYPE_NAMES["MXFP4"],
         _blocks_mxfp4(rng, 64).tobytes()),
        ("blk.0.attn_q_a_norm.weight", (256,), quants_ref.TYPE_NAMES["F32"],
         rng.standard_normal(256, dtype=np.float32).tobytes()),
        # Header-only test: IQ2_XXS is recognized for parsing, dequant is not implemented.
        ("blk.1.ffn_up_exps.weight", (256, 4, 2), quants_ref.TYPE_NAMES["IQ2_XXS"],
         rng.integers(0, 256, size=8 * 66, dtype=np.uint8).tobytes()),
    ]

    kv_bytes = b"".join(_kv(k, t, v) for k, t, v in metadata)
    tensor_infos = []
    data_offset = 0
    for name, dims, type_id, payload in tensors:
        tensor_infos.append(_tensor_info(name, dims, type_id, data_offset))
        data_offset += len(payload)
        pad = (-data_offset) % ALIGNMENT
        data_offset += pad

    header = struct.pack("<IIQQ", MAGIC, VERSION, len(tensors), len(metadata))
    header += kv_bytes + b"".join(tensor_infos)
    header += b"\x00" * ((-len(header)) % ALIGNMENT)
    data = bytearray()
    for _name, _dims, _type_id, payload in tensors:
        data += payload
        data += b"\x00" * ((-len(data)) % ALIGNMENT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + data)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("sample.gguf"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    p = build(args.out, args.seed)
    print(f"wrote {p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
