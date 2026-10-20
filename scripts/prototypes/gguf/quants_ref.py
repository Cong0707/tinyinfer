"""NumPy reference dequantizers for a subset of GGUF tensor types.

tinyinfer keeps weights in their packed storage form and dequantizes on the
consumer side, so only dequantization is implemented here.  Every byte layout
below matches llama.cpp ggml/src/ggml-common.h and is cross-checked against
gguf-py's reference implementation by selftest.py.

Implemented types:
  Q4_K, Q5_K, Q6_K, Q8_0, MXFP4, F32, F16, BF16
plus the HF -> GGUF MXFP4 nibble repack used by llama.cpp's DeepSeek-V4
converter (conversion/base.py:repack_mxfp4_blocks).
"""
from __future__ import annotations

import numpy as np

# (block_size, type_size_in_bytes) from gguf-py/gguf/constants.py GGML_QUANT_SIZES
QUANT_SIZES: dict[str, tuple[int, int]] = {
    "F32": (1, 4),
    "F16": (1, 2),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q8_1": (32, 36),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_K": (256, 292),
    "IQ2_XXS": (256, 66),
    "IQ2_XS": (256, 74),
    "IQ3_XXS": (256, 98),
    "IQ1_S": (256, 50),
    "IQ4_NL": (32, 18),
    "IQ3_S": (256, 110),
    "IQ2_S": (256, 82),
    "IQ4_XS": (256, 136),
    "I8": (1, 1),
    "I16": (1, 2),
    "I32": (1, 4),
    "I64": (1, 8),
    "F64": (1, 8),
    "IQ1_M": (256, 56),
    "BF16": (1, 2),
    "TQ1_0": (256, 54),
    "TQ2_0": (256, 66),
    "MXFP4": (32, 17),
    "NVFP4": (64, 36),
    "Q1_0": (128, 18),
    "Q2_0": (64, 18),
}

# ggml_type enum values (full set needed to parse a model header).
# Dequantization is only implemented for the types in _DEQUANTIZERS.
TYPE_IDS: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0",
}
TYPE_NAMES = {v: k for k, v in TYPE_IDS.items()}

QK_K = 256
K_SCALE_SIZE = 12

# E2M1 magnitudes; sign is the high bit.  This is the "real" value table.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
# ggml's kvalues_mxfp4 stores doubled magnitudes and dequant multiplies by 0.5
KVALUES_MXFP4_DOUBLED = np.array([0, 1, 2, 3, 4, 6, 8, 12,
                                  0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def _bytes(data: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(data)
    if a.dtype != np.uint8:
        a = a.view(np.uint8)
    return a


def _as_blocks(data: np.ndarray, type_size: int) -> np.ndarray:
    a = _bytes(data)
    if a.size % type_size != 0:
        raise ValueError(f"data size {a.size} is not a multiple of block type size {type_size}")
    return a.reshape(-1, type_size)


def _f16(cols: np.ndarray) -> np.ndarray:
    """View a contiguous (n, 2) uint8 array as fp16 -> float32."""
    return np.ascontiguousarray(cols).view(np.float16).astype(np.float32).reshape(-1)


def _q4k_scales_mins(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the 12-byte Q4_K/Q5_K 6-bit scale/min encoding.

    Layout (one super-block, 8 sub-blocks of 32 values):
      bytes 0..3 : low 6 bits of scales 0..3
      bytes 4..7 : low 6 bits of mins 0..3
      bytes 8..11: low nibble = scales 4..7, high nibble = mins 4..7
      bits 6..7 of bytes 0..3 hold the high 2 bits of scales 4..7
      bits 6..7 of bytes 4..7 hold the high 2 bits of mins 4..7
    """
    s = np.ascontiguousarray(scales).view(np.uint8).reshape(-1, 3, 4)
    d, m, md = s[:, 0, :], s[:, 1, :], s[:, 2, :]
    sc = np.concatenate([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape(-1, 8), mn.reshape(-1, 8)


def dequantize_q4_k(data: np.ndarray) -> np.ndarray:
    """Q4_K: 256 values per 144-byte block, x = d*sc*q - dmin*m."""
    blocks = _as_blocks(data, 144)
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    dmin = _f16(blocks[:, 2:4])
    sc, mn = _q4k_scales_mins(blocks[:, 4:16])
    qs = blocks[:, 16:144].reshape(n, 4, 32)

    q = np.empty((n, QK_K), dtype=np.float32)
    for j in range(4):
        q[:, j * 64:j * 64 + 32] = qs[:, j] & 0x0F
        q[:, j * 64 + 32:j * 64 + 64] = qs[:, j] >> 4

    d32 = d[:, None] * np.repeat(sc, 32, axis=1)
    m32 = dmin[:, None] * np.repeat(mn, 32, axis=1)
    return d32 * q - m32


def dequantize_q5_k(data: np.ndarray) -> np.ndarray:
    """Q5_K: 256 values per 176-byte block; 4-bit quants plus 1 high bit."""
    blocks = _as_blocks(data, 176)
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    dmin = _f16(blocks[:, 2:4])
    sc, mn = _q4k_scales_mins(blocks[:, 4:16])
    qh = blocks[:, 16:48]          # 32 bytes, one bit per 32-value sub-block
    qs = blocks[:, 48:176].reshape(n, 4, 32)

    q = np.empty((n, QK_K), dtype=np.float32)
    for j in range(4):
        lo = (qs[:, j] & 0x0F) + 16 * ((qh >> np.uint8(2 * j)) & 1)
        hi = (qs[:, j] >> 4) + 16 * ((qh >> np.uint8(2 * j + 1)) & 1)
        q[:, j * 64:j * 64 + 32] = lo
        q[:, j * 64 + 32:j * 64 + 64] = hi

    d32 = d[:, None] * np.repeat(sc, 32, axis=1)
    m32 = dmin[:, None] * np.repeat(mn, 32, axis=1)
    return d32 * q - m32


def dequantize_q6_k(data: np.ndarray) -> np.ndarray:
    """Q6_K: 256 values per 210-byte block; 6-bit signed quants."""
    blocks = _as_blocks(data, 210)
    n = blocks.shape[0]
    ql = blocks[:, 0:128].reshape(n, 2, 64)
    qh = blocks[:, 128:192].reshape(n, 2, 32)
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32).reshape(n, 16)
    d = _f16(blocks[:, 208:210])

    out = np.empty((n, QK_K), dtype=np.float32)
    for chunk in range(2):
        ql0 = ql[:, chunk, :32]
        ql1 = ql[:, chunk, 32:]
        qh0 = qh[:, chunk, :]
        q1 = (ql0 & 0x0F) | ((qh0 & 0x03) << 4)
        q2 = (ql1 & 0x0F) | (((qh0 >> 2) & 0x03) << 4)
        q3 = (ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4)
        q4 = (ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4)
        q = np.concatenate([q1, q2, q3, q4], axis=-1).astype(np.float32) - 32.0
        sc = scales[:, chunk * 8:(chunk + 1) * 8]
        out[:, chunk * 128:(chunk + 1) * 128] = q * (d[:, None] * np.repeat(sc, 16, axis=1))
    return out


def dequantize_q8_0(data: np.ndarray) -> np.ndarray:
    """Q8_0: 32 values per 34-byte block, x = d*q."""
    blocks = _as_blocks(data, 34)
    d = _f16(blocks[:, 0:2])
    q = blocks[:, 2:34].view(np.int8).astype(np.float32)
    return d[:, None] * q


def e8m0_to_float(e: np.ndarray) -> np.ndarray:
    """Decode E8M0 biased exponent bytes (uint8) to float32 powers of two."""
    e = np.asarray(e, dtype=np.int32)
    # e=0 is the minimum normal exponent in ggml; 2^(-127).  0xFF is not masked.
    return np.ldexp(np.ones_like(e, dtype=np.float32), e - 127).astype(np.float32)


def dequantize_mxfp4(data: np.ndarray) -> np.ndarray:
    """MXFP4: 32 values per 17-byte block (one E8M0 scale + 16 E2M1 bytes).

    Byte j holds element j in the low nibble and element j+16 in the high
    nibble, matching ggml block_mxfp4.
    """
    blocks = _as_blocks(data, 17)
    n = blocks.shape[0]
    e = blocks[:, 0].astype(np.int32)
    # ggml_e8m0_to_fp32: 2^(e-127); e=0 maps to 2^-127.
    scale = np.ldexp(np.float32(1.0), e - 127)
    qs = blocks[:, 1:17]
    vals = np.empty((n, 32), dtype=np.float32)
    vals[:, :16] = KVALUES_MXFP4_DOUBLED[(qs & 0x0F).astype(np.intp)] * 0.5
    vals[:, 16:] = KVALUES_MXFP4_DOUBLED[(qs >> 4).astype(np.intp)] * 0.5
    return vals * scale[:, None]


def repack_hf_mxfp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Repack HF/compressed-tensors MXFP4 to ggml block_mxfp4 byte layout.

    Source (DeepSeek-V4 experts):
      packed uint8 [rows, cols/2], element 2i in the low nibble, 2i+1 in high
      scale  uint8 [rows, cols/32], one E8M0 byte per 32-element group
    Destination (ggml block_mxfp4):
      per group: one scale byte, then 16 code bytes where byte j holds
      element j in the low nibble and element j+16 in the high nibble.

    This is the same bit move as llama.cpp conversion/base.py:repack_mxfp4_blocks.
    It is lossless; no requantization happens.
    """
    p = np.ascontiguousarray(packed).view(np.uint8)
    s = np.ascontiguousarray(scale).view(np.uint8)
    rows, packed_cols = p.shape
    cols = packed_cols * 2
    if cols % 32 != 0:
        raise ValueError(f"row has {cols} values, expected a multiple of 32")
    n_blocks = cols // 32
    if s.shape != (rows, n_blocks):
        raise ValueError(f"scale shape {s.shape} does not match {(rows, n_blocks)}")

    src = p.reshape(rows, n_blocks, 16)
    lo = src & 0x0F
    hi = (src >> 4) & 0x0F
    vals = np.stack([lo, hi], axis=-1).reshape(rows, n_blocks, 32)
    qs = vals[:, :, :16] | (vals[:, :, 16:] << 4)
    raw = np.concatenate([s.reshape(rows, n_blocks, 1), qs.astype(np.uint8)], axis=-1)
    return raw.reshape(rows, n_blocks * 17)


def hf_fp4_to_float(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize HF packed E2M1 + E8M0 per-32 scale to float32 [rows, cols].

    Kept as the independent oracle for repack_hf_mxfp4.
    """
    p = np.ascontiguousarray(packed).view(np.uint8)
    s = np.asarray(scale).view(np.uint8).astype(np.int32)
    rows, packed_cols = p.shape
    out = np.empty((rows, packed_cols * 2), dtype=np.float32)
    out[:, 0::2] = E2M1[(p & 0x0F).astype(np.intp)]
    out[:, 1::2] = E2M1[(p >> 4).astype(np.intp)]
    s = np.repeat(np.ldexp(np.float32(1.0), s - 127), 32, axis=1)
    return out * s[:, :out.shape[1]]


_DEQUANTIZERS = {
    "Q4_K": dequantize_q4_k,
    "Q5_K": dequantize_q5_k,
    "Q6_K": dequantize_q6_k,
    "Q8_0": dequantize_q8_0,
    "MXFP4": dequantize_mxfp4,
}


def dequantize(data: np.ndarray, type_name: str) -> np.ndarray:
    """Dequantize raw block bytes; returns a flat (n_blocks*block_size,) array."""
    if type_name == "F32":
        return np.frombuffer(_bytes(data), dtype="<f4").astype(np.float32)
    if type_name == "F16":
        return np.frombuffer(_bytes(data), dtype="<f2").astype(np.float32)
    if type_name == "BF16":
        u = np.frombuffer(_bytes(data), dtype="<u2").astype(np.uint32)
        return (u << 16).view(np.float32)
    try:
        return _DEQUANTIZERS[type_name](data)
    except KeyError as e:
        raise NotImplementedError(f"dequantization for {type_name!r} is not implemented") from e


def dequantize_tensor(data: np.ndarray, type_name: str, dims: list[int]) -> np.ndarray:
    """Dequantize a GGUF tensor and reshape it to GGUF's logical shape.

    GGUF stores ne[0] first (innermost / contiguous).  The returned array has
    shape tuple(reversed(dims)): a 2-D weight [ne0, ne1] becomes [ne1, ne0],
    matching HF/torch row-major convention.
    """
    if not dims:
        raise ValueError("tensor has no dims")
    block, type_size = QUANT_SIZES[type_name]
    ne0 = dims[0]
    if ne0 % block != 0:
        raise ValueError(f"ne0={ne0} is not a multiple of {type_name} block size {block}")
    rows = 1
    for d in dims[1:]:
        rows *= d
    raw = np.asarray(data).reshape(-1)
    expected = rows * (ne0 // block) * type_size
    if raw.size != expected:
        raise ValueError(f"expected {expected} bytes, got {raw.size}")
    out = dequantize(raw, type_name).reshape(rows, ne0)
    out_dims = tuple(int(d) for d in reversed(dims[1:])) + (ne0,)
    return out.reshape(out_dims)
