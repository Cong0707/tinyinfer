"""Engram: n-gram hash lookups written into the residual stream (layers 1 and 14).

The 384M-row fp8 tables stay on disk; every lookup reads only the needed rows via
memory map (24 rows/token, ~264B each). Hashing/normalization follows the official
reference implementation (reference/official/engram.py, MIT).
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import TinyConfig
from .quant import dequant_fp8_block, e8m0_to_float
from .weights import ShardStore

DEAD = np.int64(-1)


# ---------------------------------------------------------------------------
# safetensors row reader (mmap, no full-tensor materialization)
# ---------------------------------------------------------------------------
class _TensorFile:
    """Row-addressable view over one tensor inside a safetensors shard."""

    def __init__(self, path: Path, tensor_name: str):
        self.path = path
        self._f = open(path, "rb")
        n = struct.unpack("<Q", self._f.read(8))[0]
        header = json.loads(self._f.read(n).decode("utf-8"))
        info = header[tensor_name]
        self.dtype = info["dtype"]
        self.shape = tuple(info["shape"])
        self.data_start = 8 + n + info["data_offsets"][0]
        self.bytes_per_row = info["data_offsets"][1] - info["data_offsets"][0]
        self.bytes_per_row //= max(self.shape[0], 1)
        self.rows = self.shape[0]

    def read_rows(self, idx: np.ndarray) -> np.ndarray:
        """idx: [..] int64 row ids -> uint8 [.., bytes_per_row]."""
        idx = np.asarray(idx, dtype=np.int64)
        flat = idx.reshape(-1)
        out = np.empty((flat.size, self.bytes_per_row), dtype=np.uint8)
        for j, r in enumerate(flat.tolist()):
            self._f.seek(self.data_start + r * self.bytes_per_row)
            out[j] = np.frombuffer(self._f.read(self.bytes_per_row), dtype=np.uint8)
        return out.reshape(idx.shape + (self.bytes_per_row,))

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# hashing (mirrors reference/official/engram.py)
# ---------------------------------------------------------------------------
def _find_next_prime(start: int, seen: set[int]) -> int:
    from sympy import isprime

    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = backend.id_to_token(token_id) if "\ufffd" in text else (normalizer.normalize_str(text) or text)
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def _hash_multipliers(layer_ids, max_ngram_size: int, vocab_size: int) -> np.ndarray:
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        rows.append(gen.integers(0, bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


class EngramHashes:
    """Position -> n-gram hash ids for all engram layers (official NgramHashState semantics)."""

    def __init__(self, cfg: TinyConfig, tokenizer):
        self.cfg = cfg
        self.max_n = cfg.engram_max_ngram_size
        self.n_heads = cfg.engram_n_heads
        self.layer_ids = tuple(cfg.engram_layer_ids)
        self.pad_id = cfg.engram_pad_id
        token_map, vocab = build_compressed_token_map(tokenizer)
        if vocab != cfg.engram_compressed_vocab_size:
            raise ValueError(f"compressed vocab mismatch: {vocab} != {cfg.engram_compressed_vocab_size}")
        self.token_map = token_map
        self.pad_id = token_map[cfg.engram_pad_id]
        # primes: one bucket range per (layer, ngram-size, head)
        primes, seen = [], set()
        for _ in self.layer_ids:
            per = []
            for _n in range(self.max_n - 1):
                sizes, cur = [], cfg.engram_vocab_size - 1
                for _h in range(self.n_heads):
                    cur = _find_next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per.append(tuple(sizes))
            primes.append(tuple(per))
        self.primes = np.array(primes, dtype=np.int64)  # [L, max_n-1, heads]
        flat = [np.cumsum([0, *[p for per in layer for p in per][:-1]]) for layer in primes]
        self.offsets = np.array(flat, dtype=np.int64)  # [L, max_n-1 * heads]
        self.multipliers = torch.tensor(_hash_multipliers(self.layer_ids, self.max_n, vocab))  # [L, max_n]
        self.n_cols = (self.max_n - 1) * self.n_heads
        self._cache: list[int] = []

    def reset(self) -> None:
        self._cache = []

    def __call__(self, ids: list[int], start_pos: int = 0) -> np.ndarray:
        """Returns int64 hash ids [L, n_engram_layers, n_cols] for the given new tokens."""
        if start_pos > len(self._cache):
            self._cache.extend([self.cfg.engram_pad_id] * (start_pos - len(self._cache)))
        elif start_pos < len(self._cache):
            self._cache = self._cache[:start_pos]
        self._cache.extend(int(t) for t in ids)
        n_new = len(ids)
        positions = np.arange(start_pos, start_pos + n_new, dtype=np.int64)
        tokens = np.stack([
            np.where(positions < shift, self.pad_id,
                     np.array([self.token_map[self._cache[p - shift]] if p - shift >= 0 else self.pad_id
                               for p in positions], dtype=np.int64))
            for shift in range(self.max_n)
        ], axis=-1)  # [n_new, max_n]
        prod = tokens[None, :, :] * self.multipliers.numpy()[:, None, :]  # [L, n_new, max_n]
        rolling = prod[..., 0]
        hashes = []
        for i in range(1, self.max_n):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            hashes.append(rolling[..., None] % self.primes[:, i - 1, :][:, None, :])
        out = np.concatenate(hashes, axis=-1) + self.offsets[:, None, :]
        return out  # [L, n_new, (max_n-1)*heads]


# ---------------------------------------------------------------------------
# Engram module
# ---------------------------------------------------------------------------
class Engram:
    """One engram layer: fp8 table (row-read), wkv projection, gated residual write."""

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, hashes: EngramHashes):
        p = f"layers.{layer_id}.attn.engram."
        self.layer_id = layer_id
        self.cfg = cfg
        self.hashes = hashes
        self.hash_index = cfg.engram_layer_ids.index(layer_id)
        self.dim = cfg.dim
        self.hc = cfg.hc_mult
        self.head_dim = cfg.engram_head_dim
        self.eps = cfg.rms_norm_eps
        self.clamp = 1e-6
        self.n_cols = hashes.n_cols

        shard = store.shard_of(p + "embed.weight")
        self.table = _TensorFile(store.model_dir / shard, "layers.%d.engram.embed.weight" % layer_id)
        self.table_scale = _TensorFile(store.model_dir / shard, "layers.%d.engram.embed.scale" % layer_id)
        self.wkv_w = store.get(p + "wkv.weight")
        self.wkv_s = store.get(p + "wkv.scale")
        self.q_weight = store.get(p + "q_weight").float()
        self.k_weight = store.get(p + "k_weight").float()
        self._bf16_cache: torch.Tensor | None = None

    def _tabular(self, rows: np.ndarray) -> torch.Tensor:
        raw = self.table.read_rows(rows)  # [..., 256] uint8
        sc = self.table_scale.read_rows(rows)  # [..., 8] uint8
        vals = raw.view(np.float8_e4m3fn) if hasattr(np, "float8_e4m3fn") else raw
        # decode in torch (fp8 -> f32) with per-32 scales
        w = torch.from_numpy(np.ascontiguousarray(raw)).view(torch.float8_e4m3fn).float()
        s = torch.from_numpy(np.ascontiguousarray(sc)).view(torch.uint8)
        s = e8m0_to_float(s)
        w = w.unflatten(-1, (-1, 32)) * s.unsqueeze(-1)
        return w.flatten(-2)

    def forward(self, x: torch.Tensor, hash_ids: np.ndarray, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [1, L, hc, dim]; hash_ids: [L, (max_n-1)*heads] for this layer."""
        rows = hash_ids.reshape(-1)
        vals = self._tabular(rows)  # [n_rows, 256]
        vals = vals.view(1, -1, self.n_cols, self.head_dim).to(x.device).to(x.dtype)
        kv = F.linear(vals.flatten(-2), self._bf16_wkv())  # [1, L, dim*(hc+1)]
        key, value = kv.split([self.hc * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc, self.dim))
        weight = self.q_weight * self.k_weight
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)

    def _bf16_wkv(self) -> torch.Tensor:
        if self._bf16_cache is None:
            self._bf16_cache = dequant_fp8_block(self.wkv_w, self.wkv_s)
        return self._bf16_cache

    def to(self, device):
        if self._bf16_cache is not None:
            self._bf16_cache = self._bf16_cache.to(device)
        self.q_weight = self.q_weight.to(device)
        self.k_weight = self.k_weight.to(device)
        return self

    def reset(self):
        pass
