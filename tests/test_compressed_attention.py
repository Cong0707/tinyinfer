"""Synthetic tests for MLACompressedAttention (layer 2 topology)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.compressed_attention import MLACompressedAttention
from conftest import model_dir
from tinyinfer.config import TinyConfig


class FakeStore:
    def __init__(self, tensors):
        self._t = tensors

    def get(self, name):
        return self._t[name]


def make_store(cfg, layer_id=2, seed=0):
    torch.manual_seed(seed)
    h, dim = cfg.head_dim, cfg.dim
    n = cfg.n_heads
    t = {}
    def put_fp8(name, shape):
        t[name + ".weight"] = (torch.randn(*shape) * 0.02).to(torch.float8_e4m3fn)
        t[name + ".scale"] = torch.full(((shape[0] + 31) // 32, (shape[1] + 31) // 32), 127, dtype=torch.uint8)
    put_fp8(f"layers.{layer_id}.attn.wq_a", (cfg.q_lora_rank, dim))
    put_fp8(f"layers.{layer_id}.attn.wq_b", (n * h, cfg.q_lora_rank))
    put_fp8(f"layers.{layer_id}.attn.wkv", (h, dim))
    put_fp8(f"layers.{layer_id}.attn.wo_a", (cfg.o_groups * cfg.o_lora_rank, n * h // cfg.o_groups))
    put_fp8(f"layers.{layer_id}.attn.wo_b", (dim, cfg.o_groups * cfg.o_lora_rank))
    t[f"layers.{layer_id}.attn.q_norm.weight"] = torch.ones(cfg.q_lora_rank)
    t[f"layers.{layer_id}.attn.kv_norm.weight"] = torch.ones(h)
    t[f"layers.{layer_id}.attn.attn_sink"] = torch.zeros(n)
    st = f"layers.{layer_id}.attn.compressor."
    t[st + "norm.weight"] = torch.ones(h)
    t[st + "wkv.weight"] = torch.randn(h, dim) * 0.02
    t[st + "wgate.weight"] = torch.randn(h, dim) * 0.02
    it = f"layers.{layer_id}.attn.indexer."
    t[it + "wk.weight"] = torch.randn(cfg.index_head_dim, h) * 0.02
    t[it + "k_norm.weight"] = torch.ones(cfg.index_head_dim)
    t[it + "weights_proj.weight"] = torch.randn(cfg.index_n_heads, dim) * 0.02
    t[it + "wq_b.weight"] = (torch.randn(cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank) * 0.02).to(torch.float8_e4m3fn)
    t[it + "wq_b.scale"] = torch.full(
        ((cfg.index_n_heads * cfg.index_head_dim + 31) // 32, (cfg.q_lora_rank + 31) // 32), 127, dtype=torch.uint8
    )
    return FakeStore(t)


def test_prefill_and_decode():
    cfg = TinyConfig.from_hf(model_dir())
    assert cfg.compress_ratios[2] == 2
    attn = MLACompressedAttention(make_store(cfg), 2, cfg)
    attn.reset()
    torch.manual_seed(1)
    x = torch.randn(1, 8, cfg.dim, dtype=torch.bfloat16) * 0.02
    y = attn.forward(x, 0)
    assert y.shape == (1, 8, cfg.dim), y.shape
    assert torch.isfinite(y.float()).all(), "prefill produced non-finite values"
    for pos in range(8, 12):
        y = attn.forward(x[:, pos % 8 : pos % 8 + 1], pos)
        assert y.shape == (1, 1, cfg.dim), y.shape
        assert torch.isfinite(y.float()).all(), f"decode step {pos} non-finite"
    print("compressed attention prefill+decode ok")


if __name__ == "__main__":
    test_prefill_and_decode()
    print("ALL OK")
