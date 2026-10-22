"""Synthetic multi-layer assembly test: TinyModel with window-only (L0) + compressed (L2) blocks."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conftest import model_dir
from tinyinfer.config import TinyConfig
from tinyinfer.model import TinyModel


class FakeStore:
    def __init__(self, tensors):
        self._t = tensors

    def get(self, name):
        return self._t[name]


def build_fake_store(cfg, layers, seed=0):
    torch.manual_seed(seed)
    t = {}
    dim, h, n = cfg.dim, cfg.head_dim, cfg.n_heads
    t["embed.weight"] = torch.randn(cfg.vocab_size, dim, dtype=torch.bfloat16) * 0.02
    t["head.weight"] = torch.randn(cfg.vocab_size, dim, dtype=torch.bfloat16) * 0.02
    t["norm.weight"] = torch.ones(dim)

    def fp8(name, shape):
        t[name + ".weight"] = (torch.randn(*shape) * 0.02).to(torch.float8_e4m3fn)
        t[name + ".scale"] = torch.full(((shape[0] + 31) // 32, (shape[1] + 31) // 32), 127, dtype=torch.uint8)

    for lid in layers:
        p = f"layers.{lid}."
        fp8(p + "attn.wq_a", (cfg.q_lora_rank, dim))
        fp8(p + "attn.wq_b", (n * h, cfg.q_lora_rank))
        fp8(p + "attn.wkv", (h, dim))
        fp8(p + "attn.wo_a", (cfg.o_groups * cfg.o_lora_rank, n * h // cfg.o_groups))
        fp8(p + "attn.wo_b", (dim, cfg.o_groups * cfg.o_lora_rank))
        t[p + "attn.q_norm.weight"] = torch.ones(cfg.q_lora_rank)
        t[p + "attn.kv_norm.weight"] = torch.ones(h)
        t[p + "attn.attn_sink"] = torch.zeros(n)
        t[p + "attn_norm.weight"] = torch.ones(dim)
        t[p + "ffn_norm.weight"] = torch.ones(dim)
        t[p + "ffn.gate.weight"] = torch.randn(cfg.n_routed_experts, dim) * 0.02
        t[p + "ffn.gate.bias"] = torch.zeros(cfg.n_routed_experts)
        t[p + "ffn.gate.bias_vl"] = torch.zeros(cfg.n_routed_experts)
        for e in ("shared_experts.w1", "shared_experts.w2", "shared_experts.w3"):
            sign = -1 if e.endswith("w2") else 1
            shp = (dim, cfg.moe_inter_dim) if sign < 0 else (cfg.moe_inter_dim, dim)
            fp8(p + "ffn." + e, shp)
        for i in range(cfg.n_routed_experts):
            ep = f"{p}ffn.experts.{i}."
            for e, shape in (("w1", (cfg.moe_inter_dim, dim)), ("w3", (cfg.moe_inter_dim, dim)), ("w2", (dim, cfg.moe_inter_dim))):
                t[ep + e + ".weight"] = torch.randint(0, 256, (shape[0], shape[1] // 2), dtype=torch.uint8)
                t[ep + e + ".scale"] = torch.full((shape[0], shape[1] // 32), 127, dtype=torch.uint8)
        mix = (2 + cfg.hc_mult) * cfg.hc_mult
        t[p + "hc_attn_fn"] = torch.randn(mix, cfg.hc_mult * dim) * 0.02
        t[p + "hc_ffn_fn"] = torch.randn(mix, cfg.hc_mult * dim) * 0.02
        t[p + "hc_attn_base"] = torch.zeros(mix)
        t[p + "hc_ffn_base"] = torch.zeros(mix)
        t[p + "hc_attn_scale"] = torch.ones(3)
        t[p + "hc_ffn_scale"] = torch.ones(3)
        if cfg.compress_ratios[lid] > 0:
            st = p + "attn.compressor."
            t[st + "norm.weight"] = torch.ones(h)
            t[st + "wkv.weight"] = torch.randn(h, dim) * 0.02
            t[st + "wgate.weight"] = torch.randn(h, dim) * 0.02
            it = p + "attn.indexer."
            t[it + "wk.weight"] = torch.randn(cfg.index_head_dim, h) * 0.02
            t[it + "k_norm.weight"] = torch.ones(cfg.index_head_dim)
            t[it + "weights_proj.weight"] = torch.randn(cfg.index_n_heads, dim) * 0.02
            t[it + "wq_b.weight"] = (torch.randn(cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank) * 0.02).to(torch.float8_e4m3fn)
            t[it + "wq_b.scale"] = torch.full(
                ((cfg.index_n_heads * cfg.index_head_dim + 31) // 32, (cfg.q_lora_rank + 31) // 32), 127, dtype=torch.uint8
            )
    return FakeStore(t)


@pytest.mark.kernel
def test_two_layer_forward_and_decode():
    cfg = TinyConfig.from_hf(model_dir())
    store = build_fake_store(cfg, [0, 2], seed=7)
    model = TinyModel(store, cfg, layers=[0, 2])
    model.reset()
    ids = torch.tensor([[19923, 14, 342, 1030]])
    logits = model.forward(ids, 0)
    assert logits.shape == (1, 4, cfg.vocab_size), logits.shape
    assert torch.isfinite(logits).all()
    # one decode step through both layers
    logits = model.forward(torch.tensor([[22651]]), 4)
    assert logits.shape == (1, 1, cfg.vocab_size), logits.shape
    assert torch.isfinite(logits).all()
    print("two-layer (window + compressed) forward/decode ok")


if __name__ == "__main__":
    test_two_layer_forward_and_decode()
    print("ALL OK")
