"""Synthetic tests for the Compressor / Indexer path (layer 2 topology, no real weights)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.compressor import Compressor, Indexer, select_candidate_blocks, shared_attn
from tinyinfer.config import TinyConfig
from tinyinfer.nn import precompute_freqs_cis


class FakeStore:
    def __init__(self, tensors):
        self._t = tensors

    def get(self, name):
        return self._t[name]


def make_cfg():
    return TinyConfig.from_hf(ROOT / "cache" / "weights")


def make_store(cfg, layer_id=2, seed=0):
    torch.manual_seed(seed)
    h, dim = cfg.head_dim, cfg.dim
    t = {
        f"layers.{layer_id}.attn.compressor.norm.weight": torch.ones(h),
        f"layers.{layer_id}.attn.compressor.wkv.weight": torch.randn(h, dim) * 0.02,
        f"layers.{layer_id}.attn.compressor.wgate.weight": torch.randn(h, dim) * 0.02,
        f"layers.{layer_id}.attn.indexer.wk.weight": torch.randn(cfg.index_head_dim, h) * 0.02,
        f"layers.{layer_id}.attn.indexer.k_norm.weight": torch.ones(cfg.index_head_dim),
        f"layers.{layer_id}.attn.indexer.weights_proj.weight": torch.randn(cfg.index_n_heads, dim) * 0.02,
    }
    # fp8 wq_b + e8m0 scale (identity scales so the fp8 values pass through)
    wq = torch.randn(cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank).to(torch.float8_e4m3fn)
    t[f"layers.{layer_id}.attn.indexer.wq_b.weight"] = wq
    t[f"layers.{layer_id}.attn.indexer.wq_b.scale"] = torch.full(
        ((cfg.index_n_heads * cfg.index_head_dim + 31) // 32, (cfg.q_lora_rank + 31) // 32),
        127, dtype=torch.uint8,
    )
    return FakeStore(t)


def test_compressor_prefill():
    cfg = make_cfg()
    store = make_store(cfg)
    comp = Compressor(store, 2, cfg)
    assert comp.ratio == 2
    torch.manual_seed(1)
    x = torch.randn(1, 4, cfg.dim, dtype=torch.bfloat16)
    out = comp.forward(x, 0)
    assert out.shape == (1, 2, cfg.head_dim), out.shape
    # compare against the explicit gated pooling formula
    xf = x.float()
    kv = torch.nn.functional.linear(xf, comp.wkv)
    sc = torch.nn.functional.linear(xf, comp.wgate)
    groups = kv.unflatten(1, (-1, 2))
    gsc = sc.unflatten(1, (-1, 2))
    ref = (groups * gsc.softmax(dim=2)).sum(dim=2)
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * comp.norm_w
    assert torch.allclose(out.float(), ref, atol=1e-2)
    print("compressor prefill ok")


def test_compressor_decode():
    cfg = make_cfg()
    comp = Compressor(make_store(cfg, seed=3), 2, cfg)
    torch.manual_seed(2)
    x = torch.randn(1, 10, cfg.dim, dtype=torch.bfloat16)
    comp.forward(x[:, :6], 0)  # prefill 6 tokens -> 3 groups, state empty
    outs = []
    for pos in range(6, 10):
        o = comp.forward(x[:, pos:pos + 1], pos)
        outs.append(o)
    # groups complete at pos 7 and 9
    assert outs[0] is None and outs[2] is None
    assert outs[1] is not None and outs[1].shape == (1, 1, cfg.head_dim)
    assert outs[3] is not None and outs[3].shape == (1, 1, cfg.head_dim)
    print("compressor decode ok")


def test_indexer_prefill():
    cfg = make_cfg()
    store = make_store(cfg, seed=4)
    freqs = precompute_freqs_cis(cfg.qk_rope_head_dim, 64, cfg.max_position_embeddings if cfg.max_position_embeddings < 100000 else 4096, cfg.compress_rope_theta, 16, 32, 1)
    idx = Indexer(store, 2, cfg, freqs)
    torch.manual_seed(5)
    x = torch.randn(1, 8, cfg.dim, dtype=torch.bfloat16)
    qr = torch.randn(1, 8, cfg.q_lora_rank, dtype=torch.bfloat16)
    latent = torch.randn(1, 4, cfg.head_dim, dtype=torch.bfloat16)
    out = idx.forward(x, qr, latent, 0, offset=128)
    assert out.shape == (1, 8, min(cfg.index_topk, 4)), out.shape
    assert out.dtype == torch.int32
    valid = out[out >= 0]
    assert valid.numel() > 0
    assert int(valid.min()) >= 128 and int(valid.max()) < 128 + 4, (valid.min(), valid.max())
    assert shared_attn.index_k is not None
    print("indexer prefill ok")


def test_select_candidate_blocks():
    logits = torch.randn(2, 16)
    logits[:, 12:] = -torch.inf
    keep = select_candidate_blocks(logits, torch.tensor(12), 2, 4)
    assert keep.shape == (2, 16) and keep.dtype == torch.bool
    for row in range(2):
        assert keep[row].sum().item() > 0
        assert not keep[row, 12:].any()
    print("candidate blocks ok")


if __name__ == "__main__":
    test_compressor_prefill()
    test_compressor_decode()
    test_indexer_prefill()
    test_select_candidate_blocks()
    print("ALL OK")
