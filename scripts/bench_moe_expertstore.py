"""O4 integration check: MoE with ExpertStore vs direct ShardStore, L0 real weights."""
import sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.config import TinyConfig
from tinyinfer.loading import ExpertStore
from tinyinfer.moe import MoE
from tinyinfer.weights import ShardStore

MODEL = ROOT / "cache" / "weights"
cfg = TinyConfig.from_hf(MODEL)


def timeit(fn, iters=2):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    store = ShardStore(MODEL)
    es = ExpertStore.from_store(store, cache_bytes=2 << 30, policy="arc", vram_slots=0, device="cpu", direct=False)
    moe_direct = MoE(store, 0, cfg)
    moe_es = MoE(store, 0, cfg, expert_store=es, max_bf16_experts=32)

    torch.manual_seed(0)
    x = (torch.randn(4, cfg.dim, dtype=torch.bfloat16) * 0.02)
    with torch.inference_mode():
        y1 = moe_direct.forward(x)
        y2 = moe_es.forward(x)
    d = (y1.float() - y2.float()).abs()
    print("outputs:", tuple(y1.shape), "max_abs_diff:", d.max().item(), "finite:", torch.isfinite(y2.float()).all().item())
    print("gate experts used (direct):", len(moe_direct.ffn._expert_cache if hasattr(moe_direct, 'ffn') else moe_direct._expert_cache))
    print("expert_cache sizes:", len(moe_direct._expert_cache), len(moe_es._expert_cache))
    print("stats:", es.stats()["ram"] if es else {})

    t1 = timeit(lambda: moe_direct.forward(x))
    t2 = timeit(lambda: moe_es.forward(x))
    print(f"latency ms (4 tokens, hot cache): direct={t1:.1f} expertstore={t2:.1f}")


if __name__ == "__main__":
    main()
