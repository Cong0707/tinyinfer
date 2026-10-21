"""Print the planner's decision across a matrix of hypothetical machines.

Nothing here touches a GPU: it is the portability check for "it must also work on other
boxes". Run with PYTHONPATH=src.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.resources import Card, Hardware, plan  # noqa: E402

GB = 2**30
# measured checkpoint geometry (see model_profile): fp8 non-expert layer 0.19GB,
# largest streamed layer 0.72GB, embed/head 1.32GB each, expert blob 18.8MB
MODEL = {
    "layer_bytes": 0.19 * GB,
    "max_layer_bytes": 0.72 * GB,
    "embed_bytes": 1.32 * GB,
    "head_bytes": 1.32 * GB,
    "expert_bytes": 18.8 * 2**20,
    "n_layers": 40,
}
MACHINES = [
    # name, cards (total_gb each), ram_gb, ram_avail_gb, unified
    ("RTX 5060 8GB / 47GB (this box)", [8.0], 47.0, 35.0, False),
    ("RTX 3060 12GB / 32GB", [12.0], 32.0, 24.0, False),
    ("RTX 4090 24GB / 64GB", [24.0], 64.0, 56.0, False),
    ("RTX 5090 32GB / 128GB", [32.0], 128.0, 120.0, False),
    ("A100 80GB / 256GB", [80.0], 256.0, 240.0, False),
    ("2x RTX 4090 48GB / 128GB", [24.0, 24.0], 128.0, 120.0, False),
    ("M3 Max 48GB unified", [48.0], 48.0, 40.0, True),
    ("GTX 1650 4GB / 16GB", [4.0], 16.0, 10.0, False),
    ("CPU only (no CUDA)", [], 64.0, 56.0, False),
]
print(f"{'machine':34s} {'resident':>8s} {'stream':>6s} {'slots':>5s} {'chunk':>5s} {'cache GB':>8s}  head")
for name, cards_gb, ram, avail, unified in MACHINES:
    cards = [Card(i, "card", gb, gb) for i, gb in enumerate(cards_gb)]
    hw = Hardware(name=name, vram_gb=sum(cards_gb), ram_gb=ram, ram_avail_gb=avail,
                  unified=unified, cards=cards)
    p = plan(hw=hw, n_layers=MODEL["n_layers"], model=MODEL, speculation=True)
    print(f"{name:34s} {p.gpu_layers:8d} {p.stream_layers:6d} {p.expert_slots:5d} "
          f"{p.prefill_chunk:5d} {p.cache_bytes / GB:8.1f}  "
          f"{'cpu' if p.head_on_cpu else 'gpu'}")
    if p.layer_card and len(cards_gb) > 1:
        per = {}
        for layer, idx in p.layer_card.items():
            per[idx] = per.get(idx, 0) + 1
        print(f"{'':34s}   per-card layers {per}  slots {p.card_slots}")
