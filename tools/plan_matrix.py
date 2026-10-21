"""Print the planner's decision across a matrix of hypothetical machines.

Nothing here touches a GPU: it is the portability check for "it must also work on other
boxes". The last two columns apply the IO arithmetic (docs/lower_bounds.md) so the
predicted decode floor can be compared with a real run on that hardware.

Run with PYTHONPATH=src.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.resources import Card, Hardware, plan  # noqa: E402

GB = 2**30
# measured geometry of this checkpoint (see model_profile)
MODEL = {
    "layer_bytes": 0.19 * GB,
    "max_layer_bytes": 0.72 * GB,
    "embed_bytes": 1.32 * GB,
    "head_bytes": 1.32 * GB,
    "expert_bytes": 18.8 * 2**20,
    "n_layers": 40,
    "topk": 6,
}
H2D = 26.5e9       # pinned host->device, measured
COMPUTE_S = 0.012  # 40 layers x ~0.3ms, one token, fully resident
DISK = 4.5e9       # checkpoint volume at its best, measured
TOP6 = 6
MACHINES = [
    ("RTX 5060 8GB / 47GB (this box)", [8.0], 47.0, 35.0, False),
    ("RTX 3060 12GB / 32GB", [12.0], 32.0, 24.0, False),
    ("RTX 4090 24GB / 64GB", [24.0], 64.0, 56.0, False),
    ("RTX 5090 32GB / 128GB", [32.0], 128.0, 120.0, False),
    ("A100 80GB / 256GB", [80.0], 256.0, 240.0, False),
    ("2x RTX 4090 48GB / 128GB", [24.0, 24.0], 128.0, 120.0, False),
    ("M3 Max 48GB unified", [48.0], 48.0, 40.0, True),
    ("GTX 1650 4GB / 16GB", [4.0], 16.0, 10.0, False),
    ("CPU only", [], 64.0, 56.0, False),
]


def floor_seconds(p):
    """Bytes a token must move, divided by the measured rates."""
    stream_b = p.stream_layers * MODEL["layer_bytes"]
    experts_b = TOP6 * MODEL["n_layers"] * MODEL["expert_bytes"]
    possible = TOP6 * MODEL["n_layers"]
    covered = min(1.0, p.expert_slots / possible)
    return stream_b / H2D + experts_b * (1.0 - covered) / DISK


print(f"{'machine':32s} {'res':>3s} {'str':>3s} {'slots':>5s} {'chunk':>5s} "
      f"{'cache':>6s} {'floor s':>8s} {'tok/s':>6s}  fit")
for name, cards_gb, ram, avail, unified in MACHINES:
    cards = [Card(i, "card", gb, gb) for i, gb in enumerate(cards_gb)]
    hw = Hardware(name=name, vram_gb=sum(cards_gb), ram_gb=ram, ram_avail_gb=avail,
                  unified=unified, cards=cards)
    p = plan(hw=hw, n_layers=40, model=MODEL, speculation=True)
    # A fully covered expert set has no IO left, so the box becomes compute-bound:
    # 40 layers x ~0.3ms of fp8/fp4 GEMM at M=1 is the other floor. A CPU-only box has
    # no GPU floor at all; it runs the host path (very slowly) rather than refusing.
    if not cards:
        t = float("nan")
    elif p.fits:
        t = max(floor_seconds(p), COMPUTE_S)
    else:
        t = float("inf")
    print(f"{name:32s} {p.gpu_layers:3d} {p.stream_layers:3d} {p.expert_slots:5d} "
          f"{p.prefill_chunk:5d} {p.cache_bytes / GB:5.1f}G {t:8.2f} {1.0 / t:6.1f}  "
          f"{'cpu' if not cards else ('yes' if p.fits else 'NO ')}")
