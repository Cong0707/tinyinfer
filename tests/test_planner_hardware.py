"""Planner portability: the same checkpoint on hypothetical machines.

No GPU is touched; each case builds a synthetic Hardware descriptor and checks that the
plan is sane. This is the regression guard for "it must also work on other boxes".
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.resources import Card, Hardware, plan  # noqa: E402

GB = 2**30
# measured geometry of the checkpoint (see model_profile)
MODEL = {
    "layer_bytes": 0.19 * GB,
    "max_layer_bytes": 0.72 * GB,
    "embed_bytes": 1.32 * GB,
    "head_bytes": 1.32 * GB,
    "expert_bytes": 18.8 * 2**20,
    "n_layers": 40,
    "topk": 6,
}
N = 40


def _hw(cards_gb, ram, avail, unified=False):
    cards = [Card(i, "card", gb, gb) for i, gb in enumerate(cards_gb)]
    return Hardware(name="test", vram_gb=sum(cards_gb), ram_gb=ram,
                    ram_avail_gb=avail, unified=unified, cards=cards)


def _plan(cards_gb, ram, avail, unified=False):
    return plan(hw=_hw(cards_gb, ram, avail, unified), n_layers=N, model=MODEL,
                speculation=True)


def test_this_box_shape():
    p = _plan([8.0], 47.0, 35.0)
    assert p.fits
    assert 1 <= p.gpu_layers <= 6          # measured 4 here
    assert p.stream_layers == N - p.gpu_layers
    # A small pool (a few percent of the 240 routable experts) costs more in DMAs and event
    # waits than it saves, so the planner drops it and reinvests in residency (docs/
    # analysis.md 16). Either an absent pool or one that covers a real fraction is fine.
    assert p.expert_slots == 0 or p.expert_slots >= 0.15 * 240
    assert p.cache_bytes > 1 * GB


def test_big_card_is_fully_resident():
    p = _plan([24.0], 64.0, 56.0)
    assert p.fits
    assert p.gpu_layers == N
    assert p.stream_layers == 0
    assert p.expert_slots > 50


def test_dual_card_spreads_layers():
    p = _plan([24.0, 24.0], 128.0, 120.0)
    assert p.fits
    assert p.gpu_layers == N
    used = set(p.layer_card.values())
    assert used == {0, 1}, p.layer_card
    per = {}
    for idx in p.layer_card.values():
        per[idx] = per.get(idx, 0) + 1
    assert all(v > 0 for v in per.values()), per


def test_small_card_is_refused_not_oomed():
    p = _plan([4.0], 16.0, 8.0)
    assert not p.fits
    assert "INSUFFICIENT" in p.reason


def test_six_gb_card_still_works():
    p = _plan([6.0], 24.0, 16.0)
    assert p.fits, p.reason
    assert p.gpu_layers >= 1


def test_cpu_only_keeps_tables_on_host():
    hw = Hardware(name="cpu", vram_gb=0.0, ram_gb=64.0, ram_avail_gb=56.0, cards=[])
    p = plan(hw=hw, n_layers=N, model=MODEL)
    assert p.gpu_layers == 0
    assert p.stream_layers == N
    assert p.head_on_cpu and p.embed_on_cpu
    assert p.cache_bytes > 1 * GB


def test_unified_memory_does_not_double_book():
    # a 48GB unified box: VRAM + host cache must stay inside the pool
    p = _plan([48.0], 48.0, 40.0, unified=True)
    assert p.fits
    total_gb = p.cache_bytes / GB + p.gpu_layers * 0.19 * 1.7
    assert total_gb <= 48.0, total_gb


def test_residency_grows_with_vram():
    """Residency must grow with VRAM, up to the whole model.

    Not strictly monotone at the bottom: the ring depth is chosen from the card size, so a
    6GB card plans a 2-deep ring and an 8GB card a 3-deep one, which can cost one resident
    layer. What must hold is that residency reaches the full model on a big card and never
    collapses to nothing on a small one.
    """
    levels = [_plan([gb], 64.0, 56.0).gpu_layers for gb in (6.0, 8.0, 12.0, 16.0, 24.0)]
    assert levels[0] >= 1, levels
    assert levels[-1] == N, levels
    # beyond the ring-depth step the trend must be monotone
    assert levels[2] <= levels[3] <= levels[4], levels


def test_big_card_covers_all_routable_experts():
    """A 24GB card can hold every expert a token might route to (topk*n_layers=240).

    That is the difference between 5 and 80 tok/s: the expert IO term vanishes. The
    planner is expected to trade the 1.32GB embedding table for the last 43 slots.
    """
    p = _plan([24.0], 64.0, 56.0)
    assert p.fits
    assert p.expert_slots >= 240, p.expert_slots
    assert p.embed_on_cpu, "the table must move to the host to make room for the slots"
