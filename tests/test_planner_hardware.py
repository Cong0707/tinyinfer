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
    assert p.expert_slots >= 2
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
    # and the fallback still keeps a usable expert slot count
    assert p.expert_slots >= 2


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
    levels = [_plan([gb], 64.0, 56.0).gpu_layers for gb in (6.0, 8.0, 12.0, 16.0, 24.0)]
    assert levels == sorted(levels), levels
    assert levels[0] >= 1 and levels[-1] == N
