"""Tier planning tests: each level must expose the intended mechanisms."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.tiers import OptLevel, plan  # noqa: E402


def test_level_order():
    assert int(OptLevel.CONSERVATIVE) == 0
    assert int(OptLevel.BALANCED) == 1
    assert int(OptLevel.AGGRESSIVE) == 2
    assert int(OptLevel.EXTREME) == 3


def test_level2_is_the_engine_tier():
    t = plan(OptLevel.AGGRESSIVE)
    assert t.keep_weights_quantized
    assert t.async_expert_fetch
    assert t.batched_scatter
    assert t.slot_arena
    assert t.mtp
    assert t.expert_topk in (0, 4)  # 0 = checkpoint top-6


def test_reference_tier_does_not_approximate():
    t = plan(OptLevel.CONSERVATIVE)
    assert t.expert_topk == 0
    assert not t.cuda_graph
    assert not t.mtp


def test_extreme_reduces_topk():
    t = plan(OptLevel.EXTREME)
    assert t.expert_topk and t.expert_topk < 6
