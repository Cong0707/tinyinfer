"""Tests for the n-gram drafter (O5 speculation)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyinfer.draft import NgramDrafter


def test_repeat_pattern():
    ids = [10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12]
    d = NgramDrafter(max_n=5)
    d.observe(ids)
    draft = d.propose(ids[-4:], 3)  # suffix "14,10,11,12" -> "13,14,..." from the first cycle
    assert draft and draft[0] == 13, draft
    print("repeat pattern ok:", draft)


def test_no_match():
    d = NgramDrafter()
    d.observe([1, 2, 3])
    assert d.propose([99, 98], 4) == []
    print("no match ok")


def test_extend_chain():
    ids = list(range(20))
    d = NgramDrafter(max_n=4)
    d.observe(ids)
    draft = d.propose(ids[-3:-1], 4)  # suffix "17,18" -> "19"
    assert draft and draft[0] == 19, draft
    print("extend ok:", draft)


if __name__ == "__main__":
    test_repeat_pattern()
    test_no_match()
    test_extend_chain()
    print("ALL OK")
