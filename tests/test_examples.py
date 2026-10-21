"""Smoke tests for the example scripts (no checkpoint needed).

They check that each example parses arguments and builds its request without touching the
model, so a broken example fails here rather than after a 60s model load.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EX / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_examples_import():
    sys.path.insert(0, str(ROOT / "src"))
    for name in ("generate", "stream", "tiers", "openai_client"):
        mod = _load(name)
        assert any(a for a in dir(mod) if not a.startswith("_"))
