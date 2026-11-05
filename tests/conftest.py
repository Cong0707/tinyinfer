"""Shared test configuration.

The real checkpoint moved to E:\Models (the tinyinfer cache was cleared at the
user's request); tests that need the real config.json resolve it here instead of
hard-coding the old cache path.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(os.environ.get("TINYINFER_MODEL_DIR", r"E:/Models/DeepSeek-V4.1-Flash-1017"))


def model_dir() -> Path:
    """Real model directory: env override, E:\Models, then the legacy cache."""
    if (DEFAULT_MODEL / "config.json").exists():
        return DEFAULT_MODEL
    return ROOT / "cache" / "weights"


def _has_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


HAS_CUDA = _has_cuda()

# Cases that build a kernel or allocate device memory. They are skipped rather than
# failed on a CPU-only runner: CI runs the arithmetic half of this repo everywhere and
# the kernel half only where a GPU is attached, and a suite that reports 12 failures on
# every laptop is one people learn to ignore.
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
