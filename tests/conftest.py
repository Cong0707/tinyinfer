"""Shared test configuration.

The real checkpoint moved to E:\Models (the tinyinfer cache was cleared at the
user's request); tests that need the real config.json resolve it here instead of
hard-coding the old cache path.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(os.environ.get("TINYINFER_MODEL_DIR", r"E:/Models/DeepSeek-V4.1-Flash-1017"))


def model_dir() -> Path:
    """Real model directory: env override, E:\Models, then the legacy cache."""
    if (DEFAULT_MODEL / "config.json").exists():
        return DEFAULT_MODEL
    return ROOT / "cache" / "weights"
