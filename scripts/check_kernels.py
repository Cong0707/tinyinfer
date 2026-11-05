#!/usr/bin/env python
"""Run every native-kernel conformance case. Exits non-zero on any failure."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from tinyinfer import kcheck  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(kcheck.run_all())
