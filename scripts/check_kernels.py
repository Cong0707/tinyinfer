#!/usr/bin/env python
"""Run every native-kernel conformance case. Exits non-zero on any failure."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from tinyinfer import kcheck  # noqa: E402

if __name__ == '__main__':
    import os

    # The planner cases need a checkpoint to size things from; without one they report
    # as skipped instead of failing, so the suite still runs on a bare checkout.
    if not os.environ.get('TI_MODEL'):
        for cand in ('../models/DeepSeek-V4.1-Flash-1017',
                     'E:/models/DeepSeek-V4.1-Flash-1017'):
            p = (pathlib.Path(__file__).resolve().parent / cand).resolve()
            if p.is_dir():
                os.environ['TI_MODEL'] = str(p)
                break
    raise SystemExit(kcheck.run_all())
