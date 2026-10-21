"""The planner matrix script is the portability contract; keep its output machine-readable."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "plan_matrix.py"


def test_matrix_json_is_well_formed():
    out = subprocess.run([sys.executable, str(SCRIPT), "--json"],
                         capture_output=True, text=True, timeout=300,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    assert out.returncode == 0, out.stderr[-800:]
    rows = json.loads(out.stdout)
    assert isinstance(rows, list) and len(rows) >= 8
    names = [r["machine"] for r in rows]
    assert any("8GB" in n for n in names)
    assert any("24GB" in n for n in names)
    for r in rows:
        for key in ("machine", "resident", "stream", "slots", "chunk", "cache_gb",
                    "floor_s", "tok_per_s", "fits", "gpu"):
            assert key in r, (r["machine"], key)
        assert r["resident"] + r["stream"] == 40
    # the 8GB box is expected to fit; the card sizes in the table are the contract,
    # the fits decision is exercised by tests/test_planner_hardware.py
    by_name = {r["machine"]: r for r in rows}
    box = next(r for n, r in by_name.items() if "8GB" in n)
    assert box["fits"] and box["resident"] >= 1
