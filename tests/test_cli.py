"""CLI smoke tests with the in-process mock backend (no checkpoint needed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer import cli  # noqa: E402


def test_generate_mock(capsys):
    rc = cli.main(["generate", "hello", "--backend", "mock", "--max-tokens", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip(), "mock backend produced no text"


def test_generate_mock_json(capsys):
    rc = cli.main(["generate", "hello", "--backend", "mock", "--max-tokens", "3", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage" in out or "{" in out
