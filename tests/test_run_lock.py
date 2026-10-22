"""The machine-wide engine lock must allow exactly one engine at a time."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tinyinfer.runlock import EngineLock, EngineLockBusy

SRC = Path(__file__).resolve().parents[1] / "src"


def unique(tag: str) -> str:
    return f"Local\\tinyinfer-test-{os.getpid()}-{tag}"


def test_acquire_release_roundtrip():
    lock = EngineLock(unique("roundtrip"))
    lock.acquire()
    lock.release()
    lock.acquire()
    lock.release()


def test_second_acquire_in_same_process_fails():
    lock = EngineLock(unique("same")).acquire()
    try:
        with pytest.raises(EngineLockBusy):
            EngineLock(lock.name).acquire()
    finally:
        lock.release()


def test_other_process_holds_lock():
    name = unique("xproc")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(SRC)!r})
        from tinyinfer.runlock import EngineLock
        lock = EngineLock({name!r}).acquire()
        print("HELD", flush=True)
        sys.stdin.readline()
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline().strip()
        assert line == "HELD", f"holder did not start: {line!r}"
        with pytest.raises(EngineLockBusy):
            EngineLock(name).acquire()
    finally:
        proc.stdin.write("\n")
        proc.stdin.flush()
        proc.wait(timeout=60)
    lock = EngineLock(name).acquire()
    lock.release()


def test_different_names_do_not_collide():
    a = EngineLock(unique("a")).acquire()
    b = EngineLock(unique("b")).acquire()
    try:
        assert a.name != b.name
    finally:
        a.release()
        b.release()
