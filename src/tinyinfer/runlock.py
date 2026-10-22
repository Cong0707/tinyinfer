"""One engine per machine.

The engine claims a large pinned host pool plus the whole GPU. Two instances do not
fit: they double the claimed RAM (the box starts swapping) and on Windows a second
big cudaHostAlloc makes later cudaMalloc calls fail. A machine-wide OS lock makes the
second start fail loudly instead of silently starving the first.

Windows uses a named mutex (the handle is the ownership; closing it -- including on
process exit -- releases the name). Other platforms use a non-blocking flock on a
temp file. No lock file is left behind after a crash.
"""
from __future__ import annotations

import os
import tempfile

DEFAULT_NAME = "Local\\tinyinfer-engine"
ERROR_ALREADY_EXISTS = 183


class EngineLockBusy(RuntimeError):
    """Another engine instance already holds the machine-wide lock."""

    def __init__(self, name: str):
        super().__init__(
            f"another tinyinfer engine is already running on this machine "
            f"(lock {name!r}). Stop it first, e.g. "
            f"powershell: Get-Process python | Stop-Process")


class EngineLock:
    """A held named lock; release() or process exit gives it back."""

    def __init__(self, name: str | None = None):
        self.name = name or DEFAULT_NAME
        self._handle = None
        self._file = None

    def acquire(self) -> "EngineLock":
        if self._handle is not None or self._file is not None:
            raise EngineLockBusy(self.name)
        if os.name == "nt":
            self._acquire_win()
        else:
            self._acquire_posix()
        return self

    def release(self) -> None:
        if self._handle is not None:
            k32, handle = self._handle
            self._handle = None
            try:
                k32.CloseHandle(handle)
            except Exception:
                pass
        if self._file is not None:
            try:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    # -- platform bits ----------------------------------------------------
    def _acquire_win(self) -> None:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        handle = k32.CreateMutexW(None, False, self.name)
        err = ctypes.get_last_error()
        if not handle:
            raise OSError(f"CreateMutexW({self.name!r}) failed: {ctypes.WinError(err)}")
        if err == ERROR_ALREADY_EXISTS:
            k32.CloseHandle(handle)
            raise EngineLockBusy(self.name)
        self._handle = (k32, handle)

    def _acquire_posix(self) -> None:
        import fcntl

        path = os.path.join(tempfile.gettempdir(),
                            "tinyinfer-engine-" + str(abs(hash(self.name))) + ".lock")
        f = open(path, "a+")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            raise EngineLockBusy(self.name)
        self._file = f

    def __enter__(self) -> "EngineLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
