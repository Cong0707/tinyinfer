"""Windows OVERLAPPED async pread (ctypes): many outstanding ReadFiles on one file.

Buffered readinto() tops out at ~5.3GB/s (page-cache double copy + per-file lock).
This issues N concurrent ReadFile(OVERLAPPED) on a single HANDLE, which lets the
NVMe queue fill (target 8-11 GB/s on a PCIe5 x4 SSD).

API:
    r = AsyncFileReader(path)
    r.submit(offset, nbytes) -> ticket       # queue one read
    r.complete(ticket) -> bytearray          # wait + return data
    r.read_many(segments) -> list[bytearray] # submit all, then complete in order
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
INVALID_HANDLE_VALUE = wt.HANDLE(-1).value
GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_SHARE_READ = 0x00000001
ERROR_IO_PENDING = 997
INFINITE = 0xFFFFFFFF


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wt.DWORD),
        ("OffsetHigh", wt.DWORD),
        ("hEvent", wt.HANDLE),
    ]


kernel32.CreateFileW.restype = wt.HANDLE
kernel32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p,
                                 wt.DWORD, wt.DWORD, wt.HANDLE]
kernel32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p,
                              ctypes.POINTER(_OVERLAPPED)]
kernel32.GetOverlappedResult.argtypes = [wt.HANDLE, ctypes.POINTER(_OVERLAPPED),
                                         ctypes.POINTER(wt.DWORD), wt.BOOL]
kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wt.BOOL, wt.BOOL, wt.LPCWSTR]
kernel32.CreateEventW.restype = wt.HANDLE
kernel32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
kernel32.CloseHandle.argtypes = [wt.HANDLE]


class AsyncFileReader:
    def __init__(self, path: str, max_outstanding: int = 16, direct: bool = False):
        flags = FILE_FLAG_OVERLAPPED | (FILE_FLAG_NO_BUFFERING if direct else 0)
        self.path = str(path)
        self.handle = kernel32.CreateFileW(self.path, GENERIC_READ, FILE_SHARE_READ, None,
                                           OPEN_EXISTING, flags, None)
        if self.handle == INVALID_HANDLE_VALUE:
            raise OSError(f"CreateFileW failed for {self.path}")
        self.direct = direct
        self.max_outstanding = max(1, max_outstanding)
        self._tickets = []

    def close(self):
        for ov, buf, ev in self._tickets:
            kernel32.CloseHandle(ev)
        self._tickets.clear()
        if self.handle != INVALID_HANDLE_VALUE:
            kernel32.CloseHandle(self.handle)
            self.handle = INVALID_HANDLE_VALUE

    def submit(self, offset: int, nbytes: int) -> tuple:
        buf = ctypes.create_string_buffer(nbytes)  # zeroed, page aligned for direct
        ev = kernel32.CreateEventW(None, True, False, None)
        ov = _OVERLAPPED(Offset=offset & 0xFFFFFFFF, OffsetHigh=(offset >> 32) & 0xFFFFFFFF, hEvent=ev)
        ok = kernel32.ReadFile(self.handle, buf, nbytes, None, ctypes.byref(ov))
        if not ok:
            err = ctypes.get_last_error()
            if err != ERROR_IO_PENDING:
                kernel32.CloseHandle(ev)
                raise OSError(f"ReadFile failed ({err})")
        t = (ov, buf, ev)
        self._tickets.append(t)
        return t

    def complete(self, ticket) -> bytes:
        ov, buf, ev = ticket
        kernel32.WaitForSingleObject(ev, INFINITE)
        done = wt.DWORD(0)
        kernel32.GetOverlappedResult(self.handle, ctypes.byref(ov), ctypes.byref(done), False)
        kernel32.CloseHandle(ev)
        self._tickets.remove(ticket)
        return bytes(buf.raw[: done.value])

    def read_many(self, segments) -> list[bytes]:
        tickets = [self.submit(off, n) for off, n in segments]
        return [self.complete(t) for t in tickets]
