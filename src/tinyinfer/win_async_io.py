"""Windows OVERLAPPED async pread (ctypes): many outstanding ReadFiles on one file.

Buffered readinto() tops out at ~5.3GB/s (page-cache double copy + per-file lock).
This issues N concurrent ReadFile(OVERLAPPED) on a single HANDLE with 4K-aligned
VirtualAlloc buffers and aligned offsets (NO_BUFFERING), letting the NVMe queue fill.

    r = AsyncFileReader(path)          # direct=True by default
    data = r.read_many([(offset, nbytes), ...])
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ULONG_PTR = ctypes.c_size_t
INVALID_HANDLE_VALUE = wt.HANDLE(-1).value
GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_SHARE_READ = 0x00000001
ERROR_IO_PENDING = 997
INFINITE = 0xFFFFFFFF
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
ALIGN = 4096


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", _ULONG_PTR),
        ("InternalHigh", _ULONG_PTR),
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
kernel32.VirtualAlloc.restype = ctypes.c_void_p
kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]


class AsyncFileReader:
    def __init__(self, path: str, direct: bool = True):
        flags = FILE_FLAG_OVERLAPPED | (FILE_FLAG_NO_BUFFERING if direct else 0)
        self.path = str(path)
        self.direct = direct
        self.handle = kernel32.CreateFileW(self.path, GENERIC_READ, FILE_SHARE_READ, None,
                                           OPEN_EXISTING, flags, None)
        if self.handle == INVALID_HANDLE_VALUE:
            raise OSError(f"CreateFileW failed for {self.path} ({ctypes.get_last_error()})")

    def close(self):
        if self.handle != INVALID_HANDLE_VALUE:
            kernel32.CloseHandle(self.handle)
            self.handle = INVALID_HANDLE_VALUE

    def _submit(self, offset: int, nbytes: int):
        prefix = offset & (ALIGN - 1)
        aligned_off = offset - prefix
        aligned_len = ((prefix + nbytes + ALIGN - 1) // ALIGN) * ALIGN
        buf = kernel32.VirtualAlloc(None, aligned_len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not buf:
            raise MemoryError("VirtualAlloc failed")
        ev = kernel32.CreateEventW(None, True, False, None)
        ov = _OVERLAPPED(Offset=aligned_off & 0xFFFFFFFF,
                         OffsetHigh=(aligned_off >> 32) & 0xFFFFFFFF, hEvent=ev)
        ok = kernel32.ReadFile(self.handle, buf, aligned_len, None, ctypes.byref(ov))
        if not ok and ctypes.get_last_error() != ERROR_IO_PENDING:
            err = ctypes.get_last_error()
            kernel32.CloseHandle(ev)
            kernel32.VirtualFree(buf, 0, MEM_RELEASE)
            raise OSError(f"ReadFile failed ({err})")
        return ev, ov, buf, prefix, nbytes

    def _finish(self, t) -> bytes:
        ev, ov, buf, prefix, nbytes = t
        kernel32.WaitForSingleObject(ev, INFINITE)
        done = wt.DWORD(0)
        kernel32.GetOverlappedResult(self.handle, ctypes.byref(ov), ctypes.byref(done), False)
        kernel32.CloseHandle(ev)
        out = ctypes.string_at(buf + prefix, nbytes)
        kernel32.VirtualFree(buf, 0, MEM_RELEASE)
        return out

    def read_many(self, segments) -> list:
        tickets = [self._submit(off, n) for off, n in segments]
        return [self._finish(t) for t in tickets]
