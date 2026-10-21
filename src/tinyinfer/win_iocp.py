"""Deep asynchronous shard IO for Windows: IOCP + FILE_FLAG_NO_BUFFERING.

The expert stream stalls because a decode layer only needs six 18.8MB blobs, so a
thread pool tops out at QD=6 (~6.3GB/s) while the device does 11GB/s. IoRing/IOCP
lets us keep 32+ reads in flight with one thread and no per-read pinned allocation.

APIs (Windows 8+, FILE_FLAG_OVERLAPPED, overlapped buffers must be sector aligned):
    CreateIoCompletionPort, ReadFile(OVERLAPPED), GetQueuedCompletionStatus
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import threading

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wt.DWORD),
        ("OffsetHigh", wt.DWORD),
        ("hEvent", wt.HANDLE),
    ]


GENERIC_READ = 0x80000000
FILE_SHARE_READ = 1
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_OVERLAPPED = 0x40000000
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
PAGE_READWRITE = 0x04
INFINITE = 0xFFFFFFFF
ERROR_IO_PENDING = 997
PAGE = 4096

k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p,
                            wt.DWORD, wt.DWORD, wt.HANDLE]
k32.CreateIoCompletionPort.restype = wt.HANDLE
k32.CreateIoCompletionPort.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.c_size_t, wt.DWORD]
k32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p,
                         ctypes.POINTER(OVERLAPPED)]
k32.GetQueuedCompletionStatus.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD),
                                          ctypes.POINTER(ctypes.c_size_t),
                                          ctypes.POINTER(ctypes.POINTER(OVERLAPPED)),
                                          wt.DWORD]
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
k32.CloseHandle.argtypes = [wt.HANDLE]
k32.GetFileSizeEx.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_longlong)]


class IocpShardReader:
    """One thread, N reads in flight, no per-read pinned allocation."""

    def __init__(self, depth: int = 32, slot_bytes: int = 20 << 20):
        self.depth = int(depth)
        self.slot_bytes = int(slot_bytes)
        self.iocp = k32.CreateIoCompletionPort(wt.HANDLE(-1), None, 0, self.depth)
        if not self.iocp:
            raise OSError(ctypes.get_last_error(), "CreateIoCompletionPort")
        self.slots = [k32.VirtualAlloc(None, self.slot_bytes, MEM_COMMIT | MEM_RESERVE,
                                       PAGE_READWRITE) for _ in range(self.depth)]
        self.ovs = [OVERLAPPED() for _ in range(self.depth)]
        self._base = ctypes.addressof(self.ovs[0])
        self._step = ctypes.sizeof(OVERLAPPED)
        self._index_of = {ctypes.addressof(ov): i for i, ov in enumerate(self.ovs)}
        self.free = list(range(self.depth))
        self.handles: dict[str, int] = {}
        self.sizes: dict[str, int] = {}
        self._lock = threading.Lock()

    def _handle(self, path: str) -> int:
        h = self.handles.get(path)
        if h is not None:
            return h
        h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, None)
        if h in (None, -1, 0xFFFFFFFFFFFFFFFF):
            raise OSError(ctypes.get_last_error(), f"CreateFileW({path})")
        k32.CreateIoCompletionPort(h, self.iocp, 0, 0)
        v = ctypes.c_longlong(0)
        k32.GetFileSizeEx(h, ctypes.byref(v))
        self.handles[path] = h
        self.sizes[path] = int(v.value)
        return h

    def _submit(self, path: str, offset: int, nbytes: int, idx: int) -> None:
        h = self._handle(path)
        ov = self.ovs[idx]
        ov.Internal = 0
        ov.InternalHigh = 0
        ov.Offset = offset & 0xFFFFFFFF
        ov.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
        ov.hEvent = None
        ok = k32.ReadFile(h, self.slots[idx], nbytes, None, ctypes.byref(ov))
        if not ok:
            err = ctypes.get_last_error()
            if err != ERROR_IO_PENDING:
                raise OSError(err, f"ReadFile({path}+{offset})")

    def _pump(self) -> tuple[int, int]:
        n = wt.DWORD(0)
        key = ctypes.c_size_t(0)
        pov = ctypes.POINTER(OVERLAPPED)()
        ok = k32.GetQueuedCompletionStatus(self.iocp, ctypes.byref(n), ctypes.byref(key),
                                           ctypes.byref(pov), INFINITE)
        if not pov:
            raise OSError(ctypes.get_last_error(), "GetQueuedCompletionStatus")
        if not ok:
            raise OSError(ctypes.get_last_error(), "overlapped read failed")
        addr = ctypes.addressof(pov.contents)
        idx = self._index_of.get(addr)
        if idx is None:
            raise RuntimeError(f"completion for unknown OVERLAPPED at {addr:#x} "
                               f"(base {self._base:#x}, step {self._step})")
        return idx, int(n.value)

    def _size(self, path: str) -> int:
        self._handle(path)
        return self.sizes[path]

    def read_many(self, spans: list[tuple[str, int, int]]) -> list[bytes]:
        """spans: [(path, offset, nbytes)]. Returns one bytes object per span.

        NO_BUFFERING needs sector-aligned requests, so each span is widened to page
        boundaries, clamped to the file size, and trimmed on the way out.
        """
        aligned: list[tuple[str, int, int]] = []
        heads: list[tuple[int, int]] = []
        for (path, off, n) in spans:
            a_off = off & ~(PAGE - 1)
            a_end = (off + n + PAGE - 1) & ~(PAGE - 1)
            size = self._size(path)
            a_end = min(a_end, size & ~(PAGE - 1))
            if a_end < off + n:
                # last partial sector of a shard: read the tail through the cache
                a_end = -1
            aligned.append((path, a_off, a_end - a_off if a_end > 0 else 0))
            heads.append((off - a_off, n))
        out: list[bytes | None] = [None] * len(spans)
        pending: list[tuple[int, int]] = []  # (span_index, slot)
        next_i = 0
        done = 0
        while done < len(spans):
            while next_i < len(spans) and self.free:
                path, off, n = aligned[next_i]
                if n == 0:  # tail span: buffered read
                    o0, want = heads[next_i]
                    with open(path, "rb") as f:
                        f.seek(off + o0)
                        out[next_i] = f.read(want)
                    next_i += 1
                    done += 1
                    continue
                if n > self.slot_bytes:
                    raise ValueError(f"span {n} exceeds slot {self.slot_bytes}")
                idx = self.free.pop()
                self._submit(path, off, n, idx)
                pending.append((next_i, idx))
                next_i += 1
            if not pending:
                raise RuntimeError("no IO in flight but spans remain")
            idx, nbytes = self._pump()
            span_i = next(i for (i, s) in pending if s == idx)
            path, off, n = spans[span_i]
            if nbytes < n:
                raise OSError(f"short read {nbytes}/{n} at {path}+{off}")
            head, want = heads[span_i]
            out[span_i] = ctypes.string_at(self.slots[idx] + head, want)
            pending = [(i, s) for (i, s) in pending if s != idx]
            self.free.append(idx)
            done += 1
        return out  # type: ignore[return-value]

    def read_blob(self, blob) -> bytes:
        """Read an expert blob; one ReadFile per contiguous segment."""
        segs = blob.segments
        parts = self.read_many([(str(segs[0].shard) if False else str(getattr(s, "shard")),
                                int(s.file_offset), int(s.nbytes)) for s in segs])
        return b"".join(parts)

    def close(self) -> None:
        for h in self.handles.values():
            k32.CloseHandle(h)
        self.handles.clear()
        for p in self.slots:
            if p:
                k32.VirtualFree(p, 0, 0x8000)
        self.slots = []
        if self.iocp:
            k32.CloseHandle(self.iocp)
            self.iocp = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()