"""Host memory budget: the single place that decides how much RAM may be pinned.

Every large host allocation (the pinned layer slab, the pinned expert cache) asks here
first and gets whatever is safe *right now*, so the same code runs unchanged on a 16GB
laptop, a 47GB workstation or a 512GB server. Nothing in the engine may assume a
machine: if a request cannot be honoured the caller degrades (pageable bytes, streaming
instead of pinning) instead of paging the box into the ground, which is what an
un-budgeted 16GB slab + 6GB pinned cache did on a 47GB box.

The accounting is deliberately stateless about our own allocations: Windows and Linux
both subtract a fresh allocation from "available" immediately, so re-reading available
on every request already accounts for everything we (and every other process) hold.
"""
from __future__ import annotations

import os
import threading

GB = 1 << 30


def _physical_ram() -> int:
    """Physical RAM without importing anything heavy (psutil can fail under pressure)."""
    try:
        if os.name == "nt":
            import ctypes

            class _MEMSTATUS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MEMSTATUS()
            st.dwLength = ctypes.sizeof(_MEMSTATUS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
        else:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


class HostBudget:
    def __init__(self, reserve_gb: float | None = None):
        self._lock = threading.Lock()
        self.total = 0
        self._avail0 = 0
        try:
            import psutil

            vm = psutil.virtual_memory()
            self.total = int(vm.total)
            self._avail0 = int(vm.available)
        except Exception:
            pass
        if not self.total:
            # Under memory pressure an import can fail; fall back to the OS so the
            # budget never silently collapses to zero (which is what turned every slab
            # grant into "0 bytes available" and dropped the engine to per-tensor H2D).
            self.total = _physical_ram()
        # Budget a fixed FRACTION OF PHYSICAL RAM, not "available". On Windows
        # psutil.available excludes the standby list while every shard we read fills it,
        # so after loading the model "available" reads 7GB on a 47GB box and every grant
        # came back short -- the slab fell back to per-tensor copies, the box swapped to
        # C: and decode collapsed to 17s/token. A share of total RAM is stable no matter
        # what the page cache is doing.
        if reserve_gb is not None:
            self.reserve = int(float(reserve_gb) * GB)
        else:
            self.reserve = max(3 * GB, int(self.total * 0.15)) if self.total else 3 * GB
        # what this process may hand out in total (weights + slab + expert cache)
        self.limit = max(0, self.total - self.reserve)
        if os.environ.get("TI_RAM_FRACTION"):
            self.limit = int(self.total * float(os.environ["TI_RAM_FRACTION"]))
        self.granted = 0

    def available(self) -> int:
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:
            return self._avail0

    def claim(self, nbytes: int, real: bool = False) -> bool:
        """Reserve the whole planned footprint up front (one measurement, one decision).

        The planner sizes every buffer from a single measurement taken at start-up. When
        allocation is lazy, other processes eat that memory in the meantime and the
        numbers the plan was built on are gone: grants come back short and the box ends up
        swapping. claim() books the planned bytes once so `room()` no longer has to consult
        live `available()` (which is exactly what made the plan drift).

        `real=True` also allocates and touches the pages so the OS commits them; that is
        the strongest guarantee but it doubles the footprint while the caches are still
        lazily filled, so it is opt-in (TI_CLAIM_REAL=1) until the caches allocate from
        the claimed pool.
        """
        nbytes = max(0, int(nbytes))
        if nbytes <= 0:
            return False
        with self._lock:
            if self.granted + nbytes > self.limit:
                return False
            if real:
                try:
                    got = self._commit(nbytes)
                except MemoryError:
                    got = 0
                if not got:
                    return False
                # book what was really taken, not what was asked for: the caches are
                # then sized from the pages we actually own
                nbytes = int(got)
            self.granted += nbytes
            self._claimed = True
            return True

    def _commit(self, nbytes: int) -> bool:
        """Commit nbytes of address space so no other process can take it.

        Windows VirtualAlloc(MEM_COMMIT) reserves *and* charges the commit limit in
        kernel time (measured: 19GB in 0.000s), unlike bytearray(), which zero-fills and
        costs ~1.7s per 19GB. This is the "claim it up front, in under a second" primitive.
        """
        try:
            import ctypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.VirtualAlloc.restype = ctypes.c_void_p
            k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                         ctypes.c_ulong, ctypes.c_ulong]
            k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
            # VirtualAlloc alone only charges the commit limit; the physical pages stay
            # unowned until written, which is why Task Manager showed a slow climb and the
            # box could still page. Touch one byte per 4KB page, in 2GB batches, so a huge
            # single allocation cannot be cut short under memory pressure; whatever was
            # really taken is what we book. Single-threaded on purpose: this runs once at
            # start-up and correctness matters more than shaving a second.
            n = int(nbytes)
            batch = 2 * (1 << 30)
            taken = 0
            held = getattr(self, "_held", None)
            if held is None:
                held = self._held = []
            while taken < n:
                chunk = min(batch, n - taken)
                sub = k32.VirtualAlloc(None, chunk, 0x1000 | 0x2000, 0x04)
                if not sub:
                    break
                try:
                    buf = (ctypes.c_char * chunk).from_address(sub)
                    for off in range(0, chunk, 4096):
                        buf[off] = b"\x01"
                except Exception:
                    k32.VirtualFree(sub, 0, 0x8000)
                    break
                held.append((k32, sub))
                taken += chunk
            return taken
        except Exception:
            return False

    def room(self) -> int:
        """Bytes we may still hand out from our share of physical RAM.

        Deliberately NOT derived from psutil.available(): on Windows that number drops
        as the file cache fills with shard data we just read, so after loading the model
        it reads 7GB on a 47GB box and every large grant failed (the slab fell back to
        per-tensor copies and the box swapped to C:). Our share of *total* RAM is the
        stable quantity; the standby list is reclaimable by the OS on demand.
        """
        if getattr(self, "_claimed", False):
            # claimed up front: the plan's own arithmetic is authoritative, live
            # available() would just make it drift with whatever else is running
            return max(0, self.limit - self.granted)
        if not self.limit:
            return max(0, self.available() - self.reserve)
        # Both bounds matter: our share caps the total we hand out, and the live
        # available number stops us when other allocations (model weights, page cache)
        # already used the memory. Taking only the share caused an 8GB card to try a
        # 23GB slab and take the box to its page file.
        return max(0, min(self.available() - self.reserve, self.limit - self.granted))

    def grant(self, want: int) -> int:
        """Ask for want bytes; returns how much is safe to take now (0..want)."""
        want = max(0, int(want))
        with self._lock:
            got = min(want, self.room())
            self.granted += got
            return got

    def release(self, nbytes: int) -> None:
        """Give back a grant that was not (fully) used, e.g. a failed allocation."""
        with self._lock:
            self.granted = max(0, self.granted - max(0, int(nbytes)))

    def stats(self) -> dict:
        return {
            "total_gb": round(self.total / GB, 1),
            "avail_gb": round(self.available() / GB, 1),
            "reserve_gb": round(self.reserve / GB, 1),
            "granted_gb": round(self.granted / GB, 1),
        }


_BUDGET: HostBudget | None = None


def budget() -> HostBudget:
    """Process-wide budget (created on first use)."""
    global _BUDGET
    if _BUDGET is None:
        import os

        _BUDGET = HostBudget(os.environ.get("TI_RAM_RESERVE_GB"))
    return _BUDGET
