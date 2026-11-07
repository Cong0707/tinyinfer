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
import sys
import threading

GB = 1 << 30


def available_commit() -> int:
    """Bytes the OS can still commit, i.e. what VirtualAlloc/cudaHostAlloc may charge.

    On Windows this is ullAvailPageFile (physical + pagefile minus what is already
    committed), and it is the real ceiling for pinned host memory -- not
    ullAvailPhys. An engine that sizes its expert cache from free *physical* RAM can
    ask for more pinned memory than the commit limit allows, which fails deep inside
    the driver as "cuMemSetAccess ... out of memory" with plenty of RAM showing free.
    Returns 0 when the platform cannot report it (callers then skip the clamp).
    """
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
                return int(st.ullAvailPageFile)
        else:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


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


_CUDART = None


def _cudart():
    """Locate the CUDA runtime that ships with torch (for cudaHostAlloc)."""
    global _CUDART
    if _CUDART is not False and _CUDART is not None:
        return _CUDART
    if _CUDART is False:
        return None
    try:
        import ctypes
        import glob
        import os

        import torch

        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        for path in glob.glob(os.path.join(lib_dir, "cudart*.dll")):
            try:
                lib = ctypes.CDLL(path)
                lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_size_t, ctypes.c_uint]
                lib.cudaHostAlloc.restype = ctypes.c_int
                lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
                lib.cudaFreeHost.restype = ctypes.c_int
                _CUDART = lib
                return lib
            except Exception:
                continue
    except Exception:
        pass
    _CUDART = False
    return None


def _touch_pages(buf, n: int) -> None:
    """Commit every 4KB page of a host allocation right now.

    The C loop below is a single strided numpy write, which is ordered and therefore
    single-threaded: on a 300GB claim that is 78M page faults on one core (measured:
    the whole start-up sat in this function, one core at 100%, for tens of seconds).
    The faults themselves are independent per range, so split the buffer into
    cpu_count() chunks and touch them from threads -- the page-fault work is in the
    kernel, so it scales with cores.

    MADV_POPULATE_WRITE (Linux >= 5.14) does the same thing in one syscall; try it
    first and fall back to the threaded walk.
    """
    import os as _os
    try:
        import ctypes as _ct

        _libc = _ct.CDLL("libc.so.6", use_errno=True)
        _rc = _libc.madvise(_ct.c_void_p(_ct.addressof(_ct.c_char.from_buffer(buf))),
                            _ct.c_size_t(n), 23)  # MADV_POPULATE_WRITE
        if _rc == 0:
            return
    except Exception:
        pass

    ncpu = max(1, min(32, (_os.cpu_count() or 4)))
    if n < (1 << 20) or ncpu == 1:
        try:
            import numpy as np

            np.frombuffer(buf, dtype=np.uint8)[::4096] = 1
            return
        except Exception:
            pass
        for off in range(0, n, 4096):
            buf[off] = b"\x01"
        return

    step = (n + ncpu - 1) // ncpu
    step -= step % 4096
    ranges = [(o, min(n, o + step)) for o in range(0, n, step)]

    def _do(lo_hi):
        lo, hi = lo_hi
        try:
            import numpy as _np

            _np.frombuffer(buf, dtype=_np.uint8, offset=lo, count=hi - lo)[::4096] = 1
        except Exception:
            for off in range(lo, hi, 4096):
                buf[off] = b"\x01"

    from concurrent.futures import ThreadPoolExecutor as _TPE

    with _TPE(max_workers=ncpu, thread_name_prefix="tinyinfer-touch") as _ex:
        list(_ex.map(_do, ranges))


class HostPool:
    """One big host allocation that every pageable buffer carves from.

    Rationale: the engine knows its whole footprint at start-up. Allocating it once and
    sub-allocating keeps every byte inside memory the process already owns, so a later
    cache fill cannot fail (or push the box to the page file) because some other program
    took the RAM in between -- the failure mode seen in practice.

    Also usable as a pinned pool by passing the buffer reported by a CUDA host alloc.
    """

    def __init__(self, total: int):
        self.total = max(0, int(total))
        self._view = None
        self._off = 0
        self._lock = threading.Lock()
        self._owner = None

    @classmethod
    def from_claimed(cls, buffer, nbytes: int) -> "HostPool":
        pool = cls(nbytes)
        view = memoryview(buffer)
        try:
            view = view.cast("B")
        except (TypeError, ValueError):
            pass
        pool._view = view[: int(nbytes)]
        pool._owner = buffer          # keep the allocation alive
        # Absolute address of byte 0: a tensor built from a pool slice can be located by
        # data_ptr() - _base, which is what the flat-streaming setup uses to prove a whole
        # layer already lies in one contiguous region and skip the copy.
        try:
            import ctypes

            pool._base = int(ctypes.addressof(buffer))
        except Exception:
            pool._base = None
        return pool

    def slice_at(self, offset: int, n: int):
        """Pool bytes [offset, offset+n) as a memoryview, or None when out of range."""
        if self._view is None or offset < 0 or n <= 0 or offset + n > self.total:
            return None
        return self._view[offset:offset + n]

    @property
    def used(self) -> int:
        return self._off

    @property
    def free(self) -> int:
        return max(0, self.total - self._off)

    def alloc(self, n: int, align: int = 4096) -> memoryview | None:
        """Carve n bytes (4096-aligned by default). Returns None when full."""
        n = int(n)
        if n <= 0 or self._view is None:
            return None
        with self._lock:
            start = (self._off + align - 1) // align * align
            if start + n > self.total:
                return None
            self._off = start + n
            return self._view[start:start + n]

    def tensor(self, n: int, align: int = 4096):
        """Carve n bytes out of the pool as a writable uint8 torch tensor.

        The pool is created by cudaHostAlloc, so the slice is pinned and can be used
        as the source of an async H2D. torch.frombuffer keeps the buffer alive itself,
        but the memoryview is stored on the tensor too so the export cannot be dropped
        while the layer still points at it.
        """
        mv = self.alloc(n, align=align)
        if mv is None:
            return None
        import torch

        t = torch.frombuffer(mv, dtype=torch.uint8)
        t._tiny_pool_mv = mv
        return t

    def reset(self) -> None:
        with self._lock:
            self._off = 0


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
            # Return the bytes we really own (callers size their buffers from this and
            # copy into the claimed range; a bare True silently became 1 byte).
            return nbytes

    def _commit(self, nbytes: int) -> bool:
        """Commit nbytes of address space so no other process can take it.

        Windows VirtualAlloc(MEM_COMMIT) reserves *and* charges the commit limit in
        kernel time (measured: 19GB in 0.000s), unlike bytearray(), which zero-fills and
        costs ~1.7s per 19GB. This is the "claim it up front, in under a second" primitive.
        """
        if sys.platform != "win32":
            return self._commit_posix(nbytes)
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
            # One contiguous block, retried at half size: the cache slices its entries
            # out of this single range, so it must not be a chain of separate mappings.
            n = int(nbytes)
            held = getattr(self, "_held", None)
            if held is None:
                held = self._held = []
            # Preferred path: CUDA host (pinned) memory. It is DMA-able (the layer slab and
            # the expert cache both feed async H2D from it), and pinned pages cannot be
            # paged out at all -- which is exactly what "claimed" has to mean. Measured:
            # cudaHostAlloc(8GB) returns in 0.6s.
            pinned = None
            try:
                pinned = _cudart()
            except Exception:
                pinned = None
            if pinned is not None and n >= (256 << 20):
                # Retry at half size before giving up: cudaHostAlloc fails as a whole when
                # the request is larger than what the driver will pin, and falling
                # straight to ordinary pages would hand the cache memory the OS is free
                # to page out -- exactly the failure this pool exists to prevent. A
                # smaller pinned pool still is DMA-able; a larger pageable one is not.
                # Step down by 10% rather than halving: the driver's pinned ceiling sits
                # somewhere below physical RAM, and halving throws away up to half of it
                # (28GB refused -> 14GB taken, when 26GB would have worked). Ten percent
                # converges in a handful of tries and keeps the expert cache, which is
                # sized from what we actually get, as large as the driver allows.
                want = int(n)
                while want >= (256 << 20):
                    ptr = ctypes.c_void_p()
                    rc = pinned.cudaHostAlloc(ctypes.byref(ptr), want, 0)
                    if rc == 0 and ptr.value:
                        # No VirtualLock here: cudaHostAlloc pages are already
                        # non-pageable, and locking 20GB into the working set only
                        # burns the PFN database / non-paged quota, which on Windows
                        # showed up as later cudaMalloc calls failing with OOM.
                        held = getattr(self, "_held", None)
                        if held is None:
                            held = self._held = []
                        view = (ctypes.c_char * want).from_address(ptr.value)
                        # commit the physical pages NOW: cudaHostAlloc reserves the
                        # range, the pages land on first touch
                        _touch_pages(view, want)
                        held.append(("pinned", pinned, ptr.value, want, view))
                        self._claim_buf = view
                        self._claim_bytes = want
                        self._claim_pinned = True
                        if want < n:
                            print(f"[tinyinfer] host pool: cudaHostAlloc shrank "
                                  f"{n/2**30:.1f}GB -> {want/2**30:.1f}GB", flush=True)
                        return want
                    # 10% steps, not halving: the driver's pinned ceiling sits somewhere
                    # below physical RAM, and halving throws away up to half of it (28GB
                    # refused -> 14GB taken when 26GB would have worked). The expert
                    # cache is sized from what we actually get, so a better probe is
                    # worth more than a fast one.
                    want = int(want * 0.9)
            while n >= (256 << 20):
                # Prefer large pages: their pages cannot be trimmed by the memory
                # manager, which is the only way to keep the cache resident for the
                # whole run (normal pages were observed leaving the working set mid-run
                # even after VirtualLock). Needs SeLockMemoryPrivilege (admin) and a
                # size that is a multiple of the 2MB large-page granularity; fall back
                # to ordinary committed pages when the privilege is absent.
                try:
                    k32.GetLargePageMinimum.restype = ctypes.c_size_t
                    gran = int(k32.GetLargePageMinimum() or 0)
                except Exception:
                    gran = 0
                ptr = 0
                if gran:
                    big = (n // gran) * gran
                    if big >= gran:
                        ptr = k32.VirtualAlloc(None, big,
                                               0x20000000 | 0x1000 | 0x2000, 0x04)
                if not ptr:
                    ptr = k32.VirtualAlloc(None, n, 0x1000 | 0x2000, 0x04)
                if ptr:
                    try:
                        buf = (ctypes.c_char * n).from_address(ptr)
                        _touch_pages(buf, n)
                    except Exception:
                        k32.VirtualFree(ptr, 0, 0x8000)
                        ptr = None
                if ptr:
                    # hold the array OBJECT, not just the pointer: after a pointer is
                    # stored the underlying pages can be reclaimed (observed as the working
                    # set falling from 22.7GB to 11.1GB mid-run)
                    # Lock the pages in the working set as well: touching alone makes
                    # them resident, but the OS may still trim them under pressure, which
                    # showed up as the working set falling 22.6GB -> 10.5GB mid-run. A
                    # locked page cannot be trimmed, which is what "claimed" has to mean
                    # for the cache to stay in RAM.
                    try:
                        k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                        k32.VirtualLock.restype = ctypes.c_int
                        k32.VirtualLock(ptr, n)
                    except Exception:
                        pass
                    held.append((k32, ptr, n, buf))
                    self._claim_buf = buf
                    self._claim_bytes = n
                    return n
                n //= 2
            return 0
        except Exception:
            return False

    def _commit_posix(self, nbytes: int) -> int:
        """mmap + touch: the POSIX counterpart of VirtualAlloc + faulting the pages in.

        Same contract as the Windows path: one contiguous anonymous mapping, every 4KB
        page written so the pages are really owned, halving on failure, and the bytes that
        were actually taken are what gets booked. mlock is attempted but not required --
        a container or an ordinary user shell usually has RLIMIT_MEMLOCK far below what a
        model slab needs, and refusing the pool over that would be worse than a pool the
        kernel may page out under pressure.
        """
        import ctypes
        import mmap

        n = int(nbytes)
        held = getattr(self, "_held", None)
        if held is None:
            held = self._held = []
        # 一次 mmap 到位。失败就直接报出来，不要减半重试：300GB 拿不到时
        # 150GB 也拿不到，而每次重试都要重新 mmap + touch，纯属浪费时间。
        # 池子小了引擎会静默降级（缓存变小、每 token 重新读盘），那种"能跑但慢
        # 十倍"的结果比启动失败更难查。
        try:
            buf = mmap.mmap(-1, n)
        except (OSError, MemoryError, ValueError) as exc:
            raise MemoryError(
                f"cannot reserve {n / (1 << 30):.1f}GB host pool: {exc}. "
                f"Lower TI_HOST_POOL_GB / TI_RAM_FRACTION, or free memory."
            ) from exc
        # 逐页 touch 的唯一目的是"保证这些页真的属于我"，代价是内核要在启动时
        # 做 n/4KB 次缺页（300GB = 7800 万次），即使分到多线程也是十几秒的纯等待，
        # 而此刻 GPU 已经在空转。池子是个马上要被写满的缓存：专家读进来就写在
        # 上面，页迟早会被 fault 进来，所以默认交给内核按需填充（mmap 本来就是
        # lazy）。要"启动即锁定"用 TI_CLAIM_TOUCH=1。
        if __import__("os").environ.get("TI_CLAIM_TOUCH", "0") == "1":
            try:
                _touch_pages(buf, n)
            except Exception as exc:
                try:
                    buf.close()
                except Exception:
                    pass
                raise MemoryError(f"cannot fault in {n / (1 << 30):.1f}GB pool: {exc}") from exc
        if True:
            # 不 mlock。mlock 要求内核把每一页都装进物理内存并锁住，280GB 意味着
            # 一次强制 populate 7000 万页 —— 这是启动单核 100% 的真正来源，
            # 而且默认的 RLIMIT_MEMLOCK 通常只有几十 MB 到几 GB，280GB 必然失败，
            # 只是失败之前已经白干了。池子里的页在被写时会自然 fault 进来，锁不锁
            # 对正确性没有影响；真要锁用 TI_CLAIM_MLOCK_GB 指定一个合理上限。
            _mlock_gb = __import__("os").environ.get("TI_CLAIM_MLOCK_GB")
            if _mlock_gb:
                try:
                    libc = ctypes.CDLL("libc.so.6", use_errno=True)
                    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
                    _mn = min(n, int(float(_mlock_gb) * (1 << 30)))
                    libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(_mn))
                except Exception:
                    pass
            held.append(("mmap", buf, n))
            self._claim_buf = buf
            self._claim_bytes = n
            self._claim_pinned = False
            return n
        return 0

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
