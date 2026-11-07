"""Hand-written decode kernels: nvcc -> cubin -> CUDA driver API.

No torch C++ extension build. scripts/build_native.ps1 runs nvcc (under the MSVC
environment that nvcc insists on) and this module loads the cubin with nvcuda.dll, so
changing a kernel is one nvcc call instead of a rebuild of torch.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_DIR = Path(__file__).with_name("build")
_MODULES: dict[str, int] = {}
_KERNELS: dict[tuple[str, str], int] = {}
_LIB = None


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    # The driver library has a different name per platform, and it is the only file name
    # in this module that is platform-specific: hardcoding nvcuda.dll made every kernel
    # lookup fail on Linux with a bare OSError, which the callers' try/except turns into
    # "silently use torch instead" -- correct output, 5-10x slower, no message.
    if sys.platform == "win32":
        lib = ctypes.CDLL("nvcuda.dll")
    else:
        last = None
        for _cand in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(_cand)
                break
            except OSError as exc:  # driver installed under another soname
                last = exc
        else:
            raise RuntimeError("libcuda not found (is the NVIDIA driver installed?)") from last
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    lib.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.cuModuleLoad.restype = ctypes.c_int
    lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                        ctypes.c_char_p]
    lib.cuModuleGetFunction.restype = ctypes.c_int
    lib.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                                   ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                   ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_void_p)]
    lib.cuLaunchKernel.restype = ctypes.c_int
    # opt-in dynamic shared memory above the 48KB default. Marlin asks for the whole
    # 99KB this card allows, and a launch with more than 48KB of dynamic smem fails
    # with CUDA_ERROR_INVALID_VALUE unless the function's attribute was raised first.
    lib.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.cuFuncSetAttribute.restype = ctypes.c_int
    rc = lib.cuInit(0)
    if rc != 0:
        raise RuntimeError(f"cuInit failed: {rc}")
    # the driver API needs a current context: torch's primary context is created lazily,
    # and without it cuModuleLoad returns CUDA_ERROR_INVALID_CONTEXT (201)
    import torch

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1, device="cuda")
    _LIB = lib
    return lib


def module(name: str) -> int:
    mod = _MODULES.get(name)
    if mod is not None:
        return mod
    lib = _lib()
    path = _DIR / f"{name}.cubin"
    if not path.exists():
        # the cubins are build products, not sources: they are specific to one GPU
        # architecture, so the repo ships only the .cu files
        raise FileNotFoundError(
            f"{path} is missing. The kernels are compiled per architecture: run "
            f"scripts/build_native.ps1 on Windows, or nvcc -cubin -arch=sm_<xx> in "
            f"src/tinyinfer/native")
    handle = ctypes.c_void_p()
    rc = lib.cuModuleLoad(ctypes.byref(handle), str(path).encode())
    if rc != 0:
        raise RuntimeError(f"cuModuleLoad({path}) failed: {rc}")
    _MODULES[name] = handle
    return handle


def kernel(name: str, func: str):
    key = (name, func)
    fn = _KERNELS.get(key)
    if fn is not None:
        return fn
    lib = _lib()
    handle = ctypes.c_void_p()
    rc = lib.cuModuleGetFunction(ctypes.byref(handle), module(name), func.encode())
    if rc != 0:
        raise RuntimeError(f"cuModuleGetFunction({name}, {func}) failed: {rc}")
    _KERNELS[key] = handle
    return handle


def available(name: str, func: str | None = None) -> bool:
    """True when the cubin is present and, if given, the kernel symbol resolves."""
    try:
        if func is not None:
            kernel(name, func)
        return (_DIR / f"{name}.cubin").exists()
    except Exception:
        return False


# Fast path: a bound launcher for one (kernel, signature, geometry). Building the
# ctypes arg array costs more than the launch itself (measured: launch() 17.3us vs
# torch.ops._C 8.3us for the same 4096-wide rms_norm), because every call re-tests each
# argument's type and re-allocs the pointer array. A BoundLaunch caches the geometry and
# the argument kinds, so the steady state is one list comprehension plus cuLaunchKernel.
import threading as _threading

# (stream_handle, c_void_p, Stream object) per thread. Keeping the Stream alive matters:
# the handle is a raw address into it.
_STREAM_CACHE: dict[int, tuple[int, object, object]] = {}
# 每次 launch 回查 torch.cuda.current_stream() 的开销在 launch 热路径上很显著；
# 引擎内部不切换 stream，默认直接用缓存。外部调用者若自己切流，置
# TI_STREAM_CHECK=1 打开逐次校验。
_STREAM_CHECK = __import__("os").environ.get("TI_STREAM_CHECK") == "1"


class BoundLaunch:
    """A kernel + signature + launch geometry, pre-resolved for repeated calls."""

    __slots__ = ("_fn", "_grid", "_block", "_shared", "_kinds", "_n", "_raw", "_ptrs")

    def __init__(self, fn, grid, block, kinds, shared=0):
        self._fn = fn
        self._grid = tuple((list(grid) + [1, 1, 1])[:3])
        self._block = tuple((list(block) + [1, 1, 1])[:3])
        self._shared = shared
        # 'T' tensor, 'O' tensor the kernel writes, 'n' optional pointer (None -> NULL),
        # 'i' int32, 'f' float32.
        self._kinds = kinds
        self._n = len(kinds)
        # one 8-byte slot per argument, with a stable address for the pointer array
        self._raw = (ctypes.c_uint64 * self._n)()
        self._ptrs = (ctypes.c_void_p * self._n)()
        for _i in range(self._n):
            self._ptrs[_i] = ctypes.cast(ctypes.byref(self._raw, _i * 8),
                                         ctypes.c_void_p)

    def __call__(self, *args, stream=None) -> None:
        import torch

        if len(args) != self._n:
            raise TypeError(f"expected {self._n} args, got {len(args)}")
        # cuLaunchKernel takes an array of POINTERS to argument values, so the values
        # live in one preallocated 8-byte slot each and the pointer array is built once.
        # Rebuilding a ctypes object per argument cost ~7us for Marlin's 18-argument
        # signature, which is most of the launch time when the kernel itself only runs
        # for 12us (measured: 26us vs 19us against the C++ binding at N=2048).
        kinds = self._kinds
        raw = self._raw
        for i in range(self._n):
            a = args[i]
            k = kinds[i]
            if k == "n":
                raw[i] = 0 if a is None else a.data_ptr()
            elif k == "T" or k == "O":
                raw[i] = a.data_ptr()
                # No record_stream here. Every caller in this engine keeps its tensors
                # alive for the duration of the call (they are Python locals), so the
                # allocator cannot recycle them mid-launch. Recording instead pinned
                # buffers to the stream long after they were dead, which fragmented a
                # card this full badly enough to break the 40-layer plan and tripped
                # CUDACachingAllocator's "!handles_.at(i)" assert. The generic launch()
                # below still records, for one-off call sites that pass temporaries.
            elif k == "f":
                raw[i] = _FLOAT_BITS(a)
            else:
                raw[i] = a & 0xFFFFFFFF
        ptrs = self._ptrs
        if stream is None:
            # 缓存 Stream wrapper。per-kernel-launch 调 torch.cuda.current_stream() 会在
            # 内部走 _get_device_index + 构造 Stream 对象，py-spy 采样里 Stream.__new__
            # 和 current_stream 是仅次于 cuLaunchKernel 的热点；一个 token 1200+ 次
            # launch，这些纯开销就吃掉了可观的 CPU 时间（而 CPU 已经是瓶颈）。
            # 默认直接用缓存，只有 TI_STREAM_CHECK=1 时才每次回查 torch —— 引擎内部
            # 不切换 stream，graph capture 也在独立路径上，回查只有在外部调用者
            # 自己切流时才需要。
            tid = _threading.get_ident()
            ent = _STREAM_CACHE.get(tid)
            if ent is None or _STREAM_CHECK:
                cur = torch.cuda.current_stream()
                raw = int(cur.cuda_stream)
                if ent is None or ent[0] != raw:
                    stream = ctypes.c_void_p(raw)
                    _STREAM_CACHE[tid] = (raw, stream, cur)
                else:
                    stream = ent[1]
            else:
                stream = ent[1]
        gx, gy, gz = self._grid
        bx, by, bz = self._block
        rc = _lib().cuLaunchKernel(self._fn, gx, gy, gz, bx, by, bz, self._shared,
                                   stream, ptrs, None)
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel failed: {rc}")
        # holders must outlive cuLaunchKernel, which copies the argument block
        return None





def _FLOAT_BITS(v) -> int:
    """Reinterpret a Python float as the 32 bits a kernel reads for a float argument."""
    import struct

    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


# CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
_CU_FUNC_MAX_DYNAMIC_SMEM = 8


def set_dynamic_smem(name: str, func: str, nbytes: int) -> None:
    """Raise the function's dynamic shared memory cap to nbytes.

    Required before any launch asking for more than 48KB, which Marlin does (it uses
    the entire opt-in budget so one block can hold a whole K-slice of stages).
    """
    fn = kernel(name, func)
    rc = _lib().cuFuncSetAttribute(fn, _CU_FUNC_MAX_DYNAMIC_SMEM, int(nbytes))
    if rc != 0:
        raise RuntimeError(f"cuFuncSetAttribute failed: {rc}")


def bind(name: str, func: str, grid, block, kinds, shared: int = 0) -> BoundLaunch:
    """Resolve a cubin kernel once and return a reusable launcher."""
    return BoundLaunch(kernel(name, func), grid, block, kinds, shared)


def launch(fn: int, grid, block, args, stream=None, shared: int = 0) -> None:
    """Generic launch (kept for one-off call sites; prefer bind() in hot paths).

    fn is launched with python args: tensors -> data_ptr, int -> int32, float -> float32.

    The stream defaults to torch's CURRENT stream rather than 0. Legacy stream 0 is not
    captured by torch.cuda.graph_capture, so a kernel launched there executes for real
    during capture and is absent from every replay -- the graph comes out empty and the
    recurrent state it was supposed to advance stays frozen (observed as NaN logits on
    the second replay of Qwen3.5's GatedDeltaNet).
    """
    import torch

    if stream is None:
        stream = int(torch.cuda.current_stream().cuda_stream)
    holders = []
    ptrs = (ctypes.c_void_p * len(args))()
    # torch's caching allocator only knows a tensor is in use when a torch op touches it.
    # The driver API takes a bare pointer, so once the Python reference is dropped the
    # block goes straight back to the pool while the kernel may still be reading it. That
    # showed up as decode garbage that varied with layer depth and between runs (layers
    # 0-30 bit-stable, 32/34/38/39 flipping) because a later allocation reused the memory
    # of a weight pointer the launch had already queued.
    _cur = torch.cuda.current_stream() if torch.cuda.is_available() else None
    for i, a in enumerate(args):
        if torch.is_tensor(a):
            holders.append(ctypes.c_void_p(a.data_ptr()))
            if _cur is not None:
                try:
                    a.record_stream(_cur)
                except Exception:
                    pass
        elif isinstance(a, float):
            holders.append(ctypes.c_float(a))
        elif isinstance(a, int):
            holders.append(ctypes.c_int(a))
        else:
            raise TypeError(f"unsupported kernel arg {type(a)}")
        ptrs[i] = ctypes.cast(ctypes.byref(holders[i]), ctypes.c_void_p)
    gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
    bx, by, bz = (list(block) + [1, 1, 1])[:3]
    rc = _lib().cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, shared, ctypes.c_void_p(int(stream)),
                               ptrs, None)
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel failed: {rc}")