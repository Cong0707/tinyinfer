"""Hand-written decode kernels: nvcc -> cubin -> CUDA driver API.

No torch C++ extension build. scripts/build_native.ps1 runs nvcc (under the MSVC
environment that nvcc insists on) and this module loads the cubin with nvcuda.dll, so
changing a kernel is one nvcc call instead of a rebuild of torch.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

_DIR = Path(__file__).with_name("native")
_MODULES: dict[str, int] = {}
_KERNELS: dict[tuple[str, str], int] = {}
_LIB = None


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL("nvcuda.dll")
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
        raise FileNotFoundError(f"{path} (run scripts/build_native.ps1)")
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


def launch(fn: int, grid, block, args, stream: int = 0, shared: int = 0) -> None:
    """Launch fn with python args (tensors -> data_ptr, int -> int32, float -> float32)."""
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
