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


def launch(fn: int, grid, block, args, stream: int = 0, shared: int = 0) -> None:
    """Launch fn with python args (tensors -> data_ptr, int -> int32, float -> float32)."""
    import torch

    if stream is None:
        stream = int(torch.cuda.current_stream().cuda_stream)
    holders = []
    ptrs = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        if torch.is_tensor(a):
            holders.append(ctypes.c_void_p(a.data_ptr()))
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
