"""ctypes binding for the cutlass-based fp8 blockwise GEMM (build/tinyinfer_fp8.dll)."""
from __future__ import annotations

import ctypes
import os
import pathlib

_DLL = None
_ERR = None


def _load():
    global _DLL, _ERR
    if _DLL is not None or _ERR is not None:
        return _DLL
    here = pathlib.Path(__file__).resolve().parents[3]  # repo root: src/tinyinfer/kernels/x.py
    for cand in (here / 'build' / 'tinyinfer_fp8.dll',
                 here / 'build' / 'libtinyinfer_fp8.so'):
        if cand.exists():
            try:
                dll = ctypes.CDLL(str(cand))
                fn = dll.tinyinfer_fp8_blockwise_gemm
                fn.restype = ctypes.c_int
                fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
                err = dll.tinyinfer_fp8_last_error
                err.restype = ctypes.c_char_p
                err.argtypes = []
                _DLL = dll
                return dll
            except OSError as exc:
                _ERR = exc
                return None
    _ERR = FileNotFoundError('tinyinfer_fp8 not built')
    return None


def available() -> bool:
    return _load() is not None


def reason():
    _load()
    return _ERR


def fp8_blockwise_gemm(a, sfa, b, sfb, out, stream=None):
    """C[M,N] = A[M,K] @ B[N,K]^T with 1x32x32 block scales. Tensors must be contiguous."""
    dll = _load()
    if dll is None:
        raise RuntimeError(f'tinyinfer_fp8 not available: {_ERR!r}')
    m, k = a.shape
    n = b.shape[0]
    if stream is None:
        import torch

        stream = torch.cuda.current_stream().cuda_stream
    rc = dll.tinyinfer_fp8_blockwise_gemm(
        a.data_ptr(), sfa.data_ptr(), b.data_ptr(), sfb.data_ptr(), out.data_ptr(),
        int(m), int(n), int(k), ctypes.c_void_p(stream))
    if rc != 0:
        detail = dll.tinyinfer_fp8_last_error()
        raise RuntimeError(
            f'tinyinfer_fp8_blockwise_gemm returned {rc}: '
            f'{(detail or b"").decode("utf-8", "replace")}')
    return out
