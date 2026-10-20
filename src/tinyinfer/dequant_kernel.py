"""Lazy tilelang kernel cache for fused fp8->bf16 block dequantization."""
import torch

try:
    import tilelang
    import tilelang.language as T
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

_KERNELS: dict = {}


if _OK:
    @tilelang.jit
    def _dequant_kernel(O: int, I: int, BLK: int = 32, threads: int = 128):
        @T.prim_func
        def kernel(
            W: T.Tensor([O, I], "float8_e4m3fn"),
            S: T.Tensor([O // BLK, I // BLK], "uint8"),
            Y: T.Tensor([O, I], "bfloat16"),
        ):
            with T.Kernel(O // BLK, I // BLK, threads=threads) as (bo, bi):
                for i in T.Parallel(BLK):
                    for j in T.Parallel(BLK):
                        raw = T.Cast("float32", S[bo, bi])
                        scale = T.exp2(raw - 127.0)
                        Y[bo * BLK + i, bi * BLK + j] = T.Cast(
                            "bfloat16",
                            T.Cast("float32", W[bo * BLK + i, bi * BLK + j]) * scale,
                        )
        return kernel


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """weight [O, I] float8_e4m3fn, scale [O/32, I/32] e8m0 bytes -> bf16."""
    if not _OK:
        raise RuntimeError("tilelang unavailable")
    O, I = weight.shape
    key = (O, I)
    k = _KERNELS.get(key)
    if k is None:
        k = _dequant_kernel(O, I)
        _KERNELS[key] = k
    out = torch.empty(O, I, dtype=torch.bfloat16, device=weight.device)
    k(weight.contiguous(), scale.contiguous().view(torch.uint8), out)
    return out
