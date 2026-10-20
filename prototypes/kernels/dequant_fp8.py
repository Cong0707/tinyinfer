"""Fused fp8(32x32 block) -> bf16 dequant as one tilelang kernel.

HF weights store fp8 E4M3 with a ue8m0 scale per 32x32 block. The torch path
materializes a dense scale (repeat/expand) and does several elementwise passes
(~2ms for wo_a [8192,4096]); this kernel reads fp8 + scales and writes bf16 once.
"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def dequant_fp8_block_kernel(O: int, I: int, BLK_O: int = 32, BLK_I: int = 32, threads: int = 128,
                             scale_dtype: str = "uint8", out_dtype: str = "bfloat16"):
    """Grid over (O/BLK_O, I/BLK_I) blocks; each block dequantizes one 32x32 tile."""
    import tilelang.language as TL

    @T.prim_func
    def kernel(
        W: T.Tensor([O, I], "float8_e4m3fn"),
        S: T.Tensor([O // BLK_O, I // BLK_I], scale_dtype),
        Y: T.Tensor([O, I], out_dtype),
    ):
        with T.Kernel(O // BLK_O, I // BLK_I, threads=threads) as (bo, bi):
            for i in T.Parallel(BLK_O):
                for j in T.Parallel(BLK_I):
                    # e8m0: value = 2^(u8 - 127)
                    raw = T.Cast("float32", S[bo, bi])
                    scale = T.exp2(raw - 127.0)
                    Y[bo * BLK_O + i, bi * BLK_I + j] = T.Cast(out_dtype, T.Cast("float32", W[bo * BLK_O + i, bi * BLK_I + j]) * scale)

    return kernel


def dequant_ref(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    from tinyinfer.quant import dequant_fp8_block
    return dequant_fp8_block(w, s)


def main() -> int:
    import sys, time
    sys.path.insert(0, "E:/Develop/Deepseek/tinyinfer/src")

    O, I = 8192, 4096
    torch.manual_seed(0)
    w = (torch.randn(O, I, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    s = torch.full((O // 32, I // 32), 127, dtype=torch.uint8, device="cuda")  # scale 1.0
    ref = dequant_ref(w, s)

    k = dequant_fp8_block_kernel(O, I)
    out = torch.empty(O, I, dtype=torch.bfloat16, device="cuda")
    k(w, s, out)
    torch.cuda.synchronize()
    err = (out.float() - ref.float()).abs().max().item()
    print(f"correctness max abs err: {err:.5f}")

    def bench(fn, n=50):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n): fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n

    print(f"torch dequant: {bench(lambda: dequant_ref(w, s)):.3f} ms")
    print(f"tilelang dequant: {bench(lambda: k(w, s, out)):.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
