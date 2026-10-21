"""CPU vs GPU decision probe: for each decode-step operation, measure host compute
time vs H2D + device compute time. The crossover rule is:

    cpu_time < data_bytes / h2d_bw + gpu_time   ->  do it on the CPU

Usage: python scripts/cpu_gpu_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

H2D_PAGEABLE = 8.9e9  # measured pageable copy bandwidth
H2D_PINNED = 26.0e9


def bench(fn, iters: int = 50) -> float:
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device={device}  h2d pinned={H2D_PINNED/1e9:.1f}GB/s pageable={H2D_PAGEABLE/1e9:.1f}GB/s")

    # --- 1) HC post/comb bmm (measured 1.9ms on GPU per call) -------------
    b, hc, d = 1, 4, 5120
    x = torch.randn(b, hc, d, device=device)
    res = torch.randn(b, hc, d, device=device)
    post = torch.randn(b, hc, device=device)
    comb = torch.randn(b, hc, hc, device=device)
    def hc_post_gpu():
        return post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * res.unsqueeze(-2), dim=2)
    t_gpu = bench(hc_post_gpu)
    xc, resc, postc, combc = x.cpu(), res.cpu(), post.cpu(), comb.cpu()
    def hc_post_cpu():
        return postc.unsqueeze(-1) * xc.unsqueeze(-2) + torch.sum(combc.unsqueeze(-1) * resc.unsqueeze(-2), dim=2)
    t_cpu = bench(hc_post_cpu)
    nbytes = (x.numel() + res.numel() + post.numel() + comb.numel()) * 4
    t_h2d = nbytes / H2D_PAGEABLE * 1e3
    t_back = nbytes / H2D_PAGEABLE * 1e3
    print(f"HC post/comb: gpu={t_gpu:.3f}ms cpu={t_cpu:.3f}ms bytes={nbytes/1e3:.0f}KB "
          f"h2d+R={t_h2d+t_back:.3f}ms -> {'CPU WINS' if t_cpu < t_gpu + t_h2d + t_back else 'GPU WINS'}")

    # --- 2) fp8 expert GEMV (M=1) -----------------------------------------
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from tinyinfer.gemm import fp8_linear
        from tinyinfer.quant import dequant_fp8_block
        w8 = torch.randn(1280, 5120, device=device).to(torch.float8_e4m3fn)
        s8 = torch.rand(160, 160, dtype=torch.uint8, device=device)
        x1 = torch.randn(1, 5120, scene := device, dtype=torch.bfloat16)
        t_gpu = bench(lambda: fp8_linear(x1, w8, s8))
        w8c, s8c, x1c = w8.cpu(), s8.cpu(), x1.cpu()
        w_dq = dequant_fp8_block(w8c, s8c)
        def cpu_lin():
            return torch.nn.functional.linear(x1c.float(), w_dq)
        t_cpu = bench(cpu_lin, 20)
        nbytes = w8.numel() + s8.numel() + x1.numel() * 2
        t_h2d = nbytes / H2D_PINNED * 1e3
        print(f"fp8 GEMV 1280x5120: gpu={t_gpu:.3f}ms cpu={t_cpu:.3f}ms bytes={nbytes/1e3:.0f}KB "
              f"h2d={t_h2d:.3f}ms -> {'CPU WINS' if t_cpu < t_gpu + t_h2d else 'GPU WINS'}")
    except Exception as e:
        print("fp8 probe skipped:", e)

    # --- 3) attention sinks / small elementwise ---------------------------
    v = torch.randn(1, 128, 64, device=device)
    def small_gpu():
        return v * 1.01 + 0.5
    t_gpu = bench(small_gpu, 200)
    vc = v.cpu()
    t_cpu = bench(lambda: vc * 1.01 + 0.5, 200)
    nbytes = v.numel() * 2
    print(f"tiny elementwise 128x64: gpu={t_gpu:.4f}ms cpu={t_cpu:.4f}ms h2d={nbytes/H2D_PAGEABLE*1e3:.4f}ms")


if __name__ == "__main__":
    main()
