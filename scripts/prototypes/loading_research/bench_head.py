"""Where should the last-token head live on 8 GB VRAM?

Measures the released head shape (129280 x 5120 bf16, 1.324 GB) four ways:

* CPU bf16 matvec (the recommended default; current model.py uses fp32)
* GPU bf16 resident + GEMV
* bf16 streamed over PCIe (pinned) + GEMV
* fp8 head (per-row scale) H2D and torch dequant path; rowwise fp8 GEMM is
  probed because torch._scaled_mm support depends on arch/torch version.

Usage:
    python bench_head.py --threads 16
"""
from __future__ import annotations

import argparse
import time

import torch


def bench(fn, warmup: int = 3, runs: int = 5) -> float:
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t) / runs


def bench_gpu(fn, warmup: int = 5, runs: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rows", type=int, default=129280)
    ap.add_argument("--dim", type=int, default=5120)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    V, D = args.rows, args.dim
    nbytes = V * D * 2
    g = torch.Generator().manual_seed(0)
    w = torch.randn(V, D, dtype=torch.bfloat16, generator=g)
    h = torch.randn(D, dtype=torch.bfloat16, generator=g)
    print(f"head {V}x{D} bf16 = {nbytes / 1e9:.3f} GB, threads={args.threads}")

    dt = bench(lambda: torch.mv(w, h))
    print(f"CPU bf16 mv: {dt * 1e3:6.2f} ms  ({nbytes / dt / 1e9:.1f} GB/s effective)")
    del w
    wf = torch.randn(V, D, dtype=torch.float32, generator=g)
    hf = h.float()
    dt = bench(lambda: torch.mv(wf, hf), warmup=2, runs=3)
    print(f"CPU fp32 mv: {dt * 1e3:6.2f} ms (plus 2.65 GB fp32 head)")
    del wf, hf

    if not torch.cuda.is_available():
        return 0
    torch.cuda.init()
    wb = torch.randn(V, D, dtype=torch.bfloat16, generator=g).cuda()
    hb = h.cuda()
    dt = bench_gpu(lambda: torch.mv(wb, hb), warmup=5, runs=20)
    print(f"GPU bf16 mv resident: {dt * 1e3:5.2f} ms (occupies {nbytes / 1e9:.2f} GB VRAM)")
    del wb, hb
    torch.cuda.empty_cache()

    for pin in (False, True):
        src = torch.empty(V, D, dtype=torch.bfloat16, pin_memory=pin)
        dst = torch.empty_like(src, device="cuda")
        dt = bench(lambda: dst.copy_(src), warmup=1, runs=3)
        print(f"H2D bf16 head pinned={str(pin):<5s}: {dt * 1e3:5.1f} ms ({nbytes / dt / 1e9:.2f} GB/s)")
        del src, dst
    torch.cuda.empty_cache()

    torch.set_num_threads(args.threads)
    wsrc = torch.randn(V, D, dtype=torch.bfloat16, generator=g)
    scale = wsrc.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    wq = (wsrc.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    src8 = torch.empty(V, D, dtype=torch.float8_e4m3fn, pin_memory=True)
    src8.copy_(wq)
    dst8 = torch.empty_like(wq, device="cuda")
    dt = bench(lambda: dst8.copy_(src8), warmup=1, runs=3)
    print(f"H2D fp8 head (0.66 GB) pinned: {dt * 1e3:5.1f} ms ({nbytes / 2 / dt / 1e9:.2f} GB/s)")
    hh = h.cuda()
    sg = scale.squeeze(1).cuda()
    dt = bench_gpu(lambda: torch.mv(dst8.to(torch.bfloat16) * sg.to(torch.bfloat16)[:, None], hh),
                   warmup=2, runs=5)
    print(f"GPU fp8->bf16 dequant + mv: {dt * 1e3:5.2f} ms (torch path, no fused kernel)")
    try:
        a = h.float().div(h.float().abs().max() / 448).to(torch.float8_e4m3fn).view(1, D).cuda()
        b = wq.t().cuda()  # [D, V], column-major view
        sa = torch.ones(1, 1, dtype=torch.float32, device="cuda")
        sb = scale.amax().view(1, 1).to(torch.float32).cuda()
        torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        dt = bench_gpu(lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16),
                       warmup=3, runs=10)
        print(f"torch._scaled_mm fp8 head (per-tensor scale): OK, {dt * 1e3:.2f} ms on GPU")
    except Exception as exc:  # noqa: BLE001 - probe only
        print(f"torch._scaled_mm fp8 head: unavailable ({type(exc).__name__}: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
