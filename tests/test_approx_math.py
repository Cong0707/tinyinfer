"""Error bounds for the EXTREME-tier approximate math."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyinfer.approx_math import fast_exp, fast_rsqrt, fast_sigmoid, fast_silu, poly_cos, poly_sin


def rel_err(got, ref):
    d = (got.float() - ref.float()).abs()
    return (d / ref.float().abs().clamp_min(1e-3)).max().item()


def main():
    torch.manual_seed(0)
    x = torch.randn(10000) * 3
    print("fast_exp rel_err:", rel_err(fast_exp(x), torch.exp(x)))
    xr = torch.rand(10000) * 10 + 0.1
    print("fast_rsqrt rel_err:", rel_err(fast_rsqrt(xr), torch.rsqrt(xr)))
    print("fast_sigmoid abs_err:", (fast_sigmoid(x) - torch.sigmoid(x)).abs().max().item())
    print("fast_silu abs_err:", (fast_silu(x) - torch.nn.functional.silu(x)).abs().max().item())
    a = torch.randn(10000) * 2 * torch.pi
    print("poly_sin abs_err:", (poly_sin(a) - torch.sin(a)).abs().max().item())
    print("poly_cos abs_err:", (poly_cos(a) - torch.cos(a)).abs().max().item())

    y = torch.randn(1 << 20) * 3
    import time
    def t(fn):
        fn(y); t0 = time.perf_counter()
        for _ in range(10): fn(y)
        return (time.perf_counter() - t0) / 10 * 1e3
    print(f"ms/1M (CPU, reference only): exp={t(torch.exp):.2f} fast_exp={t(fast_exp):.2f} "
          f"silu={t(torch.nn.functional.silu):.2f} fast_silu={t(fast_silu):.2f}")


if __name__ == "__main__":
    main()
