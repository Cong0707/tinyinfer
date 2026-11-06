"""O3 A/B: real L0 fp8 weight, dequant+bf16 matmul vs official tilelang fp8_gemm."""
import sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.gemm import available, fp8_linear
from tinyinfer.quant import dequant_fp8_block
from tinyinfer.weights import ShardStore

MODEL = ROOT / "cache" / "weights"


def main():
    print("cuda:", torch.cuda.is_available())
    if not available():
        print("official kernel import failed"); return
    store = ShardStore(MODEL)
    w = store.get("layers.0.attn.wq_a.weight").cuda()      # [1280, 5120] fp8
    s = store.get("layers.0.attn.wq_a.scale").cuda()       # [40, 160] e8m0
    print("w", tuple(w.shape), w.dtype, "| s", tuple(s.shape), s.dtype)

    torch.manual_seed(0)
    x = (torch.randn(8, w.shape[1], dtype=torch.bfloat16) * 0.02).cuda()

    # reference: dequant weight to bf16, plain matmul
    w_f = dequant_fp8_block(w.cpu(), s.cpu()).cuda()
    y_ref = torch.nn.functional.linear(x, w_f)

    # tilelang fp8 gemm
    y_tl = fp8_linear(x, w, s)
    diff = (y_ref.float() - y_tl.float()).abs()
    rel = (diff.max() / y_ref.float().abs().max().clamp_min(1e-6)).item()
    print(f"shape={tuple(y_tl.shape)} max_abs_diff={diff.max().item():.6f} rel={rel:.6f} "
          f"finite={torch.isfinite(y_tl.float()).all().item()}")

    def t(fn, iters=20):
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3
    a = t(lambda: torch.nn.functional.linear(x, w_f))
    b = t(lambda: fp8_linear(x, w, s))
    print(f"latency ms: dequant+bf16={a:.3f} fp8_gemm={b:.3f} speedup={a/b:.2f}x")


if __name__ == "__main__":
    main()
