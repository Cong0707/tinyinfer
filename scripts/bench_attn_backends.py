"""O2 A/B: L0 real weights, PyTorch reference vs tilelang sm120 sparse_attn."""
import sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tinyinfer.kernels as K
from tinyinfer.attention import MLAWindowAttention
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore

MODEL = ROOT / "cache" / "weights"
CFG = TinyConfig.from_hf(MODEL)


def to_cuda(attn):
    for k, v in list(vars(attn).items()):
        if isinstance(v, torch.Tensor):
            setattr(attn, k, v.cuda())
    return attn


def run(attn, x, tilelang: bool):
    K._TL_STATE = None if tilelang else False
    attn.reset()
    with torch.inference_mode():
        y = attn.forward(x, 0)
    torch.cuda.synchronize()
    return y


def timeit(attn, x, tilelang: bool, iters=5):
    K._TL_STATE = None if tilelang else False
    attn.reset()
    with torch.inference_mode():
        attn.forward(x, 0)  # warmup (also JIT compile)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            attn.reset()
            attn.forward(x, 0)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    store = ShardStore(MODEL)
    attn = to_cuda(MLAWindowAttention(store, 0, CFG))
    torch.manual_seed(0)
    x = (torch.randn(1, 8, CFG.dim, dtype=torch.bfloat16) * 0.02).cuda()

    y_ref = run(attn, x, tilelang=False)
    y_tl = run(attn, x, tilelang=True)
    diff = (y_ref.float() - y_tl.float()).abs()
    rel = (diff.max() / y_ref.float().abs().max().clamp_min(1e-6)).item()
    print(f"shape={tuple(y_ref.shape)} dtype={y_ref.dtype}")
    print(f"max_abs_diff={diff.max().item():.6f} rel={rel:.6f} finite_ref={torch.isfinite(y_ref.float()).all().item()} finite_tl={torch.isfinite(y_tl.float()).all().item()}")
    print(f"tilelang_state={K._TL_STATE}")

    t_ref = timeit(attn, x, tilelang=False, iters=5)
    t_tl = timeit(attn, x, tilelang=True, iters=5)
    print(f"latency ms/layer: ref={t_ref:.3f} tilelang={t_tl:.3f} speedup={t_ref/t_tl:.2f}x")


if __name__ == "__main__":
    main()
