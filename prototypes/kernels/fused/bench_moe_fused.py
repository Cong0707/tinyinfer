"""Bench fused MoE expert kernels vs official fp4_gemm path (real L0 expert weights)."""
import sys, statistics
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
from tinyinfer.weights import ShardStore
import moe_fused as mf
import kernel as ak

store = ShardStore(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
def getv(name):
    t = store.get("layers.0.ffn.experts.0." + name)
    return t.to(dev)
w1 = getv("w1.weight"); s1 = getv("w1.scale")
w3 = getv("w3.weight"); s3 = getv("w3.scale")
w2 = getv("w2.weight"); s2 = getv("w2.scale")
w1p = w1.view(torch.float4_e2m1fn_x2); w3p = w3.view(torch.float4_e2m1fn_x2); w2p = w2.view(torch.float4_e2m1fn_x2)

def bench(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(3):
        ss = torch.cuda.Event(True); ee = torch.cuda.Event(True); ss.record()
        for _ in range(iters): fn()
        ee.record(); torch.cuda.synchronize(); best = min(best, ss.elapsed_time(ee) / iters)
    return best
def med(fn, rounds=3): return statistics.median([bench(fn) for _ in range(rounds)])
def rel(a, b): return (a.float() - b.float()).abs().mean().item() / b.float().abs().mean().clamp_min(1e-30).item()

def torch_step(x, weights):
    aq, asc = ak.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    g = ak.fp4_gemm(aq, asc, w1p, s1, torch.float8_e8m0fnu, act_block_size=32)
    u = ak.fp4_gemm(aq, asc, w3p, s3, torch.float8_e8m0fnu, act_block_size=32)
    g = g.float().clamp(max=10.0); u = u.float().clamp(-10.0, 10.0)
    h = (g / (1 + torch.exp(-g))) * u
    hq, hs = ak.act_quant(h.to(torch.bfloat16), 32, "ue8m0", torch.float8_e8m0fnu)
    y = ak.fp4_gemm(hq, hs, w2p, s2, torch.float8_e8m0fnu, act_block_size=32)
    return y * weights.view(-1, 1)

def fused_pq(x, weights, bm=32, bn=128):
    aq, asc = ak.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    h = mf.fused_fp4_swiglu_pq(aq, asc, w1p, s1, w3p, s3, 10.0, bm, bn)
    hq, hs = ak.act_quant(h, 32, "ue8m0", torch.float8_e8m0fnu)
    return mf.fused_fp4_down_pq(hq, hs, w2p, s2, weights, bm, bn)

def fused_bf16(x, weights, bm=32):
    h = mf.fused_fp4_swiglu(x, w1p, s1, w3p, s3, 10.0, bm)
    return mf.fused_fp4_down(h, w2p, s2, weights, bm)

for M in [8, 64]:
    x = torch.randn(M, 5120, device=dev, dtype=torch.bfloat16)
    weights = torch.rand(M, device=dev).float() * 1.5
    y_t = torch_step(x, weights)
    t_t = med(lambda: torch_step(x, weights))
    print(f"\nM={M}: official-fp4 step {t_t*1e3:.1f}us")
    for bm, bn in [(16,64),(16,128),(32,64),(32,128),(64,128)]:
        try:
            y = fused_pq(x, weights, bm, bn)
            torch.cuda.synchronize()
            t = med(lambda: fused_pq(x, weights, bm, bn))
            print(f"  pq bm={bm} bn={bn}: {t*1e3:.1f}us ({t_t/t:.2f}x), rel {rel(y,y_t):.2e}")
        except Exception as e:
            print(f"  pq bm={bm} bn={bn}: FAIL {type(e).__name__} {str(e)[:120]}")
    try:
        y = fused_bf16(x, weights); t = med(lambda: fused_bf16(x, weights))
        print(f"  bf16-internal bm=32: {t*1e3:.1f}us ({t_t/t:.2f}x), rel {rel(y,y_t):.2e}")
    except Exception as e:
        print("  bf16-internal FAIL", type(e).__name__, str(e)[:120])

# launch counts
from torch.profiler import profile, ProfilerActivity
def aten_launch_ops(fn, iters_=5):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(iters_): fn()
        torch.cuda.synchronize()
    NON=("view","as_strided","empty","detach","record_stream","contiguous","slice","select","permute","transpose","expand","unflatten","flatten","squeeze","unsqueeze","alias","to","reshape","t")
    n=0
    for e in prof.key_averages():
        if not e.key.startswith("aten::"): continue
        if not any(e.key.split("::")[-1].startswith(p) for p in NON): n+=e.count
    return n/iters_
M=8; x=torch.randn(M,5120,device=dev); weights=torch.rand(M,device=dev).float()
n_t=aten_launch_ops(lambda: torch_step(x,weights))
print(f"\nlaunch-ish ops per expert: official-fp4 {n_t:.1f}; fused pq = 2 act_quant + 2 tilelang = 4; per layer 6 experts: {6*n_t:.0f} vs 24")
store.close()
