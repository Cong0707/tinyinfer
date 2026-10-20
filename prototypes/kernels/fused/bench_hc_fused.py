"""Bench: fused HC mixes + sinkhorn vs PyTorch reference (real L0 weights, decode B=1,S=1,hc=4,D=5120)."""
import sys, statistics
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
import torch
import torch.nn.functional as F
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
from tinyinfer.weights import ShardStore
from tinyinfer.hc import hc_split_sinkhorn
import hc_fused as hf

store = ShardStore(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
fn = store.get("layers.0.hc_attn_fn").to(dev)
scale = store.get("layers.0.hc_attn_scale").to(dev)
base = store.get("layers.0.hc_attn_base").to(dev)
x = torch.randn(1, 1, 4, 5120, device=dev, dtype=torch.bfloat16)
EPS = 1e-6

def bench(fn_, warmup=10, iters=100, reps=3):
    for _ in range(warmup): fn_()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(iters): fn_()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / iters)
    return best
def med(fn_, rounds=5): return statistics.median([bench(fn_, 5, 100) for _ in range(rounds)])
def rel(a, b): return (a.float() - b.float()).abs().max().item() / b.float().abs().max().clamp_min(1e-30).item()

def torch_mixes():
    xf = x.flatten(2).float()
    rsq = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    return F.linear(xf, fn) * rsq
def torch_all(iters=20):
    mixes = torch_mixes()
    return hc_split_sinkhorn(mixes, scale, base, 4, iters, EPS)
def fused_all(iters=20):
    mixes = hf.fused_hc_mixes(x, fn, EPS)
    return hf.fused_sinkhorn(mixes, scale, base, 4, iters, EPS)

pre_t, post_t, comb_t = torch_all()
pre_f, post_f, comb_f = fused_all()
torch.cuda.synchronize()
print(f"correctness rel err: pre {rel(pre_f,pre_t):.2e} post {rel(post_f,post_t):.2e} comb {rel(comb_f,comb_t):.2e}")
p6, q6, c6 = torch_all(6); p20, q20, c20 = torch_all(20)
print(f"iters=6 vs 20 (torch): pre {rel(p6,p20):.4f} post {rel(q6,q20):.4f} comb {rel(c6,c20):.4f}")

t_t = med(torch_all); t_f = med(fused_all)
print(f"hc_mixes total (mixes+sinkhorn): torch {t_t*1e3:.1f}us vs fused {t_f*1e3:.1f}us ({t_t/t_f:.1f}x)")
t_mix = med(lambda: hf.fused_hc_mixes(x, fn, EPS))
def fused_sink_only():
    mixes = hf.fused_hc_mixes(x, fn, EPS)
    return hf.fused_sinkhorn(mixes, scale, base, 4, 20, EPS)
t_sink = med(fused_sink_only)
print(f"fused breakdown: mixes {t_mix*1e3:.1f}us, mixes+sinkhorn {t_sink*1e3:.1f}us => sinkhorn {((t_sink-t_mix)*1e3):.1f}us")

from torch.profiler import profile, ProfilerActivity
def aten_launch_ops(fn_, iters_=5):
    fn_(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(iters_): fn_()
        torch.cuda.synchronize()
    NON = ("view","as_strided","empty","detach","record_stream","contiguous","slice","select",
           "permute","transpose","expand","unflatten","flatten","squeeze","unsqueeze","alias","to","reshape","t")
    total = 0
    for e in prof.key_averages():
        if not e.key.startswith("aten::"): continue
        name = e.key.split("::")[-1]
        if not any(name.startswith(n) for n in NON): total += e.count
    return total / iters_
n_t = aten_launch_ops(torch_all); n_f = aten_launch_ops(fused_all)
print(f"host launch-ish ops per hc_mixes: torch {n_t:.1f}, fused {n_f:.1f} (+2 tilelang kernels)")
print(f"per layer (2 hc_mixes): torch {2*n_t:.1f} vs fused {2*(n_f+2):.1f}")

# ---- optional: hc_pre / hc_post ----
x_collapsed = torch.randn(1, 1, 5120, device=dev, dtype=torch.bfloat16)
residual = torch.randn(1, 1, 4, 5120, device=dev, dtype=torch.bfloat16)

def torch_pre():
    return torch.sum(pre_t.unsqueeze(-1) * x.float(), dim=2).to(torch.bfloat16)
def torch_post():
    y = post_t.unsqueeze(-1) * x_collapsed.unsqueeze(-2)
    y = y + torch.sum(comb_t.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.to(torch.bfloat16)
y_pre_t = torch_pre(); y_post_t = torch_post()
y_pre_f = hf.fused_hc_pre(x, pre_t); y_post_f = hf.fused_hc_post(x_collapsed, residual, post_t, comb_t)
torch.cuda.synchronize()
print(f"hc_pre  rel {rel(y_pre_f,y_pre_t):.2e}; hc_post rel {rel(y_post_f,y_post_t):.2e}")
t_pre_t = med(torch_pre); t_pre_f = med(lambda: hf.fused_hc_pre(x, pre_t))
t_post_t = med(torch_post); t_post_f = med(lambda: hf.fused_hc_post(x_collapsed, residual, post_t, comb_t))
print(f"hc_pre:  torch {t_pre_t*1e3:.1f}us vs fused {t_pre_f*1e3:.1f}us ({t_pre_t/t_pre_f:.1f}x)")
print(f"hc_post: torch {t_post_t*1e3:.1f}us vs fused {t_post_f*1e3:.1f}us ({t_post_t/t_post_f:.1f}x)")
n_pre = aten_launch_ops(torch_pre); n_post = aten_launch_ops(torch_post)
print(f"launch-ish ops: pre torch {n_pre:.1f}/fused 1; post torch {n_post:.1f}/fused 1")
print(f"per layer HC (2 mixes+2 pre+2 post): torch {2*n_t+2*n_pre+2*n_post:.1f} vs fused {2*2+2+2} tilelang kernels")
