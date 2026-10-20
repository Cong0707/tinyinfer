"""Exact vs approximate math kernels on sm_120: timing + accuracy.

Ops: RMSNorm (fast_rsqrt), SwiGLU (fast_silu), softmax (fast_exp), RoPE (table),
plus the fused sparse attention with fast_exp in its online softmax.
"""
import sys, time, math
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import torch.nn.functional as F
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
import approx_kernels as ak
from tinyinfer.approx_math import fast_exp as torch_fast_exp
from sparse_attn_sm120 import sparse_attn_sm120

def bench(fn, warmup=10, iters=100, reps=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / iters)
    return best

def relerr(a, b):
    a = a.float(); b = b.float()
    return (a - b).abs().mean().item() / b.abs().mean().clamp_min(1e-30).item()

print("=== 1) RMSNorm (exact kernel vs fast_rsqrt kernel vs torch eager) ===")
for dim, rows_list in [(1280, [1, 128, 512]), (512, [1, 128, 512])]:
    for rows in rows_list:
        x = torch.randn(rows, dim, device=dev); w = torch.randn(dim, device=dev)
        y_e = ak.rmsnorm(x, w, "exact"); y_f = ak.rmsnorm(x, w, "fast")
        t_e = bench(lambda: ak.rmsnorm(x, w, "exact"))
        t_f = bench(lambda: ak.rmsnorm(x, w, "fast"))
        xf = x.float()
        t_t = bench(lambda: (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-20) * w.float()).to(torch.bfloat16))
        print(f"dim={dim} rows={rows:4d} exact {t_e*1e3:7.2f}us  fast {t_f*1e3:7.2f}us  torch {t_t*1e3:7.2f}us  rel_err {relerr(y_f,y_e):.5f}")

print("\n=== 2) SwiGLU (exact exp / __expf / poly) ===")
for rows in [7, 896, 3584]:
    g = torch.randn(rows, 2304, device=dev); u = torch.randn(rows, 2304, device=dev)
    y_e = ak.swiglu(g, u, "exact"); y_h = ak.swiglu(g, u, "hw"); y_p = ak.swiglu(g, u, "poly")
    t_e = bench(lambda: ak.swiglu(g, u, "exact")); t_h = bench(lambda: ak.swiglu(g, u, "hw")); t_p = bench(lambda: ak.swiglu(g, u, "poly"))
    t_t = bench(lambda: (F.silu(g.float()) * u.float()).to(torch.bfloat16))
    print(f"rows={rows:4d} exact {t_e*1e3:7.2f}us hw {t_h*1e3:7.2f}us poly {t_p*1e3:7.2f}us torch {t_t*1e3:7.2f}us | err hw {relerr(y_h,y_e):.6f} poly {relerr(y_p,y_e):.6f}")

print("\n=== 3) softmax (exact / hw / poly) ===")
for rows in [64, 8192, 32768]:
    s = torch.randn(rows, 640, device=dev, dtype=torch.float32) * 3
    p_e = ak.softmax(s, "exact"); p_h = ak.softmax(s, "hw"); p_p = ak.softmax(s, "poly")
    t_e = bench(lambda: ak.softmax(s, "exact")); t_h = bench(lambda: ak.softmax(s, "hw")); t_p = bench(lambda: ak.softmax(s, "poly"))
    t_t = bench(lambda: torch.softmax(s, -1))
    def kl(a, b): return (b * (b.clamp_min(1e-30).log() - a.clamp_min(1e-30).log())).sum(-1).mean().item()
    print(f"rows={rows:5d} exact {t_e*1e3:7.2f}us hw {t_h*1e3:7.2f}us poly {t_p*1e3:7.2f}us torch {t_t*1e3:7.2f}us | KL(hw) {kl(p_h,p_e):.2e} KL(poly) {kl(p_p,p_e):.2e}")

print("\n=== 4) RoPE table kernel vs torch complex path ===")
rd = 64
cos = torch.randn(1024, rd // 2, device=dev).float(); sin = torch.randn(1024, rd // 2, device=dev).float()
for rows in [1, 64, 128, 8192]:
    x = torch.randn(rows, rd, device=dev); pos = torch.randint(0, 1024, (rows,), device=dev)
    y = ak.rope_table(x, cos, sin, pos)
    def torch_rope():
        xf = x.float(); a = xf[:, 0::2]; b = xf[:, 1::2]; c = cos[pos]; s = sin[pos]
        out = torch.empty_like(xf); out[:, 0::2] = a * c - b * s; out[:, 1::2] = a * s + b * c
        return out.to(torch.bfloat16)
    ok = torch.allclose(y, torch_rope())
    t_k = bench(lambda: ak.rope_table(x, cos, sin, pos)); t_t = bench(torch_rope)
    print(f"rows={rows:5d} table {t_k*1e3:7.2f}us torch {t_t*1e3:7.2f}us bitwise_equal={ok}")

print("\n=== 5) fused sparse attention: exact vs fast_exp ===")
H, D = 64, 512
def run_attn(m, Tk, mode):
    q = torch.randn(1, m, H, D, device=dev)
    kv = torch.randn(1, Tk, D, device=dev)
    sink = torch.randn(H, device=dev).float()
    idx = torch.randint(0, Tk, (1, m, Tk), device=dev, dtype=torch.int32)
    return q, kv, sink, idx
for m, name in [(1, "decode"), (128, "prefill")]:
    q, kv, sink, idx = run_attn(m, 640, "exact")
    o_e = sparse_attn_sm120(q, kv, sink, idx, D ** -0.5, exp_mode="exact")
    o_h = sparse_attn_sm120(q, kv, sink, idx, D ** -0.5, exp_mode="hw")
    o_p = sparse_attn_sm120(q, kv, sink, idx, D ** -0.5, exp_mode="poly")
    t_e = bench(lambda: sparse_attn_sm120(q, kv, sink, idx, D ** -0.5, exp_mode="exact"), 5, 30)
    t_h = bench(lambda: sparse_attn_sm120(q, kv, sink, idx, D ** -0.5, exp_mode="hw"), 5, 30)
    t_p = bench(lambda: sparse_attn_sm120(q, kv, sink, idx, D ** -0.5, exp_mode="poly"), 5, 30)
    print(f"{name}: exact {t_e*1e3:7.2f}us hw {t_h*1e3:7.2f}us poly {t_p*1e3:7.2f}us | out rel err hw {relerr(o_h,o_e):.2e} poly {relerr(o_p,o_e):.2e}")

# attention-softmax KL on a real score row (decode, 640 keys + sink)
q, kv, sink, idx = run_attn(1, 640, "exact")
sc = (q[0, 0].float() @ kv[0].float().t()) * (D ** -0.5)
sc = sc[:, idx[0, 0].long()]
s2 = torch.cat([sc, sink.view(-1, 1)], dim=1)
p_exact = torch.softmax(s2, -1)
m_ = s2.max(-1, keepdim=True).values
p_poly = torch_fast_exp(s2 - m_); p_poly = p_poly / p_poly.sum(-1, keepdim=True)
kl = (p_exact * (p_exact.clamp_min(1e-30).log() - p_poly.clamp_min(1e-30).log())).sum(-1).mean().item()
print(f"attention softmax KL (poly vs exact), 64 heads: {kl:.3e}; top1 agreement {(p_exact.argmax(-1)==p_poly.argmax(-1)).float().mean().item():.3f}")

print("\n=== 6) LM softmax perplexity proxy (4 x 129280 logits) ===")
logits = torch.randn(4, 129280, device=dev, dtype=torch.float32) * 2.0
p = torch.softmax(logits, -1)
m_ = logits.max(-1, keepdim=True).values
p_ap = torch_fast_exp(logits - m_); p_ap = p_ap / p_ap.sum(-1, keepdim=True)
kl = (p * (p.clamp_min(1e-30).log() - p_ap.clamp_min(1e-30).log())).sum(-1).mean().item()
print(f"LM softmax KL: {kl:.3e}; top1 agreement {(p.argmax(-1)==p_ap.argmax(-1)).float().mean().item():.3f}; ppl ratio proxy exp(KL) = {math.exp(kl):.6f}")

print("\n=== 7) per-layer accounting (kernel times) ===")
for M in [1, 128]:
    qn = 1280; kn = 512
    t_qe = bench(lambda: ak.rmsnorm(torch.randn(M, qn, device=dev), torch.randn(qn, device=dev), "exact"), 5, 50)
    t_qf = bench(lambda: ak.rmsnorm(torch.randn(M, qn, device=dev), torch.randn(qn, device=dev), "fast"), 5, 50)
    t_ke = bench(lambda: ak.rmsnorm(torch.randn(M, kn, device=dev), torch.randn(kn, device=dev), "exact"), 5, 50)
    t_kf = bench(lambda: ak.rmsnorm(torch.randn(M, kn, device=dev), torch.randn(kn, device=dev), "fast"), 5, 50)
    g = torch.randn(7 * M, 2304, device=dev); u = torch.randn(7 * M, 2304, device=dev)
    t_se = bench(lambda: ak.swiglu(g, u, "exact"), 5, 50); t_sp = bench(lambda: ak.swiglu(g, u, "poly"), 5, 50)
    xq = torch.randn(M * 64, rd, device=dev); posq = torch.randint(0, 1024, (M * 64,), device=dev)
    t_re = bench(lambda: ak.rope_table(xq, cos, sin, posq), 5, 50)
    print(f"M={M:4d}: rmsnorm exact {((t_qe+t_ke)*1e3):6.2f}us fast {((t_qf+t_kf)*1e3):6.2f}us | swiglu exact {t_se*1e3:6.2f}us poly {t_sp*1e3:6.2f}us | rope(q+kv) {t_re*1e3:6.2f}us")