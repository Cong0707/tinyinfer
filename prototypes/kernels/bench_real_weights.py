"""Real-weight (L0 shard) accuracy / O3 integration evidence.

Loads layer 0 attention + one routed expert from the real HF shard (model-00003, fp8/fp4),
then measures:
  A. approx math output error on real activations (fast_rsqrt, fast_exp, fast_silu, poly RoPE table)
  B. RoPE table kernel vs torch complex path on real q/kv tails
  C. O3: tilelang fp8_gemm / fp4_gemm vs dequant+bf16 at decode/prefill M (real weights)
"""
import sys, time, math
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
import torch
import torch.nn.functional as F
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore
from tinyinfer.quant import dequant_fp8_block, dequant_fp4_block
from tinyinfer.nn import rms_norm, apply_rope, precompute_freqs_cis, sparse_attn_ref, window_idxs_prefill
from tinyinfer.approx_math import fast_rsqrt, fast_exp, fast_silu, poly_sin, poly_cos
from tinyinfer import gemm
import approx_kernels as ak

MODEL = r"E:\Develop\Deepseek\tinyinfer\cache\weights"
cfg = TinyConfig.from_hf(MODEL)
store = ShardStore(MODEL)
print("cfg:", cfg.summary())

def load(p):
    return store.get(p)

# real activations from embedding
ids = torch.randint(0, cfg.vocab_size, (1, 512))
x_all = store.get("embed.weight")[ids].to(dev)   # [1,512,5120] bf16
x = x_all[:, :128]
print("x", tuple(x.shape), x.dtype, "rms", x.float().pow(2).mean().sqrt().item())

# attention weights
wq_a = dequant_fp8_block(load("layers.0.attn.wq_a.weight"), load("layers.0.attn.wq_a.scale")).to(dev)
wq_b = dequant_fp8_block(load("layers.0.attn.wq_b.weight"), load("layers.0.attn.wq_b.scale")).to(dev)
wkv = dequant_fp8_block(load("layers.0.attn.wkv.weight"), load("layers.0.attn.wkv.scale")).to(dev)
wo_a = dequant_fp8_block(load("layers.0.attn.wo_a.weight"), load("layers.0.attn.wo_a.scale")).to(dev)
wo_b = dequant_fp8_block(load("layers.0.attn.wo_b.weight"), load("layers.0.attn.wo_b.scale")).to(dev)
q_norm = load("layers.0.attn.q_norm.weight").to(dev)
kv_norm = load("layers.0.attn.kv_norm.weight").to(dev)
sink = load("layers.0.attn.attn_sink").float().to(dev)
freqs = precompute_freqs_cis(cfg.qk_rope_head_dim, 1024, 0, cfg.rope_theta, 16, 32.0, 1.0).to(dev)
win = cfg.sliding_window
D = cfg.head_dim; H = cfg.n_heads; RD = cfg.qk_rope_head_dim; scale = D ** -0.5

def rms(x, w, fast=False):
    xf = x.float()
    r = fast_rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) if fast else torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    return (xf * r * w.float()).to(torch.bfloat16)

def attn_ref_soft(q, kv, sink, idx, approx=False):
    b, s, h, d = q.shape; t = idx.shape[-1]
    valid = idx >= 0
    k = kv.gather(1, idx.clamp(min=0).reshape(b, s * t, 1).expand(b, s * t, d)).reshape(b, s, t, d)
    sc = torch.einsum("bshd,bstd->bsht", q.float(), k.float()) * scale
    sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = torch.maximum(sc.max(-1).values, sink.view(1, 1, h))
    e = fast_exp(sc - mx.unsqueeze(-1)) if approx else torch.exp(sc - mx.unsqueeze(-1))
    den = e.sum(-1) + (fast_exp(sink.view(1, 1, h) - mx) if approx else torch.exp(sink.view(1, 1, h) - mx))
    p = e / den.unsqueeze(-1)
    return torch.einsum("bsht,bstd->bshd", p, k.float()).to(q.dtype), p

def run_attn(fast_rms=False, approx_soft=False, rope_kernel=False):
    xf = x
    qr = rms(F.linear(xf, wq_a), q_norm, fast_rms)
    q = F.linear(qr, wq_b).view(1, 128, H, D)
    kv = rms(F.linear(xf, wkv), kv_norm, fast_rms)
    freqs_s = freqs[:128]
    if rope_kernel:
        # use tilelang table kernel on the rope tails (real shapes)
        cos = freqs_s.real.float().contiguous(); sin = freqs_s.imag.float().contiguous()
        pos = torch.arange(128, device=dev, dtype=torch.int32)
        for t in (q[..., -RD:], kv[..., -RD:]):
            rows = t.reshape(-1, RD)
            y = ak.rope_table(rows, cos, sin, pos.repeat_interleave(rows.size(0)//128))
            t.copy_(y.view_as(t))
    else:
        apply_rope(q[..., -RD:], freqs_s)
        apply_rope(kv[..., -RD:], freqs_s)
    idx = window_idxs_prefill(128, win, device=dev).unsqueeze(0).expand(1, -1, -1).contiguous()
    if approx_soft:
        o, p = attn_ref_soft(q, kv, sink, idx, approx=True)
    else:
        o, p = attn_ref_soft(q, kv, sink, idx, approx=False)
    apply_rope(o[..., -RD:], freqs_s, inverse=True)
    o = o.view(1, 128, cfg.o_groups, -1)
    wo = wo_a.view(cfg.o_groups, cfg.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo)
    return F.linear(o.flatten(2), wo_b), p

torch.cuda.synchronize()
x_ref, p_ref = run_attn()
x_fr, _ = run_attn(fast_rms=True)
x_fs, p_fs = run_attn(approx_soft=True)
x_rk, p_rk = run_attn(rope_kernel=True)
def rel(a, b): return (a.float() - b.float()).abs().mean().item() / b.float().abs().mean().clamp_min(1e-30).item()
print(f"\nA) real-weight attention output error: fast_rsqrt {rel(x_fr,x_ref):.5f} | fast_exp softmax {rel(x_fs,x_ref):.5f} | rope_table {rel(x_rk,x_ref):.2e}")
kl = (p_ref.clamp_min(1e-30).log() - p_fs.clamp_min(1e-30).log()).mul(p_ref).sum(-1)
kl = kl[p_ref.sum(-1) > 0].mean().item()
print(f"   attention weight KL (fast_exp, real q/kv, 128x64 heads): {kl:.3e}; top1 agreement {(p_ref.argmax(-1)==p_fs.argmax(-1)).float().mean().item():.3f}")
# rope table accuracy vs torch complex path on real q/kv tails
q2 = F.linear(rms(F.linear(x, wq_a), q_norm), wq_b).view(1,128,H,D)
kv2 = rms(F.linear(x, wkv), kv_norm)
qt = q2[..., -RD:].clone(); kt = kv2[..., -RD:].clone()
cos = freqs[:128].real.float().contiguous(); sin = freqs[:128].imag.float().contiguous()
qy = ak.rope_table(qt.reshape(-1, RD), cos, sin, torch.arange(128, device=dev, dtype=torch.int32).repeat_interleave(H))
ky = ak.rope_table(kt.reshape(-1, RD), cos, sin, torch.arange(128, device=dev, dtype=torch.int32))

def torch_rope_2d(x, cos, sin, pos):
    a = x.float()[:, 0::2]; b = x.float()[:, 1::2]
    c = cos[pos]; sn = sin[pos]
    out = torch.empty_like(x.float()); out[:, 0::2] = a*c - b*sn; out[:, 1::2] = a*sn + b*c
    return out.to(x.dtype)
cos = freqs[:128].real.float().contiguous(); sin = freqs[:128].imag.float().contiguous()
qrows = qt.reshape(-1, RD); krows = kt.reshape(-1, RD)
posq = torch.arange(128, device=dev, dtype=torch.int32).repeat_interleave(H)
posk = torch.arange(128, device=dev, dtype=torch.int32)
qy = ak.rope_table(qrows, cos, sin, posq); ky = ak.rope_table(krows, cos, sin, posk)
print(f"   rope table kernel on real q/kv rows: q max err {(qy.float()-torch_rope_2d(qrows,cos,sin,posq).float()).abs().max().item():.2e}, kv max err {(ky.float()-torch_rope_2d(krows,cos,sin,posk).float()).abs().max().item():.2e}")

def bench_r(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s0=torch.cuda.Event(True); e0=torch.cuda.Event(True); s0.record()
    for _ in range(iters): fn()
    e0.record(); torch.cuda.synchronize(); return s0.elapsed_time(e0)/iters
t_qk = bench_r(lambda: ak.rope_table(qrows, cos, sin, posq)); t_kvk = bench_r(lambda: ak.rope_table(krows, cos, sin, posk))
t_t = bench_r(lambda: (torch_rope_2d(qrows,cos,sin,posq), torch_rope_2d(krows,cos,sin,posk)))
print(f"   rope per layer (q 8192 rows + kv 128): table kernel {(t_qk+t_kvk)*1e3:.1f}us, torch {t_t*1e3:.1f}us")

# B) SwiGLU on real expert weights
x64 = x.view(-1, cfg.dim)[:64]
w1 = dequant_fp4_block(load("layers.0.ffn.experts.0.w1.weight").view(torch.float4_e2m1fn_x2), load("layers.0.ffn.experts.0.w1.scale")).to(dev)
w3 = dequant_fp4_block(load("layers.0.ffn.experts.0.w3.weight").view(torch.float4_e2m1fn_x2), load("layers.0.ffn.experts.0.w3.scale")).to(dev)
w2 = dequant_fp4_block(load("layers.0.ffn.experts.0.w2.weight").view(torch.float4_e2m1fn_x2), load("layers.0.ffn.experts.0.w2.scale")).to(dev)
gate = F.linear(x64, w1).float(); up = F.linear(x64, w3).float()
up = up.clamp(-cfg.swiglu_limit, cfg.swiglu_limit); gate = gate.clamp(max=cfg.swiglu_limit)
y_e = (F.silu(gate) * up); y_p = (fast_silu(gate) * up)
out_e = F.linear(y_e.to(torch.bfloat16), w2).float(); out_p = F.linear(y_p.to(torch.bfloat16), w2).float()
print(f"\nB) real expert SwiGLU: intermediate rel {rel(y_p,y_e):.5f}; expert output rel {rel(out_p,out_e):.5f}")

print("\nC) O3 real-weight GEMM: tilelang vs dequant+bf16")
def bench(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(3):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / iters)
    return best

# wq_b real fp8 weight
w = load("layers.0.attn.wq_b.weight").to(dev); s = load("layers.0.attn.wq_b.scale").to(dev)
wbf = dequant_fp8_block(w, s)
xq_all = F.linear(x_all.view(-1, cfg.dim), wq_a)  # [512,1280]
for M in [1, 128, 512]:
    xm = xq_all[:M].contiguous()
    ref = F.linear(xm, wbf)
    out = gemm.fp8_linear(xm, w, s, block=32)
    t_k = bench(lambda: gemm.fp8_linear(xm, w, s, block=32))
    t_b = bench(lambda: F.linear(xm, wbf))
    print(f"  wq_b fp8 M={M:4d}: tl {t_k*1e3:8.2f}us bf16 {t_b*1e3:8.2f}us TFLOPs {2*M*32768*1280/t_k/1e9:5.1f} rel_err {rel(out,ref):.5f}")
# expert w1 real fp4 weight
w1p = load("layers.0.ffn.experts.0.w1.weight").view(torch.float4_e2m1fn_x2)
s1 = load("layers.0.ffn.experts.0.w1.scale")
w1bf = dequant_fp4_block(w1p, s1).to(dev)
w1p = w1p.to(dev); s1 = s1.to(dev)
for M in [1, 8, 64]:
    xm = x_all.view(-1, cfg.dim)[:M].contiguous()
    ref = F.linear(xm, w1bf)
    out = gemm.fp4_linear(xm, w1p, s1, act_block=32)
    t_k = bench(lambda: gemm.fp4_linear(xm, w1p, s1, act_block=32))
    t_b = bench(lambda: F.linear(xm, w1bf))
    print(f"  expert0 w1 fp4 M={M:4d}: tl {t_k*1e3:8.2f}us bf16 {t_b*1e3:8.2f}us rel_err {rel(out,ref):.5f}")
store.close()