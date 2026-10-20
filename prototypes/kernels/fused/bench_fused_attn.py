"""Per-layer attention: PyTorch step-by-step vs fused tilelang kernels (real L0 weights, decode b=1,s=1,kv=129)."""
import sys, time, importlib.util, statistics
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch
import torch.nn.functional as F
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore
from tinyinfer.quant import dequant_fp8_block
from tinyinfer.nn import rms_norm, apply_rope, precompute_freqs_cis
from sparse_attn_sm120 import sparse_attn_sm120
import fused_ops as fo

MODEL = r"E:\Develop\Deepseek\tinyinfer\cache\weights"
cfg = TinyConfig.from_hf(MODEL)
store = ShardStore(MODEL)
p = "layers.0.attn."
wq_a = dequant_fp8_block(store.get(p+"wq_a.weight"), store.get(p+"wq_a.scale")).to(dev)
wq_b = dequant_fp8_block(store.get(p+"wq_b.weight"), store.get(p+"wq_b.scale")).to(dev)
wkv = dequant_fp8_block(store.get(p+"wkv.weight"), store.get(p+"wkv.scale")).to(dev)
wo_a = dequant_fp8_block(store.get(p+"wo_a.weight"), store.get(p+"wo_a.scale")).to(dev)
wo_b = dequant_fp8_block(store.get(p+"wo_b.weight"), store.get(p+"wo_b.scale")).to(dev)
wq_b_fp8 = store.get(p+"wq_b.weight").to(dev); wq_b_s = store.get(p+"wq_b.scale").to(dev)
q_norm = store.get(p+"q_norm.weight").to(dev)
kv_norm = store.get(p+"kv_norm.weight").to(dev)
sink = store.get(p+"attn_sink").float().to(dev)
freqs_cis = precompute_freqs_cis(cfg.qk_rope_head_dim, 1024, 0, cfg.rope_theta, 16, 32.0, 1.0).to(dev)
H, D, RD, OG, OL = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim, cfg.o_groups, cfg.o_lora_rank
x = store.get("embed.weight")[torch.randint(0, cfg.vocab_size, (1, 1))].to(dev)  # [1,1,5120]
start = 256
freqs = freqs_cis[start:start+1]
cos1 = freqs.real.float().contiguous()   # [1,32]
sin1 = freqs.imag.float().contiguous()
# kv: 129 slots, idx 0..128
kv = torch.randn(1, 129, D, device=dev)
idx = torch.arange(129, device=dev, dtype=torch.int32).view(1, 1, -1)
pos = torch.zeros(1, 1, 32, device=dev)

def bench(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(3):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e)/iters)
    return best

def aten_ops(fn, iters=5):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    NON_KERNEL = ("view", "as_strided", "empty", "detach", "record_stream", "contiguous",
                  "slice", "select", "permute", "transpose", "expand", "unflatten", "flatten",
                  "squeeze", "unsqueeze", "alias", "to", "reshape", "t")
    total = 0; launch = 0
    for e in prof.key_averages():
        if not e.key.startswith("aten::"): continue
        total += e.count
        name = e.key.split("::")[-1]
        if not any(name.startswith(n) for n in NON_KERNEL):
            launch += e.count
    return total / iters, launch / iters

def layer_torch():
    xf = x
    qr = rms_norm(F.linear(xf, wq_a), q_norm, cfg.rms_norm_eps)
    q = F.linear(qr, wq_b).view(1, 1, H, D)
    apply_rope(q[..., -RD:], freqs)
    kvl = rms_norm(F.linear(xf, wkv), kv_norm, cfg.rms_norm_eps)
    kvt = kvl[..., -RD:].clone(); apply_rope(kvt, freqs); kvl = torch.cat([kvl[..., :-RD], kvt], dim=-1)
    o = sparse_attn_sm120(q, kv, sink, idx, D ** -0.5)
    fo.fused_rope_inplace(o.view(H, D), cos1, sin1, RD, inverse=True)
    o = o.view(1, 1, OG, -1)
    wo = wo_a.view(OG, OL, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo)
    return F.linear(o.flatten(2), wo_b)

def layer_fused():
    xf = x
    qr_lin = F.linear(xf, wq_a).view(1, cfg.q_lora_rank)
    qr = fo.fused_rmsnorm_rope(qr_lin, q_norm, torch.empty(1,0,device=dev), torch.empty(1,0,device=dev), 0, cfg.rms_norm_eps)
    aq, asc = ak.act_quant(qr, 32, "ue8m0", torch.float8_e8m0fnu)
    q = fo.fused_gemm_rope(aq, asc, wq_b_fp8, wq_b_s, cos1, sin1, head_dim=D, rope_dim=RD).view(1, 1, H, D)
    kv_lin = F.linear(xf, wkv).view(1, D)
    kvl = fo.fused_rmsnorm_rope(kv_lin, kv_norm, cos1, sin1, RD, cfg.rms_norm_eps)
    o = sparse_attn_sm120(q, kv, sink, idx, D ** -0.5)
    apply_rope(o[..., -RD:], freqs, inverse=True)
    o = o.view(1, 1, OG, -1)
    wo = wo_a.view(OG, OL, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo)
    return F.linear(o.flatten(2), wo_b)

import kernel as ak

o_t = layer_torch(); o_f = layer_fused(); torch.cuda.synchronize()
rel = (o_f.float()-o_t.float()).abs().mean().item()/o_t.float().abs().mean().item()
print(f"layer output rel err (fused fp8 vs torch bf16): {rel:.4f}")
# fair fused-vs-fp8-reference: torch path with dequantized fp8 wq_b
wq_b_fp8_bf = dequant_fp8_block(wq_b_fp8, wq_b_s)
def layer_torch_fp8():
    xf = x
    qr = rms_norm(F.linear(xf, wq_a), q_norm, cfg.rms_norm_eps)
    aq, _ = ak.act_quant(qr, 32, "ue8m0", torch.float8_e8m0fnu)
    adeq = aq.float() * _.float().repeat_interleave(32, -1)
    q = F.linear(adeq.to(torch.bfloat16), wq_b_fp8_bf).view(1,1,H,D)
    apply_rope(q[..., -RD:], freqs)
    kvl = rms_norm(F.linear(xf, wkv), kv_norm, cfg.rms_norm_eps)
    kvt = kvl[..., -RD:].clone(); apply_rope(kvt, freqs); kvl = torch.cat([kvl[..., :-RD], kvt], dim=-1)
    o = sparse_attn_sm120(q, kv, sink, idx, D ** -0.5)
    apply_rope(o[..., -RD:], freqs, inverse=True)
    o = o.view(1,1,OG,-1); wo = wo_a.view(OG,OL,-1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo)
    return F.linear(o.flatten(2), wo_b)
of8 = layer_torch_fp8(); torch.cuda.synchronize()
print("layer fused vs torch-fp8-weight rel err:", ((o_f.float()-of8.float()).abs().mean()/of8.float().abs().mean()).item())

def med(fn, rounds=5):
    return statistics.median([bench(fn, 5, 30) for _ in range(rounds)])
t_t = med(layer_torch); t_f = med(layer_fused)
print(f"per-layer decode (median of 5): torch {t_t*1e3:.1f}us, fused {t_f*1e3:.1f}us ({t_t/t_f:.2f}x)")
c_t_all, c_t = aten_ops(layer_torch); c_f_all, c_f = aten_ops(layer_fused)
print(f"aten ops per layer: torch total {c_t_all:.1f} / launch-ish {c_t:.1f}; fused total {c_f_all:.1f} / launch-ish {c_f:.1f}")
print(f"estimated GPU launches per layer: torch {c_t:.1f} (+1 tilelang attn) = {c_t+1:.1f}; fused {c_f+6:.1f} (6 tilelang kernels)")

# component timings with real weights
qr_lin = F.linear(x, wq_a).view(1, cfg.q_lora_rank)
t1 = bench(lambda: fo.fused_rmsnorm_rope(qr_lin, q_norm, torch.empty(1,0,device=dev), torch.empty(1,0,device=dev), 0, cfg.rms_norm_eps))
t2 = bench(lambda: rms_norm(qr_lin, q_norm, cfg.rms_norm_eps))
print(f"q_norm: fused {t1*1e3:.1f}us vs torch {t2*1e3:.1f}us")
aq, asc = ak.act_quant(qr_lin, 32, "ue8m0", torch.float8_e8m0fnu)
t3 = med(lambda: fo.fused_gemm_rope(aq, asc, wq_b_fp8, wq_b_s, cos1, sin1, D, RD))
qr_n = rms_norm(qr_lin, q_norm, cfg.rms_norm_eps)
def torch_q_rope():
    q = F.linear(qr_n, wq_b).view(1,1,H,D)
    apply_rope(q[..., -RD:], freqs); return q
t4 = med(torch_q_rope)
def torch_q_gemm_only():
    return F.linear(qr_n, wq_b)
t4g = med(torch_q_gemm_only)
def torch_q_rope_only():
    q = F.linear(qr_n, wq_b).view(1,1,H,D); t = q[..., -RD:].clone(); apply_rope(t, freqs); return t
t4r = med(torch_q_rope_only)
print(f"wq_b+rope (median of 5): fused {t3*1e3:.1f}us vs torch F.linear+rope {t4*1e3:.1f}us ({t4/t3:.2f}x; matmul alone {t4g*1e3:.1f}, matmul+rope-tail {t4r*1e3:.1f})")
kv_lin = F.linear(x, wkv).view(1, D)
t5 = bench(lambda: fo.fused_rmsnorm_rope(kv_lin, kv_norm, cos1, sin1, RD, cfg.rms_norm_eps))
def torch_kv():
    kvl = rms_norm(kv_lin.view(1, 1, D), kv_norm, cfg.rms_norm_eps)
    t = kvl[..., -RD:].clone(); apply_rope(t, freqs)
    return torch.cat([kvl[..., :-RD], t], dim=-1).view(1, D)
t6 = bench(torch_kv)
print(f"kv norm+rope: fused {t5*1e3:.1f}us vs torch {t6*1e3:.1f}us")
print("cache read ok")
# rope-only inverse timing
o_dummy = torch.randn(H, D, device=dev)
t7 = bench(lambda: fo.fused_rope_inplace(o_dummy, cos1, sin1, RD, inverse=True))
def torch_inv():
    t = o_dummy[:, -RD:].clone(); apply_rope(t.view(1, 1, H, RD), freqs, inverse=True)
t8 = bench(torch_inv)
print(f"inverse rope on o: fused {t7*1e3:.1f}us vs torch {t8*1e3:.1f}us")