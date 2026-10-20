"""Decode-only fp8 GEMV vs official block_M=32 fp8_gemm (real L0 weights, M=1)."""
import sys, statistics
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch
import torch.nn.functional as F
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
from tinyinfer.weights import ShardStore
from tinyinfer.quant import dequant_fp8_block
from tinyinfer.nn import rms_norm, precompute_freqs_cis
import gemv_fp8 as gv
import kernel as ak
from tinyinfer.config import TinyConfig

store = ShardStore(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
cfg = TinyConfig.from_hf(r"E:\Develop\Deepseek\tinyinfer\cache\weights")

def load(name):
    return store.get("layers.0.attn." + name).to(dev)

def bench(fn, warmup=10, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(3):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / iters)
    return best
def med(fn, rounds=5): return statistics.median([bench(fn) for _ in range(rounds)])
def rel(a, b): return (a.float() - b.float()).abs().mean().item() / b.float().abs().mean().clamp_min(1e-30).item()

x = store.get("embed.weight")[torch.randint(0, cfg.vocab_size, (1,))].to(dev)  # [1,5120]
wq_a_w = load("wq_a.weight"); wq_a_s = load("wq_a.scale")
wq_a_bf = dequant_fp8_block(wq_a_w, wq_a_s)
qr = rms_norm(F.linear(x, wq_a_bf), store.get("layers.0.attn.q_norm.weight").to(dev), cfg.rms_norm_eps)
o = torch.randn(1, 8192, device=dev, dtype=torch.bfloat16)

CASES = [
    ("wq_a", x, wq_a_w, wq_a_s, False, 0),
    ("wq_b", qr, load("wq_b.weight"), load("wq_b.scale"), True, 512),
    ("wkv", x, load("wkv.weight"), load("wkv.scale"), True, 512),
    ("wo_b", o, load("wo_b.weight"), load("wo_b.scale"), False, 0),
]
freqs = precompute_freqs_cis(cfg.qk_rope_head_dim, 1024, 0, cfg.rope_theta, 16, 32.0, 1.0).to(dev)
cos = freqs[300:301].real.float().contiguous(); sin = freqs[300:301].imag.float().contiguous()

print("name,N,K,official_us,gemv_pq_us,gemv_bf16_us,GBs_pq,rel_pq,rel_bf")
for name, xin, w, ws, rope, head_dim in CASES:
    n, k = w.shape
    aq, asc = ak.act_quant(xin.view(1, -1), 32, "ue8m0", torch.float8_e8m0fnu)
    aq = aq.view(-1); asc = asc.view(-1)
    wbf = dequant_fp8_block(w, ws)
    ref = F.linear(xin, wbf)
    # official
    def off():
        with torch.no_grad():
            return ak.fp8_gemm(aq.view(1,-1), asc.view(1,-1), w, ws, torch.float8_e8m0fnu, block_size=32)
    off_out = off()
    t_off = med(off)
    y = gv.fused_fp8_gemv_pq(aq, asc, w, ws, rows=16, rope=rope, head_dim=head_dim, rope_dim=64, cos=cos, sin=sin)
    t_pq = med(lambda: gv.fused_fp8_gemv_pq(aq, asc, w, ws, rows=16, rope=rope, head_dim=head_dim, rope_dim=64, cos=cos, sin=sin))
    y2 = gv.fused_fp8_gemv(xin, w, ws, rows=8)
    t_bf = med(lambda: gv.fused_fp8_gemv(xin, w, ws, rows=8))
    aqo, asco = ak.act_quant(xin.view(1, -1), 32, "ue8m0", torch.float8_e8m0fnu)
    ref_bf = ((aqo.view(-1).float()*asco.view(-1).float().repeat_interleave(32,-1)) @ (w.float()*ws.float().repeat_interleave(32,0).repeat_interleave(32,1)).t()).view(1, n)
    rel_bf = rel(y2, ref_bf)
    # correctness vs same-quant dequant reference (no rope for simplicity: compare pre-rope by dequant matmul)
    refq = ((aq.float()*asc.float().repeat_interleave(32,-1)) @ (w.float()*ws.float().repeat_interleave(32,0).repeat_interleave(32,1)).t()).view(1, n)
    rel_pq = rel(y, refq)
    rope_ref = None
    if rope:
        rope_ref = refq.clone()
        for h in range(n // head_dim):
            base = h * head_dim + head_dim - 64
            a = rope_ref[:, base:base+64:2].clone(); b = rope_ref[:, base+1:base+64:2].clone()
            rope_ref[:, base:base+64:2] = a*cos - b*sin
            rope_ref[:, base+1:base+64:2] = a*sin + b*cos
        rel_pq = rel(y, rope_ref)
    print(f"{name},{n},{k},{t_off*1e3:.1f},{t_pq*1e3:.1f},{t_bf*1e3:.1f},{n*k/t_pq/1e6:.0f},{rel_pq:.2e},{rel_bf:.2e}")
    if rope:
        print(f"   rope variant rel vs dequant+rope: {rel(y, rope_ref):.2e}")

# rows sweep for the two big weights
for name in ["wq_b", "wo_b"]:
    case = [c for c in CASES if c[0] == name][0]
    _, xin, w, ws, rope, head_dim = case
    n, k = w.shape
    aq, asc = ak.act_quant(xin.view(1, -1), 32, "ue8m0", torch.float8_e8m0fnu)
    aq = aq.view(-1); asc = asc.view(-1)
    times = {}
    for rows in [8, 16, 32, 64]:
        try:
            t = med(lambda: gv.fused_fp8_gemv_pq(aq, asc, w, ws, rows=rows, rope=rope, head_dim=head_dim, rope_dim=64, cos=cos, sin=sin))
            times[f"pq{rows}"] = round(t*1e3, 1)
        except Exception as e:
            times[f"pq{rows}"] = "FAIL"
    for rows in [32, 64, 128]:
        try:
            t = med(lambda: gv.fused_fp8_gemv_pq4(aq, asc, w, ws, rows=rows, rope=rope, head_dim=head_dim, rope_dim=64, cos=cos, sin=sin))
            times[f"pq4_{rows}"] = round(t*1e3, 1)
        except Exception as e:
            times[f"pq4_{rows}"] = "FAIL " + type(e).__name__
    print(f"variant sweep {name}: {times}")
# measured effective bandwidth reference: bf16 matmul of wo_b
wbf = dequant_fp8_block(load("wo_b.weight"), load("wo_b.scale"))
t_bf16 = med(lambda: F.linear(o, wbf))
print(f"reference: wo_b bf16 F.linear {t_bf16*1e3:.1f}us, bytes 83.9MB -> {5120*8192*2/t_bf16/1e6:.0f} GB/s")
store.close()