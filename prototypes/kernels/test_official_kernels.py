"""Compile & validate the official DeepSeek-V4.1 tilelang kernels on Windows/sm_120.

Uses reference/official/kernel.py (act_quant / fp8_gemm / fp4_gemm / sparse_attn /
hc_split_sinkhorn) with random tensors shaped like the real model.
"""
import importlib.util, os, sys, time, traceback
import torch
torch.set_default_dtype(torch.bfloat16)

ROOT = r"E:\Develop\Deepseek\tinyinfer"
KERNEL_PY = os.path.join(ROOT, "reference", "official", "kernel.py")
spec = importlib.util.spec_from_file_location("official_kernel", KERNEL_PY)
kern = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kern)
print("loaded official kernel.py; tilelang", __import__("tilelang").__version__)

dev = "cuda"
torch.manual_seed(0)

def evt_time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def e8m0_from_amax(amax):
    amax = torch.clamp(amax, min=2.0 ** -120)
    return torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))

def quant_rows32(x):
    M, K = x.shape
    v = x.float().view(M, K // 32, 32)
    s = e8m0_from_amax(v.abs().amax(-1))
    q = (v / s.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, K)
    return q, s.to(torch.float8_e8m0fnu)

def quant_2d32(x):
    N, K = x.shape
    v = x.float().view(N // 32, 32, K // 32, 32)
    s = e8m0_from_amax(v.abs().amax(dim=(1, 3)))
    q = (v / s[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(N, K), s.to(torch.float8_e8m0fnu)

E2M1 = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device=dev)

def pack_fp4_codes(codes):  # [N,K] uint8 codes -> uint8 [N,K/2]
    lo = codes[..., 0::2]; hi = codes[..., 1::2]
    return (lo | (hi << 4)).contiguous().to(torch.uint8)

def quant_fp4_rows32(x):  # [N,K] -> packed [N,K/2], scales [N,K/32] e8m0, dequant ref
    N, K = x.shape
    v = x.float().view(N, K // 32, 32)
    amax = v.abs().amax(-1).clamp_min(6 * 2.0 ** -126)
    s = e8m0_from_amax(amax)  # 2^ceil(log2(amax/6)) * 6? careful: kernel uses amax/6 then fp4 max 6
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    q = (v / s.unsqueeze(-1)).clamp(-6, 6)
    codes = torch.argmin((q.unsqueeze(-1) - E2M1).abs(), dim=-1).to(torch.uint8).view(N, K)
    packed = pack_fp4_codes(codes)
    deq = E2M1[codes.long()].view(N, K // 32, 32) * s.unsqueeze(-1)
    return packed.view(torch.float4_e2m1fn_x2), s.to(torch.float8_e8m0fnu), deq.reshape(N, K)

results = []

# ---------------- act_quant (bf16 -> fp8, 32-block e8m0) ----------------
try:
    x = torch.randn(256, 5120, device=dev, dtype=torch.bfloat16)
    t0 = time.time()
    q, s = kern.act_quant(x, 32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
    torch.cuda.synchronize(); ct = time.time() - t0
    deq = q.float() * s.float().repeat_interleave(32, dim=-1)
    err = (deq - x.float()).abs().max().item()
    ms = evt_time(lambda: kern.act_quant(x, 32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu), 5, 50)
    print(f"[OK] act_quant 32 e8m0: compile {ct:.1f}s, {ms*1000:.1f} us, max err {err:.4f}")
    results.append(("act_quant", "ok", ms))
except Exception as e:
    print("[FAIL] act_quant:", type(e).__name__, str(e)[:800]); results.append(("act_quant", "fail", None))

# ---------------- fp8_gemm (A fp8 rows32, B fp8 32x32, e8m0) ----------------
try:
    M, N, K = 128, 2304, 5120
    A = torch.randn(M, K, device=dev)
    B = torch.randn(N, K, device=dev)
    aq, asc = quant_rows32(A)
    bq, bsc = quant_2d32(B)
    t0 = time.time()
    C = kern.fp8_gemm(aq, asc, bq, bsc, scale_dtype=torch.float8_e8m0fnu, block_size=32)
    torch.cuda.synchronize(); ct = time.time() - t0
    adq = aq.float() * asc.float().repeat_interleave(32, 1)
    bdq = bq.float() * bsc.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    ref = adq @ bdq.t()
    err = (C.float() - ref).abs().max().item()
    ms = evt_time(lambda: kern.fp8_gemm(aq, asc, bq, bsc, scale_dtype=torch.float8_e8m0fnu, block_size=32), 5, 20)
    tf = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"[OK] fp8_gemm 32x32e8m0 [{M}x{N}x{K}]: compile {ct:.1f}s, {ms*1000:.1f} us, {tf:.1f} TFLOPS, max err {err:.4f}")
    results.append(("fp8_gemm", "ok", ms))
except Exception as e:
    print("[FAIL] fp8_gemm:", type(e).__name__, str(e)[:1200]); results.append(("fp8_gemm", "fail", None))

# ---------------- fp4_gemm (A fp8 per-128, B fp4 32-block e8m0) ----------------
try:
    M, N, K = 128, 2304, 5120
    A = torch.randn(M, K, device=dev)
    B = torch.randn(N, K, device=dev)
    aq, asc = quant_rows32(A)           # act_block_size=32 path
    bp, bsc, bdeq = quant_fp4_rows32(B)
    t0 = time.time()
    C = kern.fp4_gemm(aq, asc, bp, bsc, scale_dtype=torch.float8_e8m0fnu, act_block_size=32)
    torch.cuda.synchronize(); ct = time.time() - t0
    adq = aq.float() * asc.float().repeat_interleave(32, 1)
    ref = adq @ bdeq.t()
    err = (C.float() - ref).abs().max().item()
    ms = evt_time(lambda: kern.fp4_gemm(aq, asc, bp, bsc, scale_dtype=torch.float8_e8m0fnu, act_block_size=32), 5, 20)
    tf = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"[OK] fp4_gemm fp8xfp4 [{M}x{N}x{K}]: compile {ct:.1f}s, {ms*1000:.1f} us, {tf:.1f} TFLOPS(weight), max err {err:.4f}")
    results.append(("fp4_gemm", "ok", ms))
except Exception as e:
    print("[FAIL] fp4_gemm:", type(e).__name__, str(e)[:1200]); results.append(("fp4_gemm", "fail", None))

# ---------------- sparse_attn ----------------
def sparse_ref(q, kv, sink, idxs, scale):
    b, s, h, d = q.shape
    t = idxs.shape[-1]
    valid = idxs >= 0
    k = kv.gather(1, idxs.clamp(min=0).reshape(b, s * t, 1).expand(b, s * t, d)).reshape(b, s, t, d)
    scores = torch.einsum("bshd,bstd->bsht", q.float(), k.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = scores.max(-1).values
    mx = torch.max(mx, sink.view(1, 1, h))
    p = torch.exp(scores - mx.unsqueeze(-1))
    den = p.sum(-1) + torch.exp(sink.view(1, 1, h) - mx)
    return torch.einsum("bsht,bstd->bshd", p, k.float()).div(den.unsqueeze(-1)).to(q.dtype)

try:
    b, m, h, d, n = 1, 32, 64, 512, 640
    q = torch.randn(b, m, h, d, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, device=dev, dtype=torch.bfloat16)
    sink = torch.randn(h, device=dev).float()
    idxs = torch.randperm(n, device=dev).unsqueeze(0).unsqueeze(0).expand(b, m, -1).contiguous().int()
    t0 = time.time()
    o = kern.sparse_attn(q, kv, sink, idxs, d ** -0.5)
    torch.cuda.synchronize(); ct = time.time() - t0
    ref = sparse_ref(q, kv, sink, idxs, d ** -0.5)
    err = (o.float() - ref.float()).abs().max().item()
    ms = evt_time(lambda: kern.sparse_attn(q, kv, sink, idxs, d ** -0.5), 5, 20)
    print(f"[OK] sparse_attn b1 m{m} h{h} d{d} kv{n} topk{idxs.shape[-1]}: compile {ct:.1f}s, {ms*1000:.1f} us, max err {err:.4f}")
    results.append(("sparse_attn_prefill", "ok", ms))
    # decode shape m=1
    q1 = torch.randn(b, 1, h, d, device=dev, dtype=torch.bfloat16)
    idx1 = torch.arange(n, device=dev).unsqueeze(0).unsqueeze(0).expand(b, 1, -1).contiguous().int()
    o1 = kern.sparse_attn(q1, kv, sink, idx1, d ** -0.5)
    torch.cuda.synchronize()
    ref1 = sparse_ref(q1, kv, sink, idx1, d ** -0.5)
    err1 = (o1.float() - ref1.float()).abs().max().item()
    ms1 = evt_time(lambda: kern.sparse_attn(q1, kv, sink, idx1, d ** -0.5), 5, 50)
    print(f"[OK] sparse_attn decode m1: {ms1*1000:.1f} us, max err {err1:.4f}")
    results.append(("sparse_attn_decode", "ok", ms1))
except Exception as e:
    print("[FAIL-expect] official sparse_attn needs 141KB smem > sm120 limit 101376B; use sparse_attn_sm120.py:", type(e).__name__, str(e)[:300]); results.append(("sparse_attn", "fail(smem)", None))

# ---------------- hc_split_sinkhorn ----------------
try:
    b, s, hcm = 2, 64, 4
    mixes = torch.randn(b, s, (2 + hcm) * hcm, device=dev, dtype=torch.float32)
    hc_scale = torch.randn(3, device=dev, dtype=torch.float32)
    hc_base = torch.randn((2 + hcm) * hcm, device=dev, dtype=torch.float32)
    t0 = time.time()
    pre, post, comb = kern.hc_split_sinkhorn(mixes, hc_scale, hc_base, hcm, 20, 1e-6)
    torch.cuda.synchronize(); ct = time.time() - t0
    # reference: tinyinfer implementation
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from tinyinfer.hc import hc_split_sinkhorn as ref_hc
    rpre, rpost, rcomb = ref_hc(mixes, hc_scale, hc_base, hcm, 20, 1e-6)
    err = max((pre - rpre).abs().max().item(), (post - rpost).abs().max().item(), (comb - rcomb).abs().max().item())
    ms = evt_time(lambda: kern.hc_split_sinkhorn(mixes, hc_scale, hc_base, hcm, 20, 1e-6), 5, 50)
    print(f"[OK] hc_split_sinkhorn: compile {ct:.1f}s, {ms*1000:.1f} us, max err {err:.2e}")
    results.append(("hc_split_sinkhorn", "ok", ms))
except Exception as e:
    print("[FAIL] hc_split_sinkhorn:", type(e).__name__, str(e)[:1200]); results.append(("hc", "fail", None))

print("\nSUMMARY", results)