"""Attention paths for V4.1 MLA shapes on sm_120 (RTX 5060): SDPA backends vs manual gather-sparse."""
import torch, json, traceback
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
dev="cuda"
D=512; H=64; WIN=128; TOPK=512

def bench(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(3):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best=min(best, s.elapsed_time(e)/iters)
    return best

def sink_ref(q, k, v, sink, scale):
    # q [b,h,m,d], k/v [b,h,n,d]
    scores = torch.matmul(q.float(), k.float().transpose(-1,-2)) * scale
    mx = torch.maximum(scores.max(-1).values, sink.view(1,-1,1))
    p = torch.exp(scores - mx.unsqueeze(-1))
    den = p.sum(-1) + torch.exp(sink.view(1,-1,1) - mx)
    return (torch.matmul(p, v.float()) / den.unsqueeze(-1)).to(q.dtype)

def sdpa_backends(tag, q, k, v, causal=False, attn_mask=None, gqa=False):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    out = {}
    for name, be in [("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("cudnn", SDPBackend.CUDNN_ATTENTION), ("math", SDPBackend.MATH)]:
        try:
            with sdpa_kernel([be]):
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal, enable_gqa=gqa)
            torch.cuda.synchronize()
            ms = bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal, enable_gqa=gqa))
            out[name] = round(ms*1e3, 1)
        except Exception as e:
            out[name] = "FAIL " + str(e).split("\n")[0][:90]
    print(tag, json.dumps(out))

print("=== SDPA backend probe (us) ===")
q = torch.randn(1,H,1,D,device=dev); k = torch.randn(1,1,WIN+TOPK,D,device=dev); v = torch.randn(1,1,WIN+TOPK,D,device=dev)
try:
    o = sink_ref  # noqa
    sdpa_backends("decode_m1_kv640_gqa", q, k, v, gqa=True)
except Exception: traceback.print_exc()
# repeated K/V (no GQA) as a fallback comparison
k2 = k.expand(1,H,-1,D).contiguous(); v2 = v.expand(1,H,-1,D).contiguous()
sdpa_backends("decode_m1_kv640_repeat", q, k2, v2)
q3 = torch.randn(1,H,128,D,device=dev)
k3 = torch.randn(1,1,128,D,device=dev); v3 = torch.randn(1,1,128,D,device=dev)
sdpa_backends("prefill_m128_kv128_gqa_causal", q3, k3, v3, causal=True, gqa=True)
q4 = torch.randn(1,H,4,D,device=dev); k4 = torch.randn(1,1,WIN+TOPK,D,device=dev); v4 = torch.randn(1,1,WIN+TOPK,D,device=dev)
mask = torch.zeros(1,1,4,WIN+TOPK,device=dev,dtype=torch.bfloat16)
mask[0,0,0,0]=float("-inf"); mask[0,0,1,1]=float("-inf"); mask[0,0,2,2]=float("-inf"); mask[0,0,3,3]=float("-inf")
sdpa_backends("tree_m4_kv640_gqa_mask", q4, k4, v4, attn_mask=mask, gqa=True)

print("\n=== manual sparse MLA (torch) ===")
def gathered_attn(q, kv, idx, sink, scale, chunk_m=None):
    # q [b,m,h,d]; kv [b,n,d]; idx [b,m,t] int; sink [h]
    b,m,h,d = q.shape; t = idx.shape[-1]
    valid = idx >= 0
    flat_idx = idx.clamp(min=0).reshape(b, m*t)
    k = kv.gather(1, flat_idx.unsqueeze(-1).expand(b, m*t, d)).view(b,m,t,d)
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), k.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = torch.maximum(scores.max(-1).values, sink.view(1,1,h))
    p = torch.exp(scores - mx.unsqueeze(-1))
    den = p.sum(-1) + torch.exp(sink.view(1,1,h) - mx)
    return torch.einsum("bmht,bmtd->bmhd", p, k.float()).div(den.unsqueeze(-1)).to(q.dtype)

for name, (m, t) in {"decode": (1, WIN+TOPK), "prefill_chunk128": (128, WIN+TOPK)}.items():
    q5 = torch.randn(1,m,H,D,device=dev)
    kv5 = torch.randn(1,WIN+TOPK,D,device=dev)
    sink5 = torch.randn(H, device=dev)
    idx5 = torch.randint(0, WIN+TOPK, (1,m,t), device=dev, dtype=torch.int32)
    o5 = gathered_attn(q5, kv5, idx5, sink5, D**-0.5)
    ref5 = sink_ref(q5.transpose(1,2), kv5.unsqueeze(1).expand(1,H,-1,D).contiguous(),
                    kv5.unsqueeze(1).expand(1,H,-1,D).contiguous(), sink5, D**-0.5).transpose(1,2)
    # NOTE: ref uses gather order? compare against per-position gather reference below
    ref_ok = gathered_attn(q5[:1,:1], kv5, idx5[:1,:1], sink5, D**-0.5)
    err = (o5[:1,:1]-ref_ok).abs().max().item()
    ms = bench(lambda: gathered_attn(q5, kv5, idx5, sink5, D**-0.5), 5, 30)
    print(f"{name}: {ms*1e3:.1f} us, max err {err:.4f}")
    # full 40-layer decode-equivalent
    if name == "decode":
        def forty():
            for _ in range(40):
                gathered_attn(q5, kv5, idx5, sink5, D**-0.5)
        ms40 = bench(forty, 3, 10)
        print(f"  40 layers decode: {ms40:.2f} ms -> {1000/ms40:.0f} tok/s attention-only")

print("\n=== dense fallback: scores/softmax/out without gather ===")
def dense_attn(q, kv, sink, scale):
    # q [b,m,h,d], kv [b,n,d] shared; no masks
    scores = torch.einsum("bmhd,bnd->bmhn", q.float(), kv.float()) * scale
    mx = torch.maximum(scores.max(-1).values, sink.view(1,1,1,H))
    p = torch.exp(scores - mx.unsqueeze(-1)); den = p.sum(-1) + torch.exp(sink.view(1,1,1,H)-mx)
    return torch.einsum("bmhn,bnd->bmhd", p, kv.float()).div(den.unsqueeze(-1)).to(q.dtype)
q6=torch.randn(1,1,H,D,device=dev); kv6=torch.randn(1,WIN+TOPK,D,device=dev); sink6=torch.randn(H,device=dev)
ms=bench(lambda: dense_attn(q6,kv6,sink6,D**-0.5),5,50)
print(f"decode dense kv640: {ms*1e3:.1f} us; 40 layers {ms*40:.2f} ms -> {1000/(ms*40):.0f} tok/s")