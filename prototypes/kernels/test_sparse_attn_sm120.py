import sys, os, time, traceback
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
import torch
torch.set_default_dtype(torch.bfloat16); torch.manual_seed(0)
from sparse_attn_sm120 import sparse_attn_sm120
dev="cuda"; D=512; H=64

def ref(q,kv,sink,idx,scale):
    b,m,h,d=q.shape; t=idx.shape[-1]
    valid=idx>=0
    kvv=kv.gather(1, idx.clamp(min=0).reshape(b,m*t,1).expand(b,m*t,d)).reshape(b,m,t,d)
    sc=torch.einsum("bmhd,bmtd->bmht",q.float(),kvv.float())*scale
    sc=sc.masked_fill(~valid.unsqueeze(2),float("-inf"))
    mx=torch.maximum(sc.max(-1).values, sink.view(1,1,h))
    p=torch.exp(sc-mx.unsqueeze(-1)); den=p.sum(-1)+torch.exp(sink.view(1,1,h)-mx)
    return torch.einsum("bmht,bmtd->bmhd",p,kvv.float()).div(den.unsqueeze(-1)).to(q.dtype)

def bench(fn,warmup=5,iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/iters

for m, t, name in [(1, 640, "decode_m1_kv640"), (128, 640, "prefill_m128_kv640"), (1, 128, "window_decode_m1")]:
    q=torch.randn(1,m,H,D,device=dev)
    kv=torch.randn(1,640,D,device=dev)
    sink=torch.randn(H,device=dev).float()
    idx=torch.randint(0,640,(1,m,t),device=dev,dtype=torch.int32)
    if t==128:
        idx=torch.randint(0,640,(1,m,128),device=dev,dtype=torch.int32)
    t0=time.time()
    try:
        o=sparse_attn_sm120(q,kv,sink,idx,D**-0.5)
        torch.cuda.synchronize(); ct=time.time()-t0
        r=ref(q,kv,sink,idx,D**-0.5)
        err=(o.float()-r.float()).abs().max().item()
        ms=bench(lambda: sparse_attn_sm120(q,kv,sink,idx,D**-0.5))
        print(f"[OK] {name}: compile {ct:.1f}s, {ms*1e3:.1f} us, err {err:.4f}")
        if m==1:
            def forty():
                for _ in range(40): sparse_attn_sm120(q,kv,sink,idx,D**-0.5)
            ms40=bench(forty,3,10)
            print(f"   40 layers: {ms40:.2f} ms -> {1000/ms40:.0f} tok/s attention-only")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:500]}")
        traceback.print_exc()