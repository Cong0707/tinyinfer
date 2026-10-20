import torch, json
torch.set_default_dtype(torch.bfloat16); dev="cuda"
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
def bench(fn,warmup=5,iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/iters
for H,D,n in [(32,128,4096),(32,128,640),(64,512,640)]:
    q=torch.randn(1,H,1,D,device=dev); k=torch.randn(1,1,n,D,device=dev); v=torch.randn(1,1,n,D,device=dev)
    out={}
    for name,be in [("flash",SDPBackend.FLASH_ATTENTION),("efficient",SDPBackend.EFFICIENT_ATTENTION),("cudnn",SDPBackend.CUDNN_ATTENTION),("math",SDPBackend.MATH)]:
        try:
            with sdpa_kernel([be]):
                F.scaled_dot_product_attention(q,k,v,enable_gqa=True)
            torch.cuda.synchronize()
            out[name]=round(bench(lambda: F.scaled_dot_product_attention(q,k,v,enable_gqa=True))*1e3,1)
        except Exception as e:
            out[name]="FAIL"
    print(f"H{H} D{D} n{n}:", out)
