import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
torch.set_default_dtype(torch.bfloat16); torch.manual_seed(0)
from sparse_attn_sm120 import sparse_attn_sm120
dev="cuda"; H=64; D=512
def bench(fn,warmup=5,iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/iters
kv=torch.randn(1,640,D,device=dev); sink=torch.randn(H,device=dev).float()
for m in [1,2,4,8,16]:
    q=torch.randn(1,m,H,D,device=dev)
    idx=torch.randint(0,640,(1,m,640),device=dev,dtype=torch.int32)
    o=sparse_attn_sm120(q,kv,sink,idx,D**-0.5); torch.cuda.synchronize()
    ms=bench(lambda: sparse_attn_sm120(q,kv,sink,idx,D**-0.5))
    print(f"m={m}: {ms*1e3:.1f} us/layer, 40 layers {ms*40:.2f} ms")
# greedy verify: argmax compare over vocab tree targets
V=129280; logits=torch.randn(4,V,device=dev)
def greedy_verify():
    return logits.argmax(-1)
greedy_verify(); torch.cuda.synchronize()
print(f"greedy argmax 4x129k: {bench(greedy_verify,10,200)*1e3:.1f} us")
