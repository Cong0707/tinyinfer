"""Decode/prefill/speculative primitives + CPU-assist overhead on RTX 5060 laptop."""
import torch, time, json
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
H=64; D=512; VOCAB=129280

def bench(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(3):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best=min(best, s.elapsed_time(e)/iters)
    return best

# ---------- fast torch decode attention (shared KV, sink as extra logit) ----------
def attn_decode(q, k, sink, scale, s2=None):
    # q [H,D], k [T,D]; sink [H]
    T = k.shape[0]
    if s2 is None or s2.shape[0]!=q.shape[0] or s2.shape[1]!=T+1:
        s2 = torch.empty(q.shape[0], T+1, device=q.device, dtype=torch.float32)
    scores = torch.mm(q, k.t()).float().mul_(scale)
    s2 = torch.empty(q.shape[0], T+1, device=q.device, dtype=torch.float32)
    s2[:, :T] = scores; s2[:, T] = sink
    p = torch.softmax(s2, dim=-1)
    return torch.mm(p[:, :T].to(q.dtype), k)

q=torch.randn(H,D,device=dev); sink=torch.randn(H,device=dev).float()
for T,name in [(128,"window128"),(640,"window+topk640")]:
    k=torch.randn(T,D,device=dev)
    o=attn_decode(q,k,sink,D**-0.5); torch.cuda.synchronize()
    ms=bench(lambda: attn_decode(q,k,sink,D**-0.5))
    def forty():
        for _ in range(40): attn_decode(q,k,sink,D**-0.5)
    ms40=bench(forty,3,10)
    print(f"torch decode attn {name}: {ms*1e3:.1f} us/layer, 40 layers {ms40:.2f} ms -> {1000/ms40:.0f} tok/s (attn only)")

# ---------- prefill: dense masked attention, S=128 ----------
S=128; N=640
qs=torch.randn(S,H,D,device=dev); kv=torch.randn(N,D,device=dev)
valid=torch.rand(S,N,device=dev)<0.5; valid[:, :64]=True
mask=torch.where(valid, torch.zeros((),device=dev), torch.full((),float("-inf"),device=dev))  # [S,N]
def prefill_dense():
    sc=torch.matmul(qs.reshape(S*H,D), kv.t()).float().view(S,H,N).mul_(D**-0.5)
    sc=sc+mask[:,None,:]
    mx=torch.maximum(sc.max(-1).values, sink.view(1,H))
    p=torch.exp(sc-mx.unsqueeze(-1))
    den=p.sum(-1)+torch.exp(sink.view(1,H)-mx)
    return torch.matmul((p/den.unsqueeze(-1)).view(S*H,N).to(kv.dtype), kv).view(S,H,D)
prefill_dense(); torch.cuda.synchronize()
ms=bench(prefill_dense,5,20)
print(f"torch prefill dense S=128 N=640: {ms*1e3:.0f} us/layer, 40 layers {ms*40:.1f} ms -> {128/(ms*40)*1000:.0f} tok/s prefill (attn only)")

# ---------- speculative: 4-token tree verify attention ----------
T=4; k=torch.randn(N,D,device=dev); qt=torch.randn(T,H,D,device=dev)
idxs=torch.stack([torch.randperm(N,device=dev)[:640] for _ in range(T)]).long()
def tree_verify():
    outs=[]
    for i in range(T):
        outs.append(attn_decode(qt[i], k, sink, D**-0.5))
    return torch.stack(outs)
tree_verify(); torch.cuda.synchronize()
ms=bench(tree_verify,5,30)
def forty_tree():
    for _ in range(40): tree_verify()
ms40=bench(forty_tree,3,10)
print(f"tree verify 4 tok x 640 kv: {ms*1e3:.1f} us/layer, 40 layers {ms40:.2f} ms")

# ---------- rejection sampling (4 draft tokens, vocab 129280) ----------
K=4
p=torch.softmax(torch.randn(K,VOCAB,device=dev).float(),-1)
q=torch.softmax(torch.randn(K,VOCAB,device=dev).float(),-1)
draft=torch.multinomial(p,1).squeeze(-1)
def reject():
    pv=p.gather(1,draft[:,None]).squeeze(1); qv=q.gather(1,draft[:,None]).squeeze(1)
    ratio=(pv/qv).clamp(max=1.0)
    u=torch.rand(K,device=dev)
    accept=(u<ratio).to(torch.int32)
    # accepted prefix length
    return torch.cumprod(accept,0).sum()
reject(); torch.cuda.synchronize()
ms=bench(reject,10,200)
t0=time.perf_counter(); r=reject(); torch.cuda.synchronize(); tl=(time.perf_counter()-t0)*1e3
print(f"rejection sampling 4 tok: kernel {ms*1e3:.1f} us, wall {tl:.2f} ms (incl sync)")

# ---------- CPU assist: router, sampling, PCIe ----------
cpu=torch.device("cpu")
gate_w=torch.randn(384,5120,dtype=torch.bfloat16)
x=torch.randn(1,5120,dtype=torch.bfloat16)
def cpu_router():
    s=torch.nn.functional.linear(x.float(), gate_w.float())
    return torch.topk(s,k=6,dim=-1).indices
cpu_router()
t0=time.perf_counter()
for _ in range(100): cpu_router()
cpu_router_ms=(time.perf_counter()-t0)*10
print(f"CPU router 384x5120 fp32: {cpu_router_ms:.3f} ms/call")

gpu_logits=torch.randn(VOCAB,device=dev)
cpu_logits=torch.empty(VOCAB)
t0=time.perf_counter(); cpu_logits.copy_(gpu_logits, non_blocking=False); torch.cuda.synchronize(); d2h=(time.perf_counter()-t0)*1e3
t0=time.perf_counter()
for _ in range(20):
    torch.softmax(cpu_logits,dim=-1).argmax()
cpu_samp=(time.perf_counter()-t0)*50
print(f"GPU->CPU logits 129280 bf16: {d2h:.2f} ms; CPU softmax+argmax: {cpu_samp:.3f} ms")

expert=torch.randn(2304,5120,dtype=torch.bfloat16)  # 23.6MB bf16
xp=torch.randn(1,5120,dtype=torch.bfloat16)
def cpu_expert():
    return torch.nn.functional.linear(xp, expert)
cpu_expert()
t0=time.perf_counter()
for _ in range(20): cpu_expert()
cpu_expert_ms=(time.perf_counter()-t0)*50
print(f"CPU one expert gate 2304x5120 bf16 (M=1): {cpu_expert_ms:.3f} ms; 6 experts*40 layers -> {cpu_expert_ms*240:.1f} ms/token = {1000/(cpu_expert_ms*240):.1f} tok/s")
# PCIe H2D 23.6MB
pin=torch.empty_like(expert, pin_memory=True); pin.copy_(expert)
buf=torch.empty_like(expert, device=dev)
for _ in range(3): buf.copy_(pin, non_blocking=True)
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(20):
    buf.copy_(pin, non_blocking=True)
torch.cuda.synchronize()
h2d=(time.perf_counter()-t0)/20
print(f"PCIe H2D 23.6MB pinned: {h2d*1e3:.2f} ms -> {23.6/(h2d*1e3):.1f} GB/s")
# sync overhead
t0=time.perf_counter()
for _ in range(100):
    (gpu_logits[0]).item()
print(f"GPU->CPU scalar .item() sync: {(time.perf_counter()-t0)*10:.3f} ms")