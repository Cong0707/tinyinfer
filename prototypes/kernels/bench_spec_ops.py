import sys, time, math
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
import torch, torch.nn.functional as F
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
import spec_ops as so
from sparse_attn_sm120 import sparse_attn_sm120

def bench(fn, warmup=10, iters=50, reps=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best=min(best,s.elapsed_time(e)/iters)
    return best

H, D, WIN = 64, 512, 128
sink = torch.randn(H, device=dev).float()

print("=== tree attention (branching draft) ===")
for m, parents in [(4, [-1, 0, 0, 1]), (5, [-1, 0, 0, 1, 1]), (8, [-1,0,1,1,2,2,3,3])]:
    start = 512
    kv = torch.randn(1, WIN + m, D, device=dev)
    q = torch.randn(1, m, H, D, device=dev)
    idx = so.tree_idxs(start, WIN, parents)
    o = sparse_attn_sm120(q, kv, sink, idx, D ** -0.5)
    ref = so.ref_attention(q, kv, sink, idx, D ** -0.5)
    err = (o.float() - ref.float()).abs().max().item()
    ms = bench(lambda: sparse_attn_sm120(q, kv, sink, idx, D ** -0.5), 5, 50)
    print(f"tree m={m} (topk={idx.shape[-1]}): {ms*1e3:.1f}us, max err {err:.4f}")

print("\n=== DSpark verify indices (window + block) ===")
for m in [4, 5, 8]:
    start = 512
    kv = torch.randn(1, WIN + m, D, device=dev)
    q = torch.randn(1, m, H, D, device=dev)
    idx = so.dspark_idxs(start, WIN, m)
    o = sparse_attn_sm120(q, kv, sink, idx, D ** -0.5)
    ref = so.ref_attention(q, kv, sink, idx, D ** -0.5)
    err = (o.float() - ref.float()).abs().max().item()
    ms = bench(lambda: sparse_attn_sm120(q, kv, sink, idx, D ** -0.5), 5, 50)
    print(f"dspark block={m}: {ms*1e3:.1f}us, max err {err:.4f}")

print("\n=== rejection sampling (K=4, V=129280) ===")
V, K = 129280, 4
for B in [1, 4]:
    base = torch.randn(B, K, V, device=dev, dtype=torch.float32)
    qp = torch.softmax(base, -1)
    pp = torch.softmax(base * 1.1 + torch.randn(B, K, V, device=dev) * 0.4, -1)
    ids = torch.multinomial(qp.reshape(-1, V), 1).reshape(B, K)
    tnext = torch.softmax(torch.randn(B, V, device=dev), -1)
    al, toks = so.rejection_sample(qp, pp, ids, target_next=tnext)
    torch.cuda.synchronize()
    ms = bench(lambda: so.rejection_sample(qp, pp, ids, target_next=tnext), 5, 100)
    print(f"B={B}: {ms*1e3:.1f}us, mean accepted {al.float().mean().item():.2f}/{K}")

print("\n=== DSpark heads (cost model, vocab 129280, rank 256) ===")
hidden = torch.randn(1, 5120, device=dev)
prev = torch.randint(0, 129280, (1,), device=dev)
markov = so.MockMarkovHead(129280, 256, dev)
conf = so.MockConfidenceHead(5120, 256, dev)
ids, logits, c = so.dspark_block(hidden, prev, markov, conf, block_size=5)
torch.cuda.synchronize()
ms = bench(lambda: so.dspark_block(hidden, prev, markov, conf, block_size=5), 5, 20)
print(f"dspark block (5 tokens, markov head + confidence): {ms*1e3:.1f}us per verify step")
plan = so.plan_verify(c, threshold=0.5, max_block=5)
print("confidence", [round(v,3) for v in c.squeeze(0).tolist()], "-> verify depth", plan.tolist())

print("\n=== end-to-end mock verify step (tree attn + rejection + dspark) ===")
def verify_step():
    ids, logits, c = so.dspark_block(hidden, prev, markov, conf, block_size=4)
    # target logits for draft positions (mock)
    tgt = torch.randn(1, 4, 129280, device=dev)
    qp = torch.softmax(logits[:, :4].float(), -1); pp = torch.softmax(tgt, -1)
    so.rejection_sample(qp, pp, ids[:, 1:5].clamp(max=129279), target_next=torch.softmax(tgt[:, -1], -1))
verify_step(); torch.cuda.synchronize()
ms = bench(verify_step, 3, 10)
print(f"mock verify step: {ms*1e3:.1f}us; equivalent {1000/ms:.0f} steps/s (heads+sampling only)")