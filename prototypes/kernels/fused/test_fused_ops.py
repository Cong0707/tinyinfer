import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
import torch.nn.functional as F
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
import fused_ops as fo

def bench(fn, warmup=10, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/iters
def rel(a,b): return (a.float()-b.float()).abs().mean().item()/b.float().abs().mean().clamp_min(1e-30).item()

def torch_rmsnorm_rope(x, w, cos, sin, rope_dim, eps=1e-20):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    if rope_dim:
        n = x.size(-1); h = rope_dim // 2
        a = y[..., n-rope_dim:n-rope_dim+2*h:2].clone(); b = y[..., n-rope_dim+1:n-rope_dim+2*h:2].clone()
        c = cos; s = sin
        y = y.clone()
        y[..., n-rope_dim:n-rope_dim+2*h:2] = a*c - b*s
        y[..., n-rope_dim+1:n-rope_dim+2*h:2] = a*s + b*c
    return y.to(torch.bfloat16)

print("=== fused_rmsnorm_rope correctness ===")
for rows, N, rd in [(1,512,64),(128,512,64),(128,1280,64),(128,1280,0),(1,5120,64)]:
    x = torch.randn(rows, N, device=dev); w = torch.randn(N, device=dev)
    half = rd//2 if rd else 0
    cos = torch.randn(rows, half, device=dev) if half else torch.empty(rows,0,device=dev)
    sin = torch.randn(rows, half, device=dev) if half else torch.empty(rows,0,device=dev)
    y = fo.fused_rmsnorm_rope(x, w, cos, sin, rd)
    ref = torch_rmsnorm_rope(x, w, cos, sin, rd)
    print(f"rows={rows} N={N} rope={rd}: rel {rel(y,ref):.2e}  max {(y.float()-ref.float()).abs().max().item():.2e}")

print("\n=== fused_rmsnorm_rope timing (decode N=512, rope=64) ===")
x = torch.randn(1, 512, device=dev); w = torch.randn(512, device=dev)
cos = torch.randn(1, 32, device=dev); sin = torch.randn(1, 32, device=dev)
t_f = bench(lambda: fo.fused_rmsnorm_rope(x, w, cos, sin, 64))
t_t = bench(lambda: torch_rmsnorm_rope(x, w, cos, sin, 64))
print(f"fused {t_f*1e3:.2f}us vs torch rmsnorm+rope {t_t*1e3:.2f}us  ({t_t/t_f:.1f}x)")

# quant helpers (same as bench_gemm)
def e8m0(a): return torch.pow(2.0, torch.ceil(torch.log2(a.clamp_min(2.0**-120)/448.0)))
def q_rows32(x):
    M,K=x.shape; s=e8m0(x.float().view(M,K//32,32).abs().amax(-1))
    return (x.float().view(M,K//32,32)/s.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn).view(M,K), s.to(torch.float8_e8m0fnu)
def q_2d32(x):
    N,K=x.shape; v=x.float().view(N//32,32,K//32,32); s=e8m0(v.abs().amax(dim=(1,3)))
    return (v/s[:,None,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).reshape(N,K), s.to(torch.float8_e8m0fnu)
def rope_ref_heads(c, rows, head_dim, rope_dim, cos, sin):
    # c [M,N] -> rotate last rope_dim of each head
    M, N = c.shape; y = c.float().clone()
    y = y.view(M, N//head_dim, head_dim)
    r = rope_dim//2
    a = y[..., head_dim-rope_dim:head_dim-rope_dim+2*r:2].clone(); b = y[..., head_dim-rope_dim+1:head_dim-rope_dim+2*r:2].clone()
    cc = cos.view(M,1,r); ss = sin.view(M,1,r)
    y[..., head_dim-rope_dim:head_dim-rope_dim+2*r:2] = a*cc - b*ss
    y[..., head_dim-rope_dim+1:head_dim-rope_dim+2*r:2] = a*ss + b*cc
    return y.view(M,N).to(torch.bfloat16)

print("\n=== fused_gemm_rope correctness ===")
for M, N, K in [(1,512,5120),(1,32768,1280),(128,512,5120),(64,32768,1280)]:
    A = torch.randn(M, K, device=dev); B = torch.randn(N, K, device=dev)
    aq, asc = q_rows32(A); bq, bsc = q_2d32(B)
    head_dim, rope_dim = 512, 64
    cos = torch.randn(M, rope_dim//2, device=dev); sin = torch.randn(M, rope_dim//2, device=dev)
    y = fo.fused_gemm_rope(aq, asc, bq, bsc, cos, sin, head_dim=head_dim, rope_dim=rope_dim)
    # reference: dequant matmul then rope
    ref_lin = (aq.float()*asc.float().repeat_interleave(32,1)) @ (bq.float()*bsc.float().repeat_interleave(32,0).repeat_interleave(32,1)).t()
    ref = rope_ref_heads(ref_lin, M, head_dim, rope_dim, cos, sin)
    print(f"M={M} N={N} K={K}: rel {rel(y,ref):.2e} max {(y.float()-ref.float()).abs().max().item():.2e}")

print("\n=== fused_gemm_rope timing (decode M=1) ===")
for N, K in [(512, 5120), (32768, 1280)]:
    A = torch.randn(1, K, device=dev); B = torch.randn(N, K, device=dev)
    aq, asc = q_rows32(A); bq, bsc = q_2d32(B)
    cos = torch.randn(1, 32, device=dev); sin = torch.randn(1, 32, device=dev)
    B_bf = (bq.float()*bsc.float().repeat_interleave(32,0).repeat_interleave(32,1)).to(torch.bfloat16)
    t_fused = bench(lambda: fo.fused_gemm_rope(aq, asc, bq, bsc, cos, sin, 512, 64), 5, 50)
    def torch_steps():
        q = F.linear(A, B_bf)
        return rope_ref_heads(q, 1, 512, 64, cos, sin)
    t_torch = bench(torch_steps, 5, 50)
    print(f"N={N} K={K}: fused {t_fused*1e3:.1f}us vs F.linear+rope {t_torch*1e3:.1f}us ({t_torch/t_fused:.2f}x)")