import sys, torch, time, math
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
import approx_kernels as ak
torch.manual_seed(0); dev="cuda"
torch.set_default_dtype(torch.bfloat16)
# rmsnorm correctness
for dim in [512, 1280]:
    x = torch.randn(37, dim, device=dev); w = torch.randn(dim, device=dev)
    y0 = ak.rmsnorm(x, w, "exact"); yf = ak.rmsnorm(x, w, "fast")
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-20) * w.float()).to(torch.bfloat16)
    print(f"rmsnorm dim{dim} exact vs ref {(y0.float()-ref.float()).abs().max().item():.5f}, fast rel {(yf.float()-y0.float()).abs().mean().item()/y0.float().abs().mean().item():.5f}")
# swiglu
for mode in ["exact","hw","poly"]:
    g = torch.randn(37, 2304, device=dev); u = torch.randn(37, 2304, device=dev)
    y = ak.swiglu(g, u, mode)
    ref = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16)
    print(f"swiglu {mode}: rel err {(y.float()-ref.float()).abs().mean().item()/ref.float().abs().mean().item():.6f}")
# softmax
for mode in ["exact","hw","poly"]:
    s = torch.randn(32, 640, device=dev, dtype=torch.float32) * 2
    p = ak.softmax(s, mode)
    ref = torch.softmax(s, -1)
    kl = (ref*(ref.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum(-1).mean().item()
    print(f"softmax {mode}: max abs {(p-ref).abs().max().item():.3e}, KL {kl:.3e}")
# rope
rows, rd = 37, 64
x = torch.randn(rows, rd, device=dev)
cos = torch.randn(128, rd//2, device=dev).float(); sin = torch.randn(128, rd//2, device=dev).float()
pos = torch.randint(0, 128, (rows,), device=dev)
y = ak.rope_table(x, cos, sin, pos)
xf = x.float(); a = xf[:, 0::2]; b = xf[:, 1::2]; c = cos[pos]; s = sin[pos]
ref = torch.empty_like(xf); ref[:,0::2] = a*c-b*s; ref[:,1::2] = a*s+b*c
ref_b = ref.to(torch.bfloat16).float()
print("rope table err vs bf16 ref:", (y.float()-ref_b).abs().max().item(), " vs fp32 ref:", (y.float()-ref).abs().max().item())