import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
import fused_ops as fo
def e8m0(a): return torch.pow(2.0, torch.ceil(torch.log2(a.clamp_min(2.0**-120)/448.0)))
def q_rows32(x):
    M,K=x.shape; s=e8m0(x.float().view(M,K//32,32).abs().amax(-1))
    return (x.float().view(M,K//32,32)/s.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn).view(M,K), s.to(torch.float8_e8m0fnu)
def q_2d32(x):
    N,K=x.shape; v=x.float().view(N//32,32,K//32,32); s=e8m0(v.abs().amax(dim=(1,3)))
    return (v/s[:,None,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).reshape(N,K), s.to(torch.float8_e8m0fnu)
def deq_a(a,s): return (a.float()*s.float().repeat_interleave(32,1))
def deq_b(w,s): return (w.float()*s.float().repeat_interleave(32,0).repeat_interleave(32,1))
M,N,K = 1,512,5120
A=torch.randn(M,K,device=dev); B=torch.randn(N,K,device=dev)
aq,asc=q_rows32(A); bq,bsc=q_2d32(B)
cos1=torch.ones(M,32,device=dev,dtype=torch.float32); sin0=torch.zeros(M,32,device=dev,dtype=torch.float32)
y = fo.fused_gemm_rope(aq,asc,bq,bsc,cos1,sin0,512,64)
ref = deq_a(aq,asc) @ deq_b(bq,bsc).t()
print("identity rope: rel", ((y.float()-ref).abs().mean()/ref.abs().mean()).item(), "max", (y.float()-ref).abs().max().item())
# now real rope, compare only rope-region columns and non-rope columns
cos = torch.randn(M,32,device=dev,dtype=torch.float32); sin = torch.randn(M,32,device=dev,dtype=torch.float32)
y2 = fo.fused_gemm_rope(aq,asc,bq,bsc,cos,sin,512,64)
ref2 = ref.clone(); n=512; rd=64
a=ref2[:, n-rd:n-rd+2*(rd//2):2].clone(); b=ref2[:, n-rd+1:n-rd+2*(rd//2):2].clone()
ref2[:, n-rd:n-rd+2*(rd//2):2] = a*cos - b*sin
ref2[:, n-rd+1:n-rd+2*(rd//2):2] = a*sin + b*cos
ref2 = ref2.to(torch.bfloat16)
print("rope: rel", ((y2.float()-ref2.float()).abs().mean()/ref2.float().abs().mean()).item(), "max", (y2.float()-ref2.float()).abs().max().item())
# per-column error to locate issue
d=(y2.float()-ref2.float()).abs().max(0).values
bad=torch.nonzero(d>1).flatten()
print("bad cols:", bad[:20].tolist(), "count", bad.numel())