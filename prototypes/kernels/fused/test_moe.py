import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
import moe_fused as mf
import kernel as ak
E2M1=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=dev)
def e8m0(a): return torch.pow(2.0, torch.ceil(torch.log2(a.clamp_min(2.0**-120)/448.0)))
def q_fp4(x, block=32):
    N,K=x.shape; v=x.float().view(N,K//block,block)
    amax=v.abs().amax(-1).clamp_min(6*2.0**-126); s=torch.pow(2.0,torch.ceil(torch.log2(amax/6.0)))
    q=(v/s.unsqueeze(-1)).clamp(-6,6)
    codes=torch.argmin((q.unsqueeze(-1)-E2M1).abs(),dim=-1).to(torch.uint8).view(N,K)
    packed=(codes[...,0::2] | (codes[...,1::2]<<4)).contiguous().to(torch.uint8).view(torch.float4_e2m1fn_x2)
    deq=(E2M1[codes.long()]*s.repeat_interleave(block,1)).view(N,K)
    return packed, s.to(torch.float8_e8m0fnu), deq
M=8; K=5120; N=2304
x=torch.randn(M,K,device=dev)
w1,w3,w2 = torch.randn(N,K,device=dev), torch.randn(N,K,device=dev), torch.randn(K,N,device=dev)
p1,s1,d1=q_fp4(w1); p3,s3,d3=q_fp4(w3); p2,s2,d2=q_fp4(w2)
# reference with official act_quant
aq,asc=ak.act_quant(x,32,"ue8m0",torch.float8_e8m0fnu)
xa=aq.float()*asc.float().repeat_interleave(32,-1)
g=xa@d1.t(); u=xa@d3.t()
g=torch.clamp(g,max=10.0); u=torch.clamp(u,-10,10)
h=(g/(1+torch.exp(-g)))*u
hq,hs=ak.act_quant(h.to(torch.bfloat16),32,"ue8m0",torch.float8_e8m0fnu)
ha=hq.float()*hs.float().repeat_interleave(32,-1)
w=torch.randn(M,device=dev).float()
y_ref=(ha@d2.t())*w[:,None]
y= mf.fused_fp4_down(mf.fused_fp4_swiglu(x,p1,s1,p3,s3,10.0),p2,s2,w)
def rel(a,b): return (a.float()-b.float()).abs().mean().item()/b.float().abs().mean().clamp_min(1e-30).item()
print("moe fused rel", rel(y,y_ref), "shapes", y.shape)
print("max abs", (y.float()-y_ref).abs().max().item())
