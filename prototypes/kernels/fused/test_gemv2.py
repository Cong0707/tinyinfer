import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
import gemv_fp8 as gv
import kernel as ak
def e8m0(a): return torch.pow(2.0, torch.ceil(torch.log2(a.clamp_min(2.0**-120)/448.0)))
def q_2d32(x):
    N,K=x.shape; v=x.float().view(N//32,32,K//32,32); s=e8m0(v.abs().amax(dim=(1,3)))
    return (v/s[:,None,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).reshape(N,K), s.to(torch.float8_e8m0fnu)
def deq_a(a,s): return a.float()*s.float().repeat_interleave(32,-1)
def deq_w(w,s): return w.float()*s.float().repeat_interleave(32,0).repeat_interleave(32,1)
N,K=32768,1280
x=torch.randn(1,K,device=dev); w=torch.randn(N,K,device=dev); wq,ws=q_2d32(w)
aq,asc=ak.act_quant(x.view(1,-1),32,"ue8m0",torch.float8_e8m0fnu); aq=aq.view(-1); asc=asc.view(-1)
ref=deq_a(aq,asc)@deq_w(wq,ws).t()
y=gv.fused_fp8_gemv_pq(aq,asc,wq,ws,rows=16); print("pq rel", ((y.float()-ref).abs().mean()/ref.abs().mean()).item())
for rows in [8,16]:
    try:
        y2=gv.fused_fp8_gemv(x,wq,ws,rows=rows)
        rel=((y2.float()-ref).abs().mean()/ref.abs().mean()).item()
        print("bf16 rows",rows,"rel",rel)
    except Exception as e:
        print("bf16 rows",rows,"FAIL",type(e).__name__,str(e)[:200])
