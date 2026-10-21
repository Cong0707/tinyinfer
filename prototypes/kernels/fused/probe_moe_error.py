import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
from tinyinfer.weights import ShardStore
import moe_fused as mf, kernel as ak
store=ShardStore(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
def g(n): return store.get("layers.0.ffn.experts.0."+n).to(dev)
w1p=g("w1.weight").view(torch.float4_e2m1fn_x2); s1=g("w1.scale")
w3p=g("w3.weight").view(torch.float4_e2m1fn_x2); s3=g("w3.scale")
w2p=g("w2.weight").view(torch.float4_e2m1fn_x2); s2=g("w2.scale")
def rel(a,b): return (a.float()-b.float()).abs().mean().item()/b.float().abs().mean().clamp_min(1e-30).item()
M=8; x=torch.randn(M,5120,device=dev); w=torch.rand(M,device=dev).float()
aq,asc=ak.act_quant(x,32,"ue8m0",torch.float8_e8m0fnu)
g1=ak.fp4_gemm(aq,asc,w1p,s1,torch.float8_e8m0fnu,act_block_size=32)
u1=ak.fp4_gemm(aq,asc,w3p,s3,torch.float8_e8m0fnu,act_block_size=32)
g1f=torch.clamp(g1.float(),max=10.0); u1f=torch.clamp(u1.float(),-10.0,10.0)
h_ref=(g1f/(1+torch.exp(-g1f)))*u1f
h_f=mf.fused_fp4_swiglu_pq(aq,asc,w1p,s1,w3p,s3,10.0,16,128)
print("h rel fused vs ref-float:", rel(h_f,h_ref), " vs ref-bf16:", rel(h_f,h_ref.to(torch.bfloat16)))
hq,hs=ak.act_quant(h_ref.to(torch.bfloat16),32,"ue8m0",torch.float8_e8m0fnu)
y_ref=ak.fp4_gemm(hq,hs,w2p,s2,torch.float8_e8m0fnu,act_block_size=32)
y_ref_w=(y_ref* w.view(-1,1))
y_f_down=mf.fused_fp4_down_pq(hq,hs,w2p,s2,w,16,128)
print("down rel fused vs ref (weights after):", rel(y_f_down, y_ref_w), " vs bf16 ref:", rel(y_f_down, y_ref_w.to(torch.bfloat16)))
# full compare with reference matching our scaling order (weights after) vs torch_step (same)
print("full fused vs torch step:", rel(mf.fused_fp4_down_pq(ak.act_quant(h_f,32,"ue8m0",torch.float8_e8m0fnu)[0], ak.act_quant(h_f,32,"ue8m0",torch.float8_e8m0fnu)[1], w2p,s2,w,16,128), y_ref_w))
