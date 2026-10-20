import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
import hc_fused as hf
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
M, D, hc = 1, 8, 4
x = torch.randn(M, D, device=dev)
res = torch.randn(M, hc, D, device=dev)
def run(post, comb):
    return hf.fused_hc_post(x.view(1,1,D), res.view(1,1,hc,D), post.view(1,1,hc), comb.view(1,1,hc,hc)).view(M,hc,D)
# 1) post=1, comb=0 -> y=x
y = run(torch.ones(M,hc,device=dev), torch.zeros(M,hc,hc,device=dev))
print("case1 x equal:", torch.allclose(y, x.unsqueeze(1).expand(M,hc,D)), "y0", y[0,0,:3].tolist(), "x", x[0,:3].tolist())
# 2) post=0, comb=identity -> y=res
y = run(torch.zeros(M,hc,device=dev), torch.eye(hc).unsqueeze(0).to(dev))
print("case2 res equal:", torch.allclose(y, res), "y", y[0,0,:3].tolist(), "res", res[0,0,:3].tolist())
# 3) post=0, comb=ones -> y[h]=sum res
y = run(torch.zeros(M,hc), torch.ones(M,hc,hc,device=dev))
ref = res.sum(1)
print("case3 sum equal:", torch.allclose(y, ref.unsqueeze(1).expand(M,hc,D), atol=1e-2), "y0", y[0,0,:3].tolist(), "ref", ref[0,:3].tolist())
# 4) post=0, comb only [0,1]=1 -> y[h]=res[1] for all h
comb = torch.zeros(M,hc,hc,device=dev); comb[:,:,1]=1
y = run(torch.zeros(M,hc), comb)
print("case4 res1 equal:", torch.allclose(y, res[:,1:2].expand(M,hc,D), atol=1e-2), "y0", y[0,0,:3].tolist(), "res1", res[0,1,:3].tolist())