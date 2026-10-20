import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
import hc_fused as hf
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
M, D, hc = 1, 8, 4
x = torch.randn(M, D, device=dev)
res = torch.randn(M, hc, D, device=dev)
post = torch.randn(M, hc, device=dev).float()
comb = torch.randn(M, hc, hc, device=dev).float()
y = hf.fused_hc_post(x.view(1,1,D), res.view(1,1,hc,D), post.view(1,1,hc), comb.view(1,1,hc,hc))
y = y.view(M, hc, D)
ref = post.unsqueeze(-1)*x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1)*res.unsqueeze(-2), dim=2)
print("rel", ((y.float()-ref.float()).abs().mean()/ref.float().abs().mean()).item())
print("y[0,0,:4]", y[0,0,:4].tolist())
print("ref[0,0,:4]", ref[0,0,:4].tolist())
for h in range(hc):
    d=(y[:,h]-ref[:,h]).abs().max().item()
    print("h",h,"max err",d)
# check whether y equals ref with comb indexed transposed
ref2 = post.unsqueeze(-1)*x.unsqueeze(-2) + torch.sum(comb.transpose(-1,-2).unsqueeze(-1)*res.unsqueeze(-2), dim=2)
print("transposed comb rel", ((y.float()-ref2.float()).abs().mean()/ref2.float().abs().mean()).item())
