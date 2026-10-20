import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
import hc_fused as hf
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16); dev="cuda"
M, D, hc = 1, 8, 4
x = torch.randn(M, D, device=dev)
res = torch.randn(M, hc, D, device=dev)
def run(post, comb):
    return hf.fused_hc_post(x.view(1,1,D), res.view(1,1,hc,D), post.view(1,1,hc), comb.view(1,1,hc,hc)).view(M,hc,D)
def ref(post, comb):
    return (post.unsqueeze(-1)*x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1)*res.unsqueeze(-2), dim=2)).to(torch.bfloat16)
for name, post, comb in [
    ("post=0,comb=ones", torch.zeros(M,hc,device=dev), torch.ones(M,hc,hc,device=dev)),
    ("post=0,comb=res1", torch.zeros(M,hc,device=dev), torch.tensor([[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]],device=dev,dtype=torch.float32)),
    ("random", torch.randn(M,hc,device=dev).float(), torch.randn(M,hc,hc,device=dev).float()),
]:
    y = run(post, comb); r = ref(post, comb)
    rel=((y.float()-r.float()).abs().mean()/r.float().abs().mean()).item()
    print(name, "rel", rel)
    for h in range(hc):
        print("  h",h,"y",y[0,h,:3].tolist(),"ref",r[0,h,:3].tolist())
