"""KV-cache quantization error study for MLA latent (512-dim), V4.1 recipe vs int4+Hadamard."""
import torch, math
torch.manual_seed(0); dev="cuda"
S,D=4096,512
kv=torch.randn(S,D,device=dev)
# inject a few outlier channels to make it realistic
kv[:, ::64] *= 8.0

def fp8_row32(x):
    v=x.view(S,D//32,32); amax=v.abs().amax(-1).clamp_min(1e-12)
    s=torch.pow(2.0,torch.ceil(torch.log2(amax/448.0))); q=(v/s.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
    return (q.float()*s.unsqueeze(-1)).view(S,D)

E2M1=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=dev)
def fp4_row16(x):
    v=x.view(S,D//16,16); amax=v.abs().amax(-1).clamp_min(1e-12)
    s=(amax/6.0).to(torch.float8_e4m3fn).float().clamp_min(1e-12)
    q=(v/s.unsqueeze(-1)).clamp(-6,6)
    codes=torch.argmin((q.unsqueeze(-1)-E2M1).abs(),dim=-1)
    return (E2M1[codes]*s.unsqueeze(-1)).view(S,D)

def int4_row32(x, rot=False):
    if rot: x = x @ H
    v=x.view(S,D//32,32); amax=v.abs().amax(-1).clamp_min(1e-12)
    s=amax/7.0; q=(v/s.unsqueeze(-1)).round().clamp(-7,7)
    y=(q*s.unsqueeze(-1)).view(S,D)
    return y @ H.t() if rot else y

def int8_row32(x, rot=False):
    if rot: x=x@H
    v=x.view(S,D//32,32); amax=v.abs().amax(-1).clamp_min(1e-12); s=amax/127.0
    q=(v/s.unsqueeze(-1)).round().clamp(-127,127)
    y=(q*s.unsqueeze(-1)).view(S,D)
    return y@H.t() if rot else y

# normalized Hadamard 512
H=torch.ones(1,1,device=dev)
while H.shape[0]<D:
    H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
H=H/math.sqrt(D)

def rel(a,b): return ((a.float()-b.float()).abs().mean()/b.float().abs().mean()).item()
def cos(a,b): return torch.nn.functional.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0).item()
print("method,bytes/val,rel_err_mean,cosine")
for name,y,bpv in [
    ("fp8_e4m3_row32_e8m0", fp8_row32(kv), 1.0),
    ("fp4_e2m1_row16_e4m3 (V4.1 compressed KV)", fp4_row16(kv), 0.5),
    ("int8_row32", int8_row32(kv), 1.0),
    ("int8_row32_hadamard", int8_row32(kv,rot=True), 1.0),
    ("int4_row32", int4_row32(kv), 0.5),
    ("int4_row32_hadamard", int4_row32(kv,rot=True), 0.5),
]:
    print(f"{name},{bpv},{rel(y,kv):.5f},{cos(y,kv):.6f}")

# attention-score error using random q on reconstructed kv
q=torch.randn(64,D,device=dev)
s_ref=q@kv.t()
for name,y,bpv in [("fp8",fp8_row32(kv),1.0),("fp4_16",fp4_row16(kv),0.5),("int4_had",int4_row32(kv,rot=True),0.5),("int8_had",int8_row32(kv,rot=True),1.0)]:
    s=q@y.t()
    print(f"score_err {name}: rel {((s-s_ref).abs().mean()/s_ref.abs().mean()).item():.5f}")
