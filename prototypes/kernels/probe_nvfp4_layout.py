import torch
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
dev="cuda"
E2M1=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=dev)
def quant(x, block=16):
    r,K=x.shape; v=x.float().view(r,K//block,block)
    s=(v.abs().amax(-1).clamp_min(1e-30)/6.0).to(torch.float8_e4m3fn); sf=s.float().clamp_min(1e-30)
    q=(v/sf.unsqueeze(-1)).clamp(-6,6)
    codes=torch.argmin((q.unsqueeze(-1)-E2M1).abs(),dim=-1).to(torch.uint8).view(r,K)
    return codes, s, (E2M1[codes.long()]*sf.repeat_interleave(block,1)).view(r,K)
def pack(codes, order):
    if order=="lo": return (codes[...,0::2] | (codes[...,1::2]<<4)).contiguous().to(torch.uint8).view(torch.float4_e2m1fn_x2)
    return ((codes[...,0::2]<<4) | codes[...,1::2]).contiguous().to(torch.uint8).view(torch.float4_e2m1fn_x2)
def blocked(x, variant):
    r,c=x.shape; nr=(r+127)//128; nc=(c+3)//4
    pad=torch.ones(nr*128,nc*4,device=x.device,dtype=x.dtype)
    pad[:r,:c]=x
    if variant=="128x4": o=pad.view(nr,128,nc,4).permute(0,2,1,3)
    elif variant=="4x128": o=pad.view(nr,128,nc,4).permute(0,2,3,1)
    elif variant=="flat": o=pad.view(nr,128,nc,4)
    elif variant=="transposed": o=pad.view(nr,128,nc,4).permute(0,2,1,3).transpose(2,3)
    return o.contiguous().reshape(-1)
def flat(x, rows_pad):
    out=torch.ones(rows_pad,x.shape[1],device=x.device,dtype=torch.float8_e4m3fn); out[:x.shape[0]]=x
    return out.reshape(-1)

M,N,K=128,128,512
A=torch.randn(M,K,device=dev); B=torch.randn(N,K,device=dev)
ca,sa,adeq=quant(A); cb,sb,bdeq=quant(B)
ref=adeq.float()@bdeq.float().t()
print("ref max", ref.abs().max().item())
for order in ["lo","hi"]:
    ap=pack(ca,order); bp=pack(cb,order)
    # dequant ref must match the packing order we use
    refl=(adeq.float()@bdeq.float().t())
    for name,fa,fb in [
        ("flat/flat", flat(sa,128), flat(sb,128)),
        ("block128/flat", blocked(sa,"128x4"), flat(sb,128)),
        ("flat/block128", flat(sa,128), blocked(sb,"128x4")),
        ("block128/block128", blocked(sa,"128x4"), blocked(sb,"128x4")),
        ("block4x128/block4x128", blocked(sa,"4x128"), blocked(sb,"4x128")),
        ("flatT/flatT", flat(sa.t().contiguous(), K//16 if False else 128), flat(sb,128)),
    ]:
        try:
            C=torch._scaled_mm(ap, bp.t(), fa, fb, out_dtype=torch.bfloat16)
            err=(C.float()-refl).abs().max().item()/refl.abs().max().item()
            print(f"order={order} {name:22s} rel_err {err:.4f}")
        except Exception as e:
            print(f"order={order} {name:22s} FAIL {type(e).__name__} {str(e)[:100]}")
