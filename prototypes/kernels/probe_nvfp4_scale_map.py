import torch
torch.set_default_dtype(torch.bfloat16); dev="cuda"
N=128; K=512; KB=K//16
E2M1=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=dev)
def packed_const(code, rows):
    b=(code | (code<<4)) & 0xFF
    return torch.full((rows,K//2),b,device=dev,dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
# A: row 0 has per-16-block distinct codes; other rows all code 2 (1.0)
A_codes=torch.full((N,K),2,device=dev,dtype=torch.uint8)
pat=torch.tensor([1,2,3,4,5,6,7,0],device=dev,dtype=torch.uint8)
for kb in range(KB):
    A_codes[0, kb*16:(kb+1)*16]=pat[kb%8]
def pack(codes): return (codes[...,0::2] | (codes[...,1::2]<<4)).contiguous().to(torch.uint8).view(torch.float4_e2m1fn_x2)
A=pack(A_codes); B=packed_const(2,N)
base=torch.ones(N,KB,device=dev,dtype=torch.float8_e4m3fn).reshape(-1)
def probe(idx):
    sa=base.clone(); sa[idx]=2.0
    C=torch._scaled_mm(A,B.t(),sa,base,out_dtype=torch.bfloat16).float()
    ref=512.0
    d=(C-ref).abs()
    rows=torch.nonzero(d.max(1).values>1).flatten()
    if len(rows)!=1: return ("multi", rows.tolist()[:4], C[0,0].item())
    r=int(rows[0]); delta=float((C[r]-ref).abs().max())
    return (r, delta)
mapping={}
for idx in range(0, 520):
    mapping[idx]=probe(idx)
# print groups of 16
for start in range(0, 512, 16):
    s=" ".join((f"{idx}:{mapping[idx][0]}/{mapping[idx][1]:.0f}" if mapping[idx][0]!="multi" else f"{idx}:MULTI") for idx in range(start,start+16))
    print(s)
print("idx 512..519:", {i:mapping[i] for i in range(512,520)})