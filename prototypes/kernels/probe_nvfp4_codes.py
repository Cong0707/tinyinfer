import torch
torch.set_default_dtype(torch.bfloat16); dev="cuda"
def packed_const(code, rows, K):
    b=(code | (code<<4)) & 0xFF
    u=torch.full((rows,K//2),b,device=dev,dtype=torch.uint8)
    return u.view(torch.float4_e2m1fn_x2)
def s1(rows,K): return torch.ones(rows*K//16, device=dev, dtype=torch.float8_e4m3fn)
N=128; K=512
for code in [0,1,2,3,4,5,6,7,8,9,15]:
    A=packed_const(code,N,K); B=packed_const(code,N,K)
    try:
        C=torch._scaled_mm(A,B.t(),s1(N,K),s1(N,K),out_dtype=torch.bfloat16)
        print(f"code {code}: C[0,0]={C[0,0].item():.3f}  C mean={C.float().mean().item():.3f}")
    except Exception as e:
        print("code",code,"FAIL",type(e).__name__,str(e)[:120])
