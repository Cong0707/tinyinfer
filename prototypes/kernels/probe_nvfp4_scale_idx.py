import torch
torch.set_default_dtype(torch.bfloat16); dev="cuda"
N=128; K=512; KB=K//16
def packed_const(code, rows):
    b=(code | (code<<4)) & 0xFF
    return torch.full((rows,K//2),b,device=dev,dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
A=packed_const(2,N); B=packed_const(2,N)
base=torch.ones(N,KB,device=dev,dtype=torch.float8_e4m3fn).reshape(-1)
def run(idx, val=2.0):
    sa=base.clone(); sa[idx]=val
    C=torch._scaled_mm(A,B.t(),sa,base,out_dtype=torch.bfloat16)
    r=C.float()
    diff=(r-512).abs()
    rows=torch.nonzero(diff.max(1).values>1).flatten()
    cols=torch.nonzero(diff.max(0).values>1).flatten()
    return r[0,0].item(), rows.tolist()[:5], len(rows), cols.tolist()[:5], len(cols)
for idx in [0,1,2,15,16,127,128,129,KB-1,KB,2*KB,127*KB+5]:
    try:
        v,rows,nr,cols,nc=run(idx)
        print(f"idx={idx:5d} C00={v:.1f} changed_rows={rows} n={nr} changed_cols={cols} nc={nc}")
    except Exception as e:
        print("idx",idx,"FAIL",type(e).__name__,str(e)[:100])
