import torch, importlib.util, os, traceback
torch.set_default_dtype(torch.bfloat16); torch.manual_seed(0)
ROOT=r"E:\Develop\Deepseek\tinyinfer"
spec=importlib.util.spec_from_file_location("k", os.path.join(ROOT,"reference","official","kernel.py"))
kern=importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev="cuda"
E2M1 = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device=dev)
def q_fp4(x):
    N,K=x.shape
    v=x.float().view(N,K//32,32)
    amax=v.abs().amax(-1).clamp_min(6*2.0**-126)
    s=torch.pow(2.0,torch.ceil(torch.log2(amax/6.0)))
    q=(v/s.unsqueeze(-1)).clamp(-6,6)
    codes=torch.argmin((q.unsqueeze(-1)-E2M1).abs(),dim=-1).to(torch.uint8)
    packed=(codes[...,0::2] | (codes[...,1::2]<<4)).contiguous().to(torch.uint8)
    print("packed uint8", packed.shape, packed.dtype, packed.stride(), "contig", packed.is_contiguous())
    bp = packed.view(torch.float4_e2m1fn_x2)
    print("bp", bp.shape, bp.stride(), bp.dtype, "elem size", bp.element_size())
    return bp, s.to(torch.float8_e8m0fnu)

N,K=2304,5120; M=1
B=torch.randn(N,K,device=dev)
bp, bsc = q_fp4(B)
print("bsc", bsc.shape, bsc.dtype, bsc.stride(), bsc.is_contiguous())
A=torch.randn(M,K,device=dev)
aq=A.to(torch.float8_e4m3fn); asc=torch.ones(M,K//32,device=dev,dtype=torch.float8_e8m0fnu)
print("aq", aq.shape, aq.is_contiguous())
try:
    c=kern.fp4_gemm(aq, asc, bp, bsc, scale_dtype=torch.float8_e8m0fnu, act_block_size=32)
    print("OK", c.shape, c.dtype)
except Exception:
    traceback.print_exc()
# also try zeros-created packed with same strides
bp2=torch.zeros(N,K//2,device=dev,dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
try:
    c=kern.fp4_gemm(aq, asc, bp2, bsc, scale_dtype=torch.float8_e8m0fnu, act_block_size=32)
    print("OK zeros", c.shape)
except Exception as e:
    print("FAIL zeros", type(e).__name__, str(e)[:200])
