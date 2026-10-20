import torch, importlib.util, os, traceback
torch.set_default_dtype(torch.bfloat16)
ROOT = r"E:\Develop\Deepseek\tinyinfer"
spec = importlib.util.spec_from_file_location("official_kernel", os.path.join(ROOT,"reference","official","kernel.py"))
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev="cuda"
def mk(M,N,K):
    aq = torch.randn(M,K,device=dev).to(torch.float8_e4m3fn)
    asc = torch.ones(M, K//32, device=dev, dtype=torch.float8_e8m0fnu)
    bpacked = torch.zeros(N, K//2, device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    bsc = torch.ones(N, K//32, device=dev, dtype=torch.float8_e8m0fnu)
    return aq, asc, bpacked, bsc
for (M,N,K) in [(1,2304,5120),(8,2304,5120),(1,128,5120),(1,2304,512),(128,2304,5120)]:
    aq, asc, bp, bsc = mk(M,N,K)
    try:
        c = kern.fp4_gemm(aq, asc, bp, bsc, scale_dtype=torch.float8_e8m0fnu, act_block_size=32)
        print("OK", M,N,K, c.shape)
    except Exception as e:
        print("FAIL", M,N,K, type(e).__name__, str(e)[:300])
