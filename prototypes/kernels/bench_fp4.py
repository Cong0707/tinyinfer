import torch, importlib.util, os, json
torch.set_default_dtype(torch.bfloat16); torch.manual_seed(0)
dev="cuda"; ROOT=r"E:\Develop\Deepseek\tinyinfer"
spec=importlib.util.spec_from_file_location("k", os.path.join(ROOT,"reference","official","kernel.py"))
kern=importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
def bench(fn, warmup=5, iters=20, reps=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best=min(best, s.elapsed_time(e)/iters)
    return best
def e8m0(a): return torch.pow(2.0, torch.ceil(torch.log2(a.clamp_min(2.0**-120)/448.0)))
def q_rows32(x):
    M,K=x.shape; s=e8m0(x.float().view(M,K//32,32).abs().amax(-1))
    return (x.float().view(M,K//32,32)/s.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn).view(M,K), s.to(torch.float8_e8m0fnu)
def q_2d32(x):
    N,K=x.shape; v=x.float().view(N//32,32,K//32,32); s=e8m0(v.abs().amax(dim=(1,3)))
    return (v/s[:,None,:,None]).clamp(-448,448).to(torch.float8_e4m3fn).reshape(N,K), s.to(torch.float8_e8m0fnu)
E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=dev)
def q_fp4(x):
    N,K=x.shape; v=x.float().view(N,K//32,32)
    amax=v.abs().amax(-1).clamp_min(6*2.0**-126); s=torch.pow(2.0,torch.ceil(torch.log2(amax/6.0)))
    q=(v/s.unsqueeze(-1)).clamp(-6,6)
    codes=torch.argmin((q.unsqueeze(-1)-E2M1).abs(),dim=-1).to(torch.uint8).view(N,K)
    packed=(codes[...,0::2] | (codes[...,1::2]<<4)).contiguous().to(torch.uint8).view(torch.float4_e2m1fn_x2)
    deq=(E2M1[codes.long()]*s.repeat_interleave(32,1)).view(N,K)
    return packed, s.to(torch.float8_e8m0fnu), deq
SHAPES=[("expert_w1w3",2304,5120),("expert_w2",5120,2304),("attn_wq_b",32768,1280),("attn_wo_b",5120,8192),("attn_wkv",512,5120)]
print("shape,M,tl_fp4_us,tl_fp4_TF,err4,tl_fp8_us,tl_fp8_TF,bf16_us")
for name,N,K in SHAPES:
    A=torch.randn(512,K,device=dev); B=torch.randn(N,K,device=dev)
    aq8,asc=q_rows32(A); bp4,bsc4,bdeq=q_fp4(B)
    ref4=(aq8.float()*asc.float().repeat_interleave(32,1))@bdeq.t()
    for M in [1,8,64,128,512]:
        aq=aq8[:M]; as_=asc[:M]
        c4=kern.fp4_gemm(aq,as_,bp4,bsc4,scale_dtype=torch.float8_e8m0fnu,act_block_size=32)
        t4=bench(lambda: kern.fp4_gemm(aq,as_,bp4,bsc4,scale_dtype=torch.float8_e8m0fnu,act_block_size=32),3,20)
        err=(c4.float()-ref4[:M]).abs().max().item()
        bq8,bsc=q_2d32(B)
        c8=kern.fp8_gemm(aq,as_,bq8,bsc,scale_dtype=torch.float8_e8m0fnu,block_size=32)
        t8=bench(lambda: kern.fp8_gemm(aq,as_,bq8,bsc,scale_dtype=torch.float8_e8m0fnu,block_size=32),3,20)
        bf=bench(lambda: A[:M]@B.t(),5,30)
        tf=lambda ms: 2*M*N*K/ms/1e9
        print(f"{name},{M},{t4*1e3:.1f},{tf(t4):.1f},{err:.3f},{t8*1e3:.1f},{tf(t8):.1f},{bf*1e3:.1f}")
