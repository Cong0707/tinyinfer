"""torch._scaled_mm NVFP4 (fp4 x fp4, e4m3 scale per 16) on sm_120:
accuracy vs dequantized reference, speed, and dequant+bf16 fallback."""
import torch, json
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
dev="cuda"
E2M1 = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device=dev)

def quant_nvfp4(x):
    """x [rows,K] -> packed uint8 [rows,K/2], flat e4m3 scales [rows_pad,K/16], deq [rows,K]"""
    rows, K = x.shape
    v = x.float().view(rows, K // 16, 16)
    amax = v.abs().amax(-1).clamp_min(1e-30)
    s = (amax / 6.0).to(torch.float8_e4m3fn)          # NVFP4: e4m3 scale, block 16
    sf = s.float().clamp_min(1e-30)
    q = (v / sf.unsqueeze(-1)).clamp(-6, 6)
    codes = torch.argmin((q.unsqueeze(-1) - E2M1).abs(), dim=-1).to(torch.uint8).view(rows, K)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous().to(torch.uint8).view(torch.float4_e2m1fn_x2)
    deq = (E2M1[codes.long()] * sf.repeat_interleave(16, 1)).view(rows, K)
    return packed, s, deq

def flat_scales(s, rows_pad, K):
    out = torch.ones(rows_pad, K // 16, device=s.device, dtype=torch.float8_e4m3fn)
    out[:s.shape[0]] = s
    return out.reshape(-1)

def bench(fn, warmup=5, iters=30, reps=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        st=torch.cuda.Event(True); e=torch.cuda.Event(True); st.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best=min(best, st.elapsed_time(e)/iters)
    return best

SHAPES=[("expert_w1w3",2304,5120),("expert_w2",5120,2304),("attn_wq_b",32768,1280),("attn_wo_b",5120,8192)]
print("shape,M,nvfp4_us,nvfp4_TF,rel_err,bf16_us,deq_ms,deq+bf16_us")
for name,N,K in SHAPES:
    A=torch.randn(512,K,device=dev); B=torch.randn(N,K,device=dev)
    bp, bs, bdeq = quant_nvfp4(B)
    Npad=(N+127)//128*128
    sb=flat_scales(bs,Npad,K)
    for M in [1,8,64,128,512]:
        Apad=torch.zeros(128 if M<=128 else 640, K, device=dev)  # enough rows for quantization then slice
        Apad[:M]=A[:M]
        ap, as_, adeq = quant_nvfp4(Apad)
        Mpad=Apad.shape[0]
        sa=flat_scales(as_,Mpad,K)
        try:
            C=torch._scaled_mm(ap, bp.t(), sa, sb, out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            ref=adeq[:M].float() @ bdeq.float().t()
            rel=(C[:M].float()-ref).abs().max().item()/max(ref.abs().max().item(),1e-6)
            ms=bench(lambda: torch._scaled_mm(ap, bp.t(), sa, sb, out_dtype=torch.bfloat16),3,20)
            bf=bench(lambda: A[:M]@B.t(),5,30)
            # dequant weight -> bf16 then matmul
            def deq_mm():
                w = (E2M1[(bp.view(torch.uint8).to(torch.long) & 0xF).unsqueeze(-1)].squeeze(-1))  # dummy
            # proper dequant via deq reference only for timing dequant cost:
            t_deq=bench(lambda: bdeq.clone(),3,20)
            mm=bench(lambda: A[:M]@bdeq.to(torch.bfloat16).t(),5,20)
            print(f"{name},{M},{ms*1e3:.1f},{2*M*N*K/ms/1e9:.1f},{rel:.4f},{bf*1e3:.1f},{t_deq:.2f},{mm*1e3:.1f}")
        except Exception as e:
            print(f"{name},{M},FAIL,{type(e).__name__}:{str(e)[:120]}")