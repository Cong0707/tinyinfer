import torch, time
torch.manual_seed(0)
dev="cuda"
def tryit(tag, fn):
    try:
        r = fn()
        print(f"[OK] {tag}: {r if not torch.is_tensor(r) else (tuple(r.shape), r.dtype, r.device)}")
        return r
    except Exception as e:
        print(f"[FAIL] {tag}: {type(e).__name__}: {str(e)[:600]}")
        return None

M,N,K = 128, 2304, 5120
a = torch.randn(M,K,device=dev,dtype=torch.bfloat16)
w = torch.randn(N,K,device=dev,dtype=torch.bfloat16)
af = a.to(torch.float8_e4m3fn)
wf = w.to(torch.float8_e4m3fn)
sa = torch.tensor(1.0, device=dev)
sb = torch.tensor(1.0, device=dev)
c = tryit("fp8 tensorwise", lambda: torch._scaled_mm(af, wf.t(), sa, sb, out_dtype=torch.bfloat16))
if c is not None:
    ref = af.float() @ wf.float().t()
    print("  max abs err vs fp32 of quantized:", (c.float()-ref).abs().max().item(), "ref scale:", ref.abs().max().item())
    print("  c[:2,:3]", c[:2,:3])

# rowwise
sa2 = a.float().abs().amax(1, keepdim=True)/448.0
sb2 = w.float().abs().amax(0, keepdim=True).t()/448.0  # [1, N]
af2 = (a.float()/sa2).clamp(-448,448).to(torch.float8_e4m3fn)
wf2 = (w.float()/sb2.t()).clamp(-448,448).to(torch.float8_e4m3fn)
c2 = tryit("fp8 rowwise", lambda: torch._scaled_mm(af2, wf2.t(), sa2, sb2, out_dtype=torch.bfloat16))

# fp4 tensorwise? scales must be e8m0 for fp4? try e4m3 NVFP4 later.
# make simple fp4 packed tensors from tiny values in {0,.5,1,1.5,2,3,4,6}*sign
print("float4 dtype ok:", torch.float4_e2m1fn_x2)
try:
    x = torch.zeros(4, 8, device=dev, dtype=torch.float4_e2m1fn_x2)
    print("created fp4 tensor", x.shape, x.stride(), x.element_size())
    print("view uint8", x.view(torch.uint8).shape)
except Exception as e:
    print("[FAIL] create fp4:", type(e).__name__, str(e)[:300])

print("torch version", torch.__version__)
print("gpu mem allocated MB", torch.cuda.memory_allocated()/2**20)
