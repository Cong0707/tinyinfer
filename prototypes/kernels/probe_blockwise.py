import torch, math, traceback
torch.manual_seed(0)
dev = "cuda"
M, N, K = 128, 2304, 5120

def full_err(tag, fn):
    try:
        r = fn()
        print(f"[OK] {tag}")
        return r
    except Exception as e:
        print(f"[FAIL] {tag}: {type(e).__name__}: {e}")
        return None

def mk(a):
    return a.contiguous()

# --- rowwise correct shapes
a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
sa = (a.float().abs().amax(1, keepdim=True) / 448.0).contiguous()
sw = (w.float().abs().amax(1, keepdim=True) / 448.0).contiguous()   # [N,1] per-row weight scale
af = (a.float() / sa).clamp(-448, 448).to(torch.float8_e4m3fn)
wf = (w.float() / sw).clamp(-448, 448).to(torch.float8_e4m3fn)
print("scale shapes", sa.shape, sw.shape)
c = full_err("fp8 rowwise (scale_b is 1xN)", lambda: torch._scaled_mm(af, wf.t(), sa, sw.t(), out_dtype=torch.bfloat16))
if c is not None:
    ref = (af.float() * sa) @ (wf.float() * sw).t()
    print("  max abs err", (c.float()-ref).abs().max().item(), "max |ref|", ref.abs().max().item())

# --- blockwise 1x128 (scale_a [M, K/128], scale_b [K/128, N])
bK = K // 128
sa = torch.ones(M, bK, device=dev) * 1.0
sb = torch.ones(bK, N, device=dev) * 1.0
af = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
wf = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
c = full_err("fp8 blockwise 1x128", lambda: torch._scaled_mm(af, wf.t(), sa, sb, out_dtype=torch.bfloat16))
if c is not None:
    ref = af.float() @ wf.float().t()
    print("  max abs err", (c.float()-ref).abs().max().item())

# --- blockwise 128x128 (scale_a [M/128, K/128], scale_b [K/128, N/128])
sa = torch.ones(M // 128, bK, device=dev)
sb = torch.ones(bK, N // 128, device=dev)
c = full_err("fp8 blockwise 128x128", lambda: torch._scaled_mm(af, wf.t(), sa, sb, out_dtype=torch.bfloat16))
if c is not None:
    ref = af.float() @ wf.float().t()
    print("  max abs err", (c.float()-ref).abs().max().item())

# --- MXFP8 e8m0: activate scale per row/32; weights per row/32. blocked layout.
def to_blocked(x):
    # x: [rows, cols] float; returns flat blocked layout [ceil(rows/128)*ceil(cols/4)*128*4]
    rows, cols = x.shape
    nr = (rows + 127)//128
    nc = (cols + 3)//4
    pad = torch.zeros(nr*128, nc*4, device=x.device, dtype=x.dtype)
    pad[:rows, :cols] = x
    blocks = pad.view(nr, 128, nc, 4).permute(0, 2, 1, 3).contiguous()
    return blocks.reshape(-1)

def e8m0_scale(x):  # x [rows, K] -> [rows, K//32], scale = 2^ceil(log2(amax/448)) as e8m0
    xb = x.float().abs().amax(-1).clamp_min(1e-30)
    s = torch.pow(2.0, torch.ceil(torch.log2(xb / 448.0)))
    return s.to(torch.float8_e8m0fnu)

for dt, tag in [(torch.float8_e4m3fn, "mxfp8-e8m0")]:
    aq = torch.randn(M, K, device=dev)
    wq = torch.randn(N, K, device=dev)
    sa = e8m0_scale(aq); sw = e8m0_scale(wq)
    af = (aq / sa.float().repeat_interleave(32, 1)).clamp(-448,448).to(dt)
    wf = (wq / sw.float().repeat_interleave(32, 1)).clamp(-448,448).to(dt)
    sa_b = to_blocked(sa.float())
    sb_b = to_blocked(sw.float())
    print("blocked scale sizes", sa_b.numel(), sb_b.numel(), "expected", 128*(M//128)*(K//32), 128*(N//128)*(K//32))
    sa_e = sa_b.view(torch.float8_e8m0fnu) if False else sa_b.to(torch.float8_e8m0fnu)
    sb_e = sb_b.to(torch.float8_e8m0fnu)
    c = full_err(tag, lambda: torch._scaled_mm(af, wf.t(), sa_e, sb_e, out_dtype=torch.bfloat16))
    if c is not None:
        ref = (af.float()*sa.float().repeat_interleave(32,1)) @ (wf.float()*sw.float().repeat_interleave(32,1)).t()
        print("  max abs err", (c.float()-ref).abs().max().item(), "max|ref|", ref.abs().max().item())

# --- fp4 mx: create packed fp4 by manual packing 2 values per byte along K
import struct
def pack_e2m1(vals):  # vals [rows, K] in {0,.5,1,1.5,2,3,4,6}*sign -> uint8 [rows, K//2]
    table = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device=vals.device)
    codes = torch.argmin((vals.unsqueeze(-1) - table).abs(), dim=-1).to(torch.uint8)
    lo = codes[..., 0::2]; hi = codes[..., 1::2]
    return (lo | (hi << 4)).contiguous()

af4 = torch.zeros(M, K//2, dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
wf4 = torch.zeros(N, K//2, dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
sa = torch.ones(M, K//32, device=dev, dtype=torch.float32)
sw = torch.ones(N, K//32, device=dev, dtype=torch.float32)
sa_b = to_blocked(sa).to(torch.float8_e8m0fnu)
sb_b = to_blocked(sw).to(torch.float8_e8m0fnu)
print("fp4 tensors", af4.shape, wf4.shape, af4.stride(), wf4.stride())
c = full_err("fp4xfp4 mx e8m0 (A packed [M,K/2], B packed [N,K/2])", lambda: torch._scaled_mm(af4, wf4.t(), sa_b, sb_b, out_dtype=torch.bfloat16))
c = full_err("fp8 A x fp4 B (mixed)", lambda: torch._scaled_mm(af, wf4.t(), sa_b, sb_b, out_dtype=torch.bfloat16))
