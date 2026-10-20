import torch, math
torch.manual_seed(0)
dev = "cuda"
M, N, K = 128, 2304, 5120

def attempt(tag, fn):
    try:
        return fn()
    except Exception as e:
        print(f"[FAIL] {tag}: {type(e).__name__}: {str(e)[:1200]}")
        return None

a = torch.randn(M, K, device=dev)
w = torch.randn(N, K, device=dev)
af = a.to(torch.float8_e4m3fn)
wf = w.to(torch.float8_e4m3fn)

# ---- 1x128 blockwise, outer-dim-major (memory order [K/128, M] and [N, K/128])
sa = torch.ones(K // 128, M, device=dev).t()          # [M, 40], stride (1, 128)
sb = torch.ones(N, K // 128, device=dev).t()          # [40, N], stride (1, 2304)
print("1x128 strides", sa.shape, sa.stride(), sb.shape, sb.stride())
c = attempt("1x128 outer-dim-major", lambda: torch._scaled_mm(af, wf.t(), sa, sb, out_dtype=torch.bfloat16))
if c is not None:
    ref = af.float() @ wf.float().t()
    print("  1x128 max abs err", (c.float() - ref).abs().max().item())

# ---- 128x128: inner-dim-major with 16-byte-aligned strides; pad inner dim N/128 to multiple of 4
sa = torch.ones(M // 128, K // 128, device=dev)       # [1, 40]
sb_buf = torch.ones(K // 128, 20, device=dev)         # pad 18 -> 20 so stride 20 floats = 80B
sb = sb_buf[:, : N // 128]
print("128x128 strides", sa.shape, sa.stride(), sb.shape, sb.stride())
c = attempt("128x128 padded", lambda: torch._scaled_mm(af, wf.t(), sa, sb, out_dtype=torch.bfloat16))
if c is not None:
    ref = af.float() @ wf.float().t()
    print("  128x128 max abs err", (c.float() - ref).abs().max().item())

# ---- MXFP8 e8m0 blocked
def to_blocked(x):
    rows, cols = x.shape
    nr = (rows + 127) // 128
    nc = (cols + 3) // 4
    pad = torch.zeros(nr * 128, nc * 4, device=x.device, dtype=x.dtype)
    pad[:rows, :cols] = x
    return pad.view(nr, 128, nc, 4).permute(0, 2, 1, 3).contiguous().reshape(-1)

def mxfp8_quant(x):
    rows, k = x.shape
    v = x.float().view(rows, k // 32, 32)
    amax = v.abs().amax(-1).clamp_min(2.0 ** -120)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = (v / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(rows, k)
    return q, scale

af8, sa_f = mxfp8_quant(a)
wf8, sw_f = mxfp8_quant(w)
sa_e8 = to_blocked(sa_f).to(torch.float8_e8m0fnu)
sb_e8 = to_blocked(sw_f).to(torch.float8_e8m0fnu)
print("mxfp8 scales", sa_e8.shape, sa_e8.numel(), sb_e8.numel())
c = attempt("mxfp8 e8m0 blocked", lambda: torch._scaled_mm(af8, wf8.t(), sa_e8, sb_e8, out_dtype=torch.bfloat16))
if c is not None:
    ref = (af8.float() * sa_f.repeat_interleave(32, 1)) @ (wf8.float() * sw_f.repeat_interleave(32, 1)).t()
    d = (c.float() - ref).abs()
    print("  mxfp8 max abs err", d.max().item(), "mean", d.mean().item(), "max|ref|", ref.abs().max().item())

# ---- FP4 (packed) probes
def pack_e2m1_codes(codes):
    lo = codes[..., 0::2]; hi = codes[..., 1::2]
    return (lo | (hi << 4)).contiguous().to(torch.uint8)

# random legal fp4 codes
codes_a = torch.randint(0, 16, (M, K), device=dev, dtype=torch.uint8)
codes_w = torch.randint(0, 16, (N, K), device=dev, dtype=torch.uint8)
a_p = pack_e2m1_codes(codes_a).view(torch.float4_e2m1fn_x2)
w_p = pack_e2m1_codes(codes_w).view(torch.float4_e2m1fn_x2)
print("packed", a_p.shape, a_p.stride(), w_p.shape, w_p.stride())

# try e8m0 mx scales, several numels
for numel_a in [20480, 40960, 81920]:
    sa_x = torch.ones(numel_a, device=dev, dtype=torch.float8_e8m0fnu)
    sb_numel = numel_a // 128 * ((N + 127)//128) * 128
    sb_x = torch.ones(sb_numel, device=dev, dtype=torch.float8_e8m0fnu)
    r = attempt(f"fp4 e8m0 scale_a numel={numel_a} scale_b={sb_numel}", lambda: torch._scaled_mm(a_p, w_p.t(), sa_x, sb_x, out_dtype=torch.bfloat16))
    if r is not None:
        print("  WORKED; output", r.shape, r.dtype)
        break

# try e4m3 nvfp4 scales
for numel_a in [20480, 40960, 81920]:
    sa_x = torch.ones(numel_a, device=dev, dtype=torch.float8_e4m3fn)
    sb_numel = numel_a // 128 * ((N + 127)//128) * 128
    sb_x = torch.ones(sb_numel, device=dev, dtype=torch.float8_e4m3fn)
    r = attempt(f"fp4 e4m3 scale_a numel={numel_a} scale_b={sb_numel}", lambda: torch._scaled_mm(a_p, w_p.t(), sa_x, sb_x, out_dtype=torch.bfloat16))
    if r is not None:
        print("  WORKED; output", r.shape, r.dtype)
        break
