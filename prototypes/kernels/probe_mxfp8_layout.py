import torch
torch.manual_seed(0)
dev = "cuda"

def to_blocked(x, variant):
    rows, cols = x.shape
    nr = (rows + 127) // 128
    nc = (cols + 3) // 4
    pad = torch.zeros(nr * 128, nc * 4, device=x.device, dtype=x.dtype)
    pad[:rows, :cols] = x
    if variant == "128x4":
        out = pad.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    elif variant == "4x128":
        out = pad.view(nr, 128, nc, 4).permute(0, 2, 3, 1)
    elif variant == "rowmajor":
        out = pad.view(nr, 128, nc, 4)
    elif variant == "kfirst":
        out = pad.view(nr, 128, nc, 4).permute(0, 2, 3, 1).transpose(2, 3)
    elif variant == "plain":
        out = x
    return out.contiguous().reshape(-1)

def e8m0_scale(x):
    rows, k = x.shape
    v = x.float().view(rows, k // 32, 32)
    amax = v.abs().amax(-1).clamp_min(2.0 ** -120)
    return torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))

def quant(x):
    rows, k = x.shape
    s = e8m0_scale(x)
    q = (x.float().view(rows, k // 32, 32) / s.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(rows, k)
    return q, s

M, N, K = 128, 128, 1024
A = torch.randn(M, K, device=dev)
B = torch.randn(N, K, device=dev)
aq, sa = quant(A)
bq, sb = quant(B)
ref = (aq.float() * sa.repeat_interleave(32, 1)) @ (bq.float() * sb.repeat_interleave(32, 1)).t()
print("ref max", ref.abs().max().item())

variants = ["128x4", "4x128", "rowmajor", "kfirst"]
ones_a = torch.ones_like(sa).float()
ones_b = torch.ones_like(sb).float()
print("\n-- A layout (B scale = 1) --")
for v in variants:
    try:
        c = torch._scaled_mm(aq, bq.t(), to_blocked(sa, v).to(torch.float8_e8m0fnu),
                             to_blocked(ones_b, "128x4").to(torch.float8_e8m0fnu), out_dtype=torch.bfloat16)
        print(f"  {v:9s} max err {(c.float()-ref).abs().max().item():.3f}")
    except Exception as e:
        print(f"  {v:9s} FAIL {type(e).__name__}: {str(e)[:300]}")
print("\n-- B layout (A scale = 1) --")
for v in variants:
    try:
        c = torch._scaled_mm(aq, bq.t(), to_blocked(ones_a, "128x4").to(torch.float8_e8m0fnu),
                             to_blocked(sb, v).to(torch.float8_e8m0fnu), out_dtype=torch.bfloat16)
        print(f"  {v:9s} max err {(c.float()-ref).abs().max().item():.3f}")
    except Exception as e:
        print(f"  {v:9s} FAIL {type(e).__name__}: {str(e)[:300]}")
print("\n-- both best-guess --")
for va in variants:
    for vb in variants:
        if va != "128x4" and vb != "128x4":
            continue
        try:
            c = torch._scaled_mm(aq, bq.t(), to_blocked(sa, va).to(torch.float8_e8m0fnu),
                                 to_blocked(sb, vb).to(torch.float8_e8m0fnu), out_dtype=torch.bfloat16)
            print(f"  A={va:9s} B={vb:9s} max err {(c.float()-ref).abs().max().item():.3f}")
        except Exception as e:
            print(f"  A={va} B={vb} FAIL {type(e).__name__}: {str(e)[:200]}")
