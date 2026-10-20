import sys, traceback
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tinyinfer.kernels import sparse_attn_tl

torch.manual_seed(0)
q = torch.randn(1, 8, 64, 512, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(1, 8, 512, dtype=torch.bfloat16, device="cuda")
sink = torch.zeros(64, device="cuda")
idxs = torch.arange(8, device="cuda", dtype=torch.int32).unsqueeze(0).unsqueeze(0).expand(1, 8, -1).contiguous()
try:
    o = sparse_attn_tl(q, kv, sink, idxs, 512 ** -0.5)
    print("OK", o.shape, torch.isfinite(o.float()).all().item())
except Exception:
    traceback.print_exc()
