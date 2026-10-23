
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch
torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda")
MD = r"E:\Models\DeepSeek-V4.1-Flash-1017"
import model as official
import kernel as offk
from tinyinfer.weights import ShardStore
from tinyinfer.config import TinyConfig
from tinyinfer.hc import hc_split_sinkhorn, HCBlockMixin
from tinyinfer.nn import rms_norm
store = ShardStore(MD); cfg = TinyConfig.from_hf(MD)
torch.manual_seed(0)
layer = 0
hc_mult = cfg.hc_mult
x = torch.randn(1, 8, hc_mult, 5120, dtype=torch.bfloat16) * 0.1
fn = store.get("layers.%d.hc_attn_fn" % layer).cuda().float()
scale = store.get("layers.%d.hc_attn_scale" % layer).cuda().float()
base = store.get("layers.%d.hc_attn_base" % layer).cuda().float()

xf = x.flatten(2).float()
import math
eps = 1e-6
rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
mixes = torch.nn.functional.linear(xf, fn) * rsqrt

pre_o, post_o, comb_o = offk.hc_split_sinkhorn(mixes, scale, base, hc_mult, cfg.hc_sinkhorn_iters, eps)
pre_w, post_w, comb_w = hc_split_sinkhorn(mixes, scale, base, hc_mult, cfg.hc_sinkhorn_iters, eps)
for name, a, b in (("pre", pre_o, pre_w), ("post", post_o, post_w), ("comb", comb_o, comb_w)):
    print(name, "rel %.6f" % float((a.float()-b.float()).abs().max()/max(1e-9, a.float().abs().max())))

# fused path
class T(HCBlockMixin):
    hc_mult = hc_mult
    hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
    hc_eps = eps
t = T()
p2, po2, c2 = t.hc_mixes(x, fn, scale, base)
for name, a, b in (("pre", pre_o, p2), ("post", post_o, po2), ("comb", comb_o, c2)):
    print("fused", name, "rel %.6f" % float((a.float()-b.float()).abs().max()/max(1e-9, a.float().abs().max())))
