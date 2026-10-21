
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch
torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda")
MD = r"E:\Models\DeepSeek-V4.1-Flash-1017"
import model as official
from tinyinfer.weights import ShardStore
from tinyinfer.config import TinyConfig
from tinyinfer.moe import Expert as OurExpert
from tinyinfer.quant import e8m0_to_float

store = ShardStore(MD)
cfg = TinyConfig.from_hf(MD)
layer, eid = 0, 0
p = "layers.%d.ffn.experts.%d." % (layer, eid)

def load_fp4(name):
    w = store.get(p + name + ".weight")
    s = store.get(p + name + ".scale")
    return w.cuda(), s.cuda()

w1, s1 = load_fp4("w1")
w2, s2 = load_fp4("w2")
w3, s3 = load_fp4("w3")
print("fp4 weights", w1.dtype, tuple(w1.shape), "scale", s1.dtype, tuple(s1.shape))

ob = official.Expert(5120, 2304, dtype=torch.float4_e2m1fn_x2, swiglu_limit=10.0).cuda()
with torch.no_grad():
    for tgt, src, sc in ((ob.w1, w1, s1), (ob.w2, w2, s2), (ob.w3, w3, s3)):
        tgt.weight.data = src.view(torch.float4_e2m1fn_x2)
        tgt.weight.scale = sc
torch.manual_seed(0)
x = torch.randn(1, 4, 5120, dtype=torch.bfloat16) * 0.1
w = torch.randn(1, 4, 1, dtype=torch.bfloat16) * 0.5
ref = ob(x, w)
print("official fp4 expert out", tuple(ref.shape))

ours = OurExpert(None, None, None, 10.0)
ours.fp4 = (w1.view(torch.uint8), s1, w2.view(torch.uint8), s2, w3.view(torch.uint8), s3)
out = ours.forward_fp4(x, w)
print("ours fp4 expert out", tuple(out.shape))
print("expert rel %.6f" % float((ref.float()-out.float()).abs().max()/ref.float().abs().max()))
