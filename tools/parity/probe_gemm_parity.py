
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch
MD = r"E:\Models\DeepSeek-V4.1-Flash-1017"
torch.set_default_dtype(torch.bfloat16)
from tinyinfer.weights import ShardStore
from tinyinfer.gemm import fp8_linear
import model as official

store = ShardStore(MD)
torch.manual_seed(0)
x = torch.randn(1, 8, 1280, device="cuda", dtype=torch.bfloat16) * 0.1
w = store.get("layers.1.attn.wq_b.weight").cuda()
s = store.get("layers.1.attn.wq_b.scale").cuda()

lin = official.Linear(1280, 32768).cuda()
with torch.no_grad():
    lin.weight.data = w
lin.weight.scale = s

ours = fp8_linear(x, w, s, block=32)
ref = lin(x)
print("ours", tuple(ours.shape), "ref", tuple(ref.shape))
print("rel %.6f" % float((ours.float() - ref.float()).abs().max() / ref.float().abs().max()))
