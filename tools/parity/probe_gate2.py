
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch
torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda")
MD = r"E:\Models\DeepSeek-V4.1-Flash-1017"
import model as official
from tinyinfer.weights import ShardStore
from tinyinfer.config import TinyConfig
from tinyinfer.moe import Gate as OurGate
store = ShardStore(MD); cfg = TinyConfig.from_hf(MD)
layer = 0
torch.manual_seed(0)
x = torch.randn(8, 5120, dtype=torch.bfloat16) * 0.1

class Args:
    dim = cfg.dim
    n_routed_experts = cfg.n_routed_experts
    n_activated_experts = cfg.n_activated_experts
    score_func = cfg.scoring_func
    gate_temp = 1.0
    norm_topk_prob = cfg.norm_topk_prob
    route_scale = cfg.route_scale
    vision_enabled = False
    def get_moe_config(self, lid):
        return (self.n_routed_experts, self.n_activated_experts)

og = official.Gate(layer, Args()).cuda()
with torch.no_grad():
    og.weight.copy_(store.get("layers.%d.ffn.gate.weight" % layer))
    og.bias.copy_(store.get("layers.%d.ffn.gate.bias" % layer))
w_ref, i_ref = og(x)

our = OurGate(store, layer, cfg)
our.weight = our.weight.cuda()
our.bias = our.bias.cuda()
w_our, i_our = our.forward(x)
print("gate weights rel %.6f" % float((w_ref.float()-w_our.float()).abs().max()/w_ref.float().abs().max()))
print("gate idx equal:", bool(torch.equal(i_ref.int().cpu(), i_our.int().cpu())),
      "mismatch", int((i_ref.int().cpu() != i_our.int().cpu()).sum()))
