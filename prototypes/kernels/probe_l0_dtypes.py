import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.weights import ShardStore
store = ShardStore(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
for n in ["layers.0.attn.wq_a.weight","layers.0.attn.wq_a.scale","layers.0.attn.wq_b.weight","layers.0.attn.wq_b.scale",
          "layers.0.attn.wkv.weight","layers.0.attn.wo_a.weight","layers.0.attn.wo_b.weight",
          "layers.0.ffn.experts.0.w1.weight","layers.0.ffn.experts.0.w1.scale",
          "layers.0.ffn.experts.0.w2.weight","layers.0.ffn.shared_experts.w1.weight",
          "embed.weight"]:
    t = store.get(n)
    print(n, t.shape, t.dtype)
