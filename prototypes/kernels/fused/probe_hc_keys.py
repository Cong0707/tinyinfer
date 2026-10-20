import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.weights import ShardStore
store = ShardStore(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
for n in ["layers.0.hc_attn_fn","layers.0.hc_attn_scale","layers.0.hc_attn_base",
          "layers.0.hc_ffn_fn","layers.0.hc_ffn_scale","layers.0.hc_ffn_base",
          "layers.0.attn.attn_sink"]:
    print(n, store.has(n), store.get(n).shape, store.get(n).dtype)
