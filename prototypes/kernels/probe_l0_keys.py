import safetensors, json, os
p = r"E:\Develop\Deepseek\tinyinfer\cache\weights\model-00003-of-00048.safetensors"
from safetensors import safe_open
with safe_open(p, framework="pt", device="cpu") as f:
    keys = list(f.keys())
print("num keys", len(keys))
for k in keys[:40]: print(k)
print("...")
attn = [k for k in keys if ".attn." in k and "experts" not in k]
print("attn-like", len(attn)); print("\n".join(attn[:40]))
ex = [k for k in keys if "experts.0." in k]
print("expert0", ex[:10])
