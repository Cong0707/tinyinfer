
import sys, math, json, types, importlib.util
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
torch.set_default_dtype(torch.bfloat16)
MD = r"E:\Models\DeepSeek-V4.1-Flash-1017"
tc = json.load(open(MD + r"\config.json"))["text_config"]
rs = dict(tc["rope_scaling"]); rs["type"] = rs.pop("rope_type")

# load DeepInfer's rotary_embedding.py without importing the package (it assumes Linux)
stub = types.ModuleType("tile_kernels.transform")
stub.apply_rotary = lambda *a, **k: None
sys.modules["tile_kernels"] = types.ModuleType("tile_kernels")
sys.modules["tile_kernels.transform"] = stub
path = r"E:\Develop\Deepseek\repositories\DeepInfer\deepinfer\model_executor\layers\rotary_embedding.py"
spec = importlib.util.spec_from_file_location("di_rotary", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rope = mod.get_rope(head_size=tc["head_dim"], rotary_dim=tc["qk_rope_head_dim"],
                    max_position=min(tc["max_position_embeddings"], 1 << 20),
                    base=tc["compress_rope_theta"], is_neox_style=True, rope_scaling=rs)
cache = rope.cos_sin_cache.cpu()
print("DeepInfer cache", tuple(cache.shape), "dtype", cache.dtype, "mscale", rope.mscale)

from tinyinfer.nn import precompute_freqs_cis
ours = precompute_freqs_cis(tc["qk_rope_head_dim"], cache.shape[0], 65536,
                            tc["compress_rope_theta"], rs["factor"], rs["beta_fast"], rs["beta_slow"])
half = cache.shape[1] // 2
ref_cos, ref_sin = cache[:, :half], cache[:, half:]
print("max |cos diff| %.6e  max |sin diff| %.6e" % (
    float((ref_cos - ours[:cache.shape[0]].real).abs().max()),
    float((ref_sin - ours[:cache.shape[0]].imag).abs().max())))
