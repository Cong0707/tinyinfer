
import sys, json, dataclasses
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch, math
MD = r"E:\Models\DeepSeek-V4.1-Flash-1017"
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
tc = json.load(open(MD + r"\config.json"))["text_config"]
import model as official
names = {f.name for f in dataclasses.fields(official.ModelArgs)}
MAP = {"hidden_size": "dim", "moe_intermediate_size": "moe_inter_dim", "num_hidden_layers": "n_layers",
       "num_attention_heads": "n_heads", "qk_rope_head_dim": "rope_head_dim", "rms_norm_eps": "norm_eps",
       "num_experts_per_tok": "n_activated_experts", "scoring_func": "score_func",
       "routed_scaling_factor": "route_scale", "sliding_window": "window_size",
       "kv_source_layer_ids": "kv_source_layers", "index_source_layer_ids": "index_source_layers",
       "candidate_source_layer_id": "candidate_source_layer", "max_position_embeddings": "max_seq_len"}
kw = {}
for k, v in tc.items():
    k2 = MAP.get(k, k)
    if k2 in names: kw[k2] = tuple(v) if isinstance(v, list) else v
for k in ("n_mtp_layers","hc_mult","hc_sinkhorn_iters","hc_eps","dspark_block_size",
          "dspark_target_layer_ids","dspark_markov_rank","dspark_noise_token_id",
          "dspark_n_routed_experts","dspark_n_activated_experts"):
    if k in tc: kw[k] = tuple(tc[k]) if isinstance(tc[k], list) else tc[k]
# DeepSeek's definition lives in rope_scaling; the reference's flat ModelArgs needs it
# spelled out or it silently falls back to original_seq_len=0 / rope_factor=40
_rs = tc.get("rope_scaling") or {}
kw["original_seq_len"] = int(_rs.get("original_max_position_embeddings", 0))
kw["rope_factor"] = int(_rs.get("factor", 40))
kw["beta_fast"] = int(_rs.get("beta_fast", 32))
kw["beta_slow"] = int(_rs.get("beta_slow", 1))
args = official.ModelArgs(**kw)

from tinyinfer.weights import ShardStore
from tinyinfer.config import TinyConfig
from tinyinfer.model import Block as OurBlock
from tinyinfer.quant import dequant_fp8_block
store = ShardStore(MD)
cfg = TinyConfig.from_hf(MD)
torch.manual_seed(0)
h = torch.randn(1, 8, 5120, device="cuda", dtype=torch.bfloat16) * 0.1

for layer in (2, 8):
    ob = official.Attention(layer, args).cuda()
    for name, param in dict(ob.named_parameters()).items():
        try:
            w = store.get("layers.%d.attn.%s" % (layer, name))
            if w.shape == param.shape:
                with torch.no_grad(): param.copy_(w)
        except Exception:
            pass
    with torch.no_grad():
        try:
            ob.wo_a.weight.copy_(dequant_fp8_block(store.get("layers.%d.attn.wo_a.weight" % layer),
                                                  store.get("layers.%d.attn.wo_a.scale" % layer)))
        except Exception as exc:
            print("wo_a", exc)
    for n, b in ob.named_buffers():
        try: b.data = store.get("layers.%d.attn.%s" % (layer, n)).to(b.device)
        except Exception: pass
    # DeepInfer (DeepSeek's serving stack) scales the YaRN cos/sin cache by
    # yarn_get_mscale(factor); the reference model.py does not. Apply it to the
    # reference so the comparison is against the model author's semantics.
    _ms = 0.1 * math.log(_rs.get("factor", 16)) + 1.0 if _rs.get("factor", 16) > 1 else 1.0
    if _ms != 1.0:
        ob.freqs_cis = (ob.freqs_cis * _ms)
    tb = OurBlock(store, layer, cfg, quant=True); tb.to("cuda")
    ref = ob(h, 0)
    ours = tb.attn.forward(h, 0)
    print("layer", layer, "attn rel %.6f" % float((ref.float()-ours.float()).abs().max()/ref.float().abs().max()))
    del ob, tb
    torch.cuda.empty_cache()
