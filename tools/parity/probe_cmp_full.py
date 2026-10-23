
import sys, json, dataclasses
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch
torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda")
MD = r"E:\Models\DeepSeek-V4.1-Flash-1017"
tc = json.load(open(MD + r"\config.json"))["text_config"]
import model as official
names = {f.name for f in dataclasses.fields(official.ModelArgs)}
MAP = {"hidden_size":"dim","moe_intermediate_size":"moe_inter_dim","num_hidden_layers":"n_layers",
       "num_attention_heads":"n_heads","qk_rope_head_dim":"rope_head_dim","rms_norm_eps":"norm_eps",
       "num_experts_per_tok":"n_activated_experts","scoring_func":"score_func",
       "routed_scaling_factor":"route_scale","sliding_window":"window_size",
       "kv_source_layer_ids":"kv_source_layers","index_source_layer_ids":"index_source_layers",
       "candidate_source_layer_id":"candidate_source_layer","max_position_embeddings":"max_seq_len"}
kw = {}
for k, v in tc.items():
    k2 = MAP.get(k, k)
    if k2 in names: kw[k2] = tuple(v) if isinstance(v, list) else v
for k in ("n_mtp_layers","hc_mult","hc_sinkhorn_iters","hc_eps","dspark_block_size",
          "dspark_target_layer_ids","dspark_markov_rank","dspark_noise_token_id",
          "dspark_n_routed_experts","dspark_n_activated_experts"):
    if k in tc: kw[k] = tuple(tc[k]) if isinstance(tc[k], list) else tc[k]
args = official.ModelArgs(**kw)

from tinyinfer.weights import ShardStore
from tinyinfer.config import TinyConfig
from tinyinfer.model import Block as OurBlock
from tinyinfer.quant import dequant_fp8_block
from tinyinfer.nn import rms_norm, apply_rope
from tinyinfer.official_kernel_ref import act_quant, fp4_act_quant, sparse_attn
store = ShardStore(MD); cfg = TinyConfig.from_hf(MD)
torch.manual_seed(0)
h = torch.randn(1, 8, 5120, dtype=torch.bfloat16) * 0.1
layer = 2
ob = official.Attention(layer, args).cuda()
for name, param in dict(ob.named_parameters()).items():
    try:
        w = store.get("layers.%d.attn.%s" % (layer, name))
        if w.shape == param.shape:
            with torch.no_grad(): param.copy_(w)
    except Exception: pass
with torch.no_grad():
    ob.wo_a.weight.copy_(dequant_fp8_block(store.get("layers.%d.attn.wo_a.weight" % layer),
                                          store.get("layers.%d.attn.wo_a.scale" % layer)))
for n, b in ob.named_buffers():
    try: b.data = store.get("layers.%d.attn.%s" % (layer, n)).to(b.device)
    except Exception: pass

# official full forward internals
x = h
qr_o = ob.q_norm(ob.wq_a(x))
q_o = ob.wq_b(qr_o).unflatten(-1, (ob.n_local_heads, ob.head_dim))
official.apply_rotary_emb(q_o[..., -ob.rope_head_dim:], ob.freqs_cis[0:8])
kvw_o, win_o = ob._window_kv(x, ob.freqs_cis[0:8], 0)
cpk_o, cpi_o = ob._compress_kv(x, qr_o, 0, kvw_o.size(1))
kv_o = torch.cat([kvw_o, cpk_o], dim=1)
idxs_o = torch.cat([win_o, cpi_o], dim=-1)
o_o = sparse_attn(q_o, kv_o, ob.attn_sink, idxs_o, ob.softmax_scale)
official.apply_rotary_emb(o_o[..., -ob.rope_head_dim:], ob.freqs_cis[0:8], True)

# ours: replicate the compressed block path
tb = OurBlock(store, layer, cfg, quant=True); tb.to("cuda"); att = tb.attn
qr_w = rms_norm(att.wq_a.forward(x), att.q_norm, cfg.rms_norm_eps)
q_w = att.wq_b.forward(qr_w).view(1, 8, att.n_heads, att.head_dim)
apply_rope(q_w[..., -att.rope_dim:], att.freqs_cis[0:8])
kvw_w = rms_norm(att.wkv.forward(x), att.kv_norm, cfg.rms_norm_eps)
apply_rope(kvw_w[..., -att.rope_dim:], att.freqs_cis[0:8])
kvw_w = act_quant(kvw_w, 32, "ue8m0", torch.float8_e8m0fnu, True)
lat_w = att.compressor.forward(x, 0)
fr = att.freqs_cis[: 8 - 8 % att.ratio : att.ratio]
apply_rope(lat_w[..., -att.rope_dim:], fr)
lat_w2 = lat_w.clone()
fp4_act_quant(lat_w2, 16, True, scale_dtype=torch.float8_e4m3fn)

print("qr   rel %.6f" % float((qr_o.float()-qr_w.float()).abs().max()/qr_o.float().abs().max()))
print("q    rel %.6f" % float((q_o.float()-q_w.float()).abs().max()/q_o.float().abs().max()))
print("kvwin rel %.6f" % float((kvw_o.float()-kvw_w.float()).abs().max()/kvw_o.float().abs().max()))
print("lpre  rel %.6f" % float((cpk_o.float()-lat_w.float()).abs().max()/cpk_o.float().abs().max()))
print("lpost rel %.6f" % float((cpk_o.float()-lat_w2.float()).abs().max()/cpk_o.float().abs().max()))

fo = ob.freqs_cis
fw = att.freqs_cis
print("freqs shape", tuple(fo.shape), tuple(fw.shape))
n = min(fo.shape[0], fw.shape[0])
r = (fo[:n].float() - fw[:n].float()).abs()
print("freqs max diff %.6e  rel %.6e" % (float(r.max()), float(r.max()/max(1e-9, fo[:n].float().abs().max()))))

