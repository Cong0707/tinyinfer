
import sys, json, dataclasses
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\reference\official")
import torch

MD = r"E:\Models\DeepSeek-V4.1-Flash-1017"
torch.set_default_dtype(torch.bfloat16)
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
    if k2 in names:
        kw[k2] = tuple(v) if isinstance(v, list) else v
for k in ("n_mtp_layers","hc_mult","hc_sinkhorn_iters","hc_eps","dspark_block_size",
          "dspark_target_layer_ids","dspark_markov_rank","dspark_noise_token_id",
          "dspark_n_routed_experts","dspark_n_activated_experts"):
    if k in tc: kw[k] = tuple(tc[k]) if isinstance(tc[k], list) else tc[k]
args = official.ModelArgs(**kw)

from tinyinfer.weights import ShardStore
from tinyinfer.config import TinyConfig
from tinyinfer.model import Block as OurBlock
store = ShardStore(MD)
our_cfg = TinyConfig.from_hf(MD)

torch.manual_seed(0)
T = 8
h_in = torch.randn(1, T, 5120, device="cuda", dtype=torch.bfloat16) * 0.1

# --- official layer 0 attention (fp8 weights, dequantised in forward) ---
ob = official.Attention(1, args).cuda()
assign = skip = 0
for name, param in dict(ob.named_parameters()).items():
    try:
        w = store.get("layers.1.attn." + name)
    except Exception:
        skip += 1; continue
    if w.shape != param.shape:
        skip += 1; continue
    try:
        with torch.no_grad(): param.copy_(w)
        assign += 1
    except Exception:
        skip += 1
for n, b in ob.named_buffers():
    try:
        b.data = store.get("layers.1.attn." + n).to(b.device)
    except Exception:
        pass
# wo_a is stored fp8 in the checkpoint and is dequantised to bf16 by convert.py, so it
# must not be copied raw (that leaves the 448.0 e4m3 payload in a bf16 parameter)
from tinyinfer.quant import dequant_fp8_block
with torch.no_grad():
    ob.wo_a.weight.copy_(dequant_fp8_block(store.get("layers.1.attn.wo_a.weight"),
                                          store.get("layers.1.attn.wo_a.scale")))
print("wo_a dequantised, absmax", float(ob.wo_a.weight.float().abs().max()))
torch.set_default_device("cuda")
print("official attn params:", assign, "assigned,", skip, "skipped")
out_o = ob(h_in, 0)
print("official out", tuple(out_o.shape), out_o.dtype, "absmax", float(out_o.abs().max()))
our_cfg = TinyConfig.from_hf(MD)
tb = OurBlock(store, 1, our_cfg, quant=True)
tb.to("cuda")
# instrument: compare the window KV path before the attention call
from tinyinfer.nn import rms_norm, apply_rope
att = tb.attn
freqs = att.freqs_cis[0:8]
kv_w0 = rms_norm(att.wkv.forward(h_in), att.kv_norm, tb.cfg.rms_norm_eps)
apply_rope(kv_w0[..., -att.rope_dim:], freqs)
kvo, idxo = ob._window_kv(h_in, ob.freqs_cis[0:8], 0)
print("kv+rope rel %.5f" % ((kvo.float() - kv_w0.float()).abs().max()/kvo.float().abs().max()))
from tinyinfer.nn import window_idxs_general
wi_w = window_idxs_general(0, 8, 0, att.window, h_in.device).unsqueeze(0).expand(1, -1, -1)
print("win idx equal:", bool(torch.equal(wi_w.int().cpu(), idxo.int().cpu())))
# confirm which sparse_attn path the block actually takes
import tinyinfer.kernels as _K
_orig = _K.sparse_attn_dispatch
_calls = {"official": 0, "other": 0}
def _spy(q, kv, sink, idxs, scale, prefer_tilelang=True):
    try:
        from tinyinfer.official_kernel_ref import sparse_attn as _off
        _calls["official"] += 1
        return _off(q, kv, sink, idxs.to(q.device).int(), float(scale))
    except Exception:
        _calls["other"] += 1
        raise
import tinyinfer.attention as _A
_A.sparse_attn_dispatch = _spy
out_w = tb.attn.forward(h_in, 0)
print("dispatch calls", _calls)
print("ours out", tuple(out_w.shape), out_w.dtype, "absmax", float(out_w.abs().max()))
d = (out_o.float() - out_w.float()).abs()
rel = d.max() / max(1e-6, out_o.float().abs().max())
print("max abs diff %.6f  rel %.6f" % (float(d.max()), float(rel)))
