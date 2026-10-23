"""Qwen3.5 layer arithmetic, called by QwenLayer.

These are the reference formulas, not transformers internals: the attention block, the
GatedDeltaNet recurrence, and the glue between the embedding and the head. They take plain
weight dicts so they work on whatever device the engine put the layer.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with the reference rounding: fp32 statistics, bf16 out."""
    from . import native

    if x.dtype == torch.bfloat16 and native.available("qwen_norm", "qwen_rms_norm_bf"):
        out = torch.empty_like(x)
        d = x.shape[-1]
        rows = x.numel() // d
        # signature: (x_bf16, w_fp32, y_bf16, rows, d, eps, plus_one) -- the weight is
        # widened on the host, and plus_one=0 matches the reference (1 + w) form
        native.launch(native.kernel("qwen_norm", "qwen_rms_norm_bf"), (rows,), (32,),
                      [x.contiguous(), weight.float().contiguous(), out, rows, d,
                       float(eps), 0])
        return out
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * h.to(x.dtype))


def _rope(x: torch.Tensor, position: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Partial RoPE over the declared fraction of head_dim."""
    rp = cfg.get("rope_parameters") or {}
    theta = float(rp.get("rope_theta", 10000.0))
    frac = float(rp.get("partial_rotary_factor", 1.0))
    dim = x.shape[-1]
    rot = int(dim * frac) // 2 * 2
    if rot <= 0:
        return x
    pos = position.float().view(-1, 1)
    freqs = torch.arange(0, rot, 2, device=x.device, dtype=torch.float32)
    inv = torch.exp(-math.log(theta) * freqs / rot)
    ang = pos * inv
    cos = torch.cos(ang).to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(ang).to(x.dtype).unsqueeze(0).unsqueeze(0)
    x1 = x[..., :rot].reshape(*x.shape[:-1], -1, 2)
    a, b = x1[..., 0], x1[..., 1]
    ro = torch.stack((a * cos[..., 0] + b * sin[..., 0],
                      -a * sin[..., 0] + b * cos[..., 0]), dim=-1).reshape(*x.shape[:-1], rot)
    if rot == dim:
        return ro
    return torch.cat([ro, x[..., rot:]], dim=-1)


def _gqa_attention(t: dict, hidden: torch.Tensor, cache, layer_id: int, cfg: dict,
                   position) -> torch.Tensor:
    """Full attention: q/k/v projections, partial RoPE, KV window, output gate."""
    b, s, _h = hidden.shape
    nh = int(cfg["num_attention_heads"])
    nkv = int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))

    q = F.linear(hidden, t["self_attn.q_proj.weight"])
    k = F.linear(hidden, t["self_attn.k_proj.weight"])
    v = F.linear(hidden, t["self_attn.v_proj.weight"])
    # attn_output_gate: q_proj emits [q | gate]
    gate = None
    if q.shape[-1] == nh * hd * 2:
        q, gate = torch.chunk(q, 2, dim=-1)
    q = q.view(b, s, nh, hd).transpose(1, 2)
    k = k.view(b, s, nkv, hd).transpose(1, 2)
    v = v.view(b, s, nkv, hd).transpose(1, 2)

    if "self_attn.q_norm.weight" in t:
        q = _rms_norm(q.transpose(1, 2), t["self_attn.q_norm.weight"], eps).transpose(1, 2)
    if "self_attn.k_norm.weight" in t:
        k = _rms_norm(k.transpose(1, 2), t["self_attn.k_norm.weight"], eps).transpose(1, 2)

    q = _rope(q, position, cfg)
    k = _rope(k, position, cfg)

    keys, values = cache.update(k, v, layer_id, {"cache_position": position})
    rep = nh // nkv
    keys = keys.repeat_interleave(rep, dim=1)
    values = values.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, keys, values, is_causal=s > 1)
    out = out.transpose(1, 2).reshape(b, s, nh * hd)
    if gate is not None:
        out = out * torch.sigmoid(gate)
    return F.linear(out, t["self_attn.o_proj.weight"])


def attention_layer(t: dict, hidden, cache, layer_id, cfg, position):
    """A full-attention decoder layer (pre-norm, residual, MLP)."""
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    h = _rms_norm(hidden, t["input_layernorm.weight"], eps)
    h = _gqa_attention(t, h, cache, layer_id, cfg, position)
    hidden = hidden + h
    h = _rms_norm(hidden, t["post_attention_layernorm.weight"], eps)
    hidden = hidden + _mlp(t, h)
    return hidden


def _mlp(t: dict, h: torch.Tensor) -> torch.Tensor:
    """SwiGLU over the fused gate/up pair when present, else the two projections."""
    from . import qwen_elt, native

    g = F.linear(h, t["mlp.gate_proj.weight"])
    u = F.linear(h, t["mlp.up_proj.weight"])
    if (h.shape[1] == 1 and h.dtype == torch.bfloat16
            and native.available("qwen_fused", "qwen_swiglu") and native.available(
                "qwen_fused", "qwen_gemv_group")):
        inter = g.shape[-1]
        return F.linear(_swiglu_kernel(g, u, inter), t["mlp.down_proj.weight"])
    return F.linear(F.silu(g) * u, t["mlp.down_proj.weight"])


def _swiglu_kernel(g: torch.Tensor, u: torch.Tensor, inter: int) -> torch.Tensor:
    from . import native

    gu = torch.cat([g.reshape(-1), u.reshape(-1)], 0).contiguous()
    out = torch.empty(inter, dtype=torch.bfloat16, device=g.device)
    native.launch(native.kernel("qwen_fused", "qwen_swiglu"),
                  ((inter + 255) // 256,), (256,), [gu, out, inter, 0.0])
    return out.view(g.shape)


def linear_layer(t: dict, hidden, cache, layer_id, cfg):
    """A GatedDeltaNet decoder layer: delta-rule recurrence plus gated norm."""
    from . import gdn

    eps = float(cfg.get("rms_norm_eps", 1e-6))
    b, s, _h = hidden.shape
    h = _rms_norm(hidden, t["input_layernorm.weight"], eps)

    nk = int(cfg["linear_num_key_heads"])
    nv = int(cfg["linear_num_value_heads"])
    kd = int(cfg["linear_key_head_dim"])
    vd = int(cfg["linear_value_head_dim"])
    conv_k = int(t["linear_attn.conv1d.weight"].shape[-1])

    x = h[0, -1]                                     # one token at a time here
    qkv = F.linear(x, t["linear_attn.in_proj_qkv.weight"])
    z = F.linear(x, t["linear_attn.in_proj_z.weight"])
    beta = torch.sigmoid(F.linear(x, t["linear_attn.in_proj_b.weight"]))
    a = F.linear(x, t["linear_attn.in_proj_a.weight"])

    conv_state = cache.layer(layer_id).conv_states[0]
    qkv = gdn.conv_step(qkv.view(1, -1), t["linear_attn.conv1d.weight"].squeeze(1),
                        t.get("linear_attn.conv1d.bias"), conv_state)

    kdim, vdim = nk * kd, nv * vd
    q_raw, k_raw, v = torch.split(qkv.view(1, -1), [kdim, kdim, vdim], -1)
    q = q_raw.view(nk, kd).unsqueeze(0)
    k = k_raw.view(nk, kd).unsqueeze(0)
    v = v.view(nv, vd).unsqueeze(0)
    if nv // nk > 1:
        r = nv // nk
        q = q.repeat_interleave(r, dim=1)
        k = k.repeat_interleave(r, dim=1)
    g = -t["linear_attn.A_log"].float().exp() * F.softplus(
        a.float() + t["linear_attn.dt_bias"])
    state = cache.layer(layer_id).recurrent_states[0] if cache.layer(layer_id).written else None
    core, last = gdn.step(q, k, v, g.unsqueeze(0), beta.unsqueeze(0).unsqueeze(-1)[..., 0],
                          state)
    cache.set_recurrent(layer_id, last)
    core = core.reshape(-1, vd)
    core = gdn.norm_gated(core, z.view(-1, vd), t["linear_attn.norm.weight"], eps)
    hidden = hidden + F.linear(core.view(1, 1, -1), t["linear_attn.out_proj.weight"])

    h2 = _rms_norm(hidden, t["post_attention_layernorm.weight"], eps)
    return hidden + _mlp(t, h2)


def model_forward(model, engine, ids, start_pos: int):
    """Embed, run every layer through the engine, then the head."""
    cfg = model.cfg
    h = engine.embed(ids)
    position = torch.arange(start_pos, start_pos + ids.shape[1], device=ids.device)
    for i in range(model.n_layers):
        layer = engine.block(i)
        h = layer.forward(h, cache=engine.kv(), position=position)
    return engine.head(h)