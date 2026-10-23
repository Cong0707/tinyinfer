"""Fuse Qwen3.5 full-attention's q/k/v projections into one GEMV.

q_proj, k_proj and v_proj read the same row. Four of the 24 layers are full-attention
(they carry 48MB + 6MB + 6MB of weights); joining saves two kernels per call, and the
benchmark on the norm kernels showed each kernel in the captured step costs ~2.5us of
gap that the graph does not remove.
"""
from __future__ import annotations

import torch


def install(model) -> int:
    from . import gemv_bf16
    if not gemv_bf16.available():
        return 0
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    n = 0
    for lay in getattr(getattr(model, "model", model), "layers", []):
        att = getattr(lay, "self_attn", None)
        if att is None or getattr(att, "_ti_qkv", None) is not None:
            continue
        parts = [getattr(att, k, None) for k in ("q_proj", "k_proj", "v_proj")]
        if any(p is None or p.bias is not None for p in parts):
            continue
        att._ti_qkv = torch.cat([p.weight.detach() for p in parts], 0).contiguous()
        att._ti_sizes = [p.weight.shape[0] for p in parts]
        n += 1
    if n == 0:
        return 0
    if getattr(M.Qwen3_5Attention, "_ti_patched", False):
        return n
    orig = M.Qwen3_5Attention.forward

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kw):
        w = getattr(self, "_ti_qkv", None)
        single = (w is not None and hidden_states.dim() == 3
                  and hidden_states.shape[1] == 1
                  and hidden_states.dtype == torch.bfloat16)
        if not single:
            return orig(self, hidden_states, position_embeddings, attention_mask,
                        past_key_values=past_key_values, **kw)
        total = w.shape[0]
        buf = getattr(self, "_ti_out", None)
        if buf is None or buf.numel() != total:
            buf = self._ti_out = torch.empty(total, dtype=torch.float32,
                                             device=hidden_states.device)
        from . import gemv_bf16 as G
        G.gemv(hidden_states[0, 0], w, buf)
        dt = hidden_states.dtype
        q_sz, k_sz, v_sz = self._ti_sizes
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_raw = buf[:q_sz].to(dt).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_raw, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            buf[q_sz:q_sz + k_sz].to(dt).view(hidden_shape)).transpose(1, 2)
        value_states = buf[q_sz + k_sz:].to(dt).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = M.apply_rotary_pos_emb(query_states, key_states,
                                                          cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx)
        from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward
        _kw = dict(dropout=0.0, scaling=self.scaling)
        _sw = getattr(self, "sliding_window", None)
        if _sw is not None:
            _kw["sliding_window"] = _sw
        attn_out, attn_w = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, **_kw)
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out), attn_w

    M.Qwen3_5Attention.forward = forward
    M.Qwen3_5Attention._ti_patched = True
    return n
