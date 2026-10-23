"""Point transformers' Qwen3.5 GatedDeltaNet at the hand-written kernels.

transformers resolves torch_recurrent_gated_delta_rule / causal_conv1d_update at call
time from the module namespace, so replacing them there is enough to take the fast path.
Only the seq_len == 1 (decode) case is replaced; prefill keeps the reference, whose
chunked form does real work that this kernel does not cover.
"""
from __future__ import annotations

import torch


def _gdn_targets():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    return M


def install(*, conv=True, rule=True) -> bool:
    from . import gdn

    if not gdn.available():
        return False
    M = _gdn_targets()

    if rule and not getattr(M, "_tinyinfer_rule", False):
        orig = M.torch_recurrent_gated_delta_rule

        def torch_recurrent_gated_delta_rule(query, key, value, g=None, beta=None,
                                             initial_state=None, output_final_state=False,
                                             use_qk_l2norm_in_kernel=False, **kw):
            if (query.shape[1] == 1 and initial_state is not None and g is not None
                    and beta is not None):
                B, _, Hk, Dk = query.shape
                Hv = value.shape[2]
                Dv = value.shape[-1]
                q = query[:, 0].transpose(0, 1).contiguous()          # [B?, Hk, Dk]
                k = key[:, 0].transpose(0, 1).contiguous()
                v = value[:, 0].transpose(0, 1).contiguous()
                if Hv != Hk:
                    rep = Hv // Hk
                    q = q.repeat_interleave(rep, dim=1)
                    k = k.repeat_interleave(rep, dim=1)
                gg = g[:, 0].transpose(0, 1).contiguous().float()
                bb = beta[:, 0].transpose(0, 1).contiguous().float()
                st = initial_state.float().contiguous()
                out = gdn.step(q, k, v, gg, bb, st, l2norm=bool(use_qk_l2norm_in_kernel))
                out = out.transpose(0, 1).unsqueeze(1)                # [B, 1, Hv, Dv]
                return out, (st if output_final_state else None)
            return orig(query, key, value, g=g, beta=beta, initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)

        M.torch_recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        M._tinyinfer_rule = True

    if conv and not getattr(M, "_tinyinfer_conv", False):
        orig_conv = M.causal_conv1d_update

        def causal_conv1d_update(hidden_states, conv_state, weight, bias=None,
                                 activation=None, **kw):
            # Decode shape only: one new token, depthwise, bias present or not.
            if (hidden_states.dim() == 3 and hidden_states.shape[-1] == 1
                    and weight.dim() == 2
                    and conv_state.shape[0] == hidden_states.shape[0]
                    and conv_state.shape[1] == hidden_states.shape[1]
                    and conv_state.dtype in (torch.float32, torch.bfloat16)
                    and conv_state.is_cuda):
                x = hidden_states[:, :, 0]
                out = gdn.conv_step(x, weight, bias, conv_state,
                                    silu=(activation == "silu"))
                return out.unsqueeze(-1)
            return orig_conv(hidden_states, conv_state, weight, bias,
                             activation=activation, **kw)

        M.causal_conv1d_update = causal_conv1d_update
        M._tinyinfer_conv = True
    return True
