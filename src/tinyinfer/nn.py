"""Minimal numeric building blocks used by the reference forward."""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F


_RMS_NATIVE = {"ok": None}


def _rms_native_ok() -> bool:
    if _RMS_NATIVE["ok"] is None:
        try:
            from . import native
            native.kernel("rms_norm", "rms_norm_kernel")
            _RMS_NATIVE["ok"] = True
        except Exception:
            _RMS_NATIVE["ok"] = False
    return _RMS_NATIVE["ok"]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF-style RMSNorm: normalize in fp32, scale by the stored weight.

    One kernel for the whole chain. The torch form is float/square/mean/rsqrt/mul/mul/to
    -- seven launches, twice per layer, 560 per decode step for a 1-row activation. The
    kernel keeps the same fp32 reduction and the same final bf16 cast.
    """
    orig_dtype = x.dtype
    if (x.is_cuda and x.dtype == torch.bfloat16 and _rms_native_ok()
            and weight.numel() == x.shape[-1]):
        try:
            from . import native
            shape = x.shape
            d = shape[-1]
            rows = 1
            for v in shape[:-1]:
                rows *= int(v)
            xc = x.reshape(rows, d).contiguous()
            wf = weight.float().contiguous()
            y = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
            native.launch(native.kernel("rms_norm", "rms_norm_kernel"), (rows,), (32,),
                          [xc, wf, y, rows, d, float(eps)], 0)
            return y.view(*shape)
        except Exception:
            pass
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(orig_dtype)


def window_idxs_prefill(seqlen: int, window: int, device=None) -> torch.Tensor:
    """Causal sliding-window slot indexes for a fresh prefill (see official get_window_topk_idxs)."""
    end = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (end - window + 1).clamp(0) + torch.arange(min(seqlen, window), device=device)
    idxs = torch.where(idxs > end, torch.full_like(idxs, -1), idxs)
    return idxs.int()


def window_idxs_general(start_pos: int, seqlen: int, old_len: int, window: int, device=None) -> torch.Tensor:
    """Window slots for a chunk inside a linear KV history.

    kv_att = [old_history (old_len, oldest first), new_chunk (seqlen)]; a query at global
    position p attends to [max(0, p-window+1) .. p]. Returns [seqlen, window] int32 (-1 = empty).
    """
    qpos = torch.arange(start_pos, start_pos + seqlen, device=device)
    back = torch.arange(window - 1, -1, -1, device=device)
    gpos = qpos[:, None] - back[None, :]  # [s, window] window start..p
    base = start_pos - old_len
    idx = gpos - base
    valid = (gpos >= 0) & (idx >= 0) & (idx < old_len + seqlen)
    return torch.where(valid, idx, torch.full_like(idx, -1)).int()


def sparse_attn_ref(
    q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, idxs: torch.Tensor, scale: float
) -> torch.Tensor:
    """Naive gather-sparse MQA attention with a learned attn-sink logit per head.

    q: [b, s, h, d]; kv: [b, n, d]; sink: [h]; idxs: [b, s, t] (-1 = empty slot).
    Equivalent to the tilelang sparse_attn kernel: scores = q.k / sqrt(d), softmax over
    gathered slots PLUS one virtual sink slot whose logit is attn_sink[h].
    """
    b, s, h, d = q.shape
    t = idxs.shape[-1]
    valid = idxs >= 0  # [b, s, t]
    safe = idxs.clamp(min=0)  # [b, s, t]
    # gather kv for every query: [b, s, t, d]
    flat = safe.reshape(b, s * t, 1).expand(b, s * t, d)
    k = kv.gather(1, flat).reshape(b, s, t, d)
    scores = torch.einsum("bshd,bstd->bsht", q.float(), k.float()) * scale  # [b, s, h, t]
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    max_scores = scores.max(dim=-1).values  # [b,s,h]
    # fold the sink in as an extra logit; a finite bound keeps fully-empty rows finite (all-zero out)
    max_final = torch.max(max_scores, sink.view(1, 1, h))
    p = torch.exp(scores - max_final.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(sink.view(1, 1, h) - max_final)
    p = p / denom.unsqueeze(-1)
    o = torch.einsum("bsht,bstd->bshd", p, k.float())
    return o.to(q.dtype)


def softmax_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Greedy multinomial sampling (same math as the official sample())."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def sample_logits(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0,
                  top_k: int = 0, repetition_penalty: float = 1.0,
                  recent_tokens: list[int] | None = None) -> int:
    """Greedy (temperature 0) or temperature/top-p/top-k sampling.

    repetition_penalty / recent_tokens follow the vLLM sampler convention: a token
    already present in the recent window gets its (positive/negative) logit divided/
    multiplied by the penalty. The default 1.0 disables it.
    """
    if repetition_penalty != 1.0 and recent_tokens:
        idx = torch.tensor(sorted(set(recent_tokens)), dtype=torch.long, device=logits.device)
        vals = logits.index_select(-1, idx).float()
        vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
        logits = logits.clone()
        logits.index_copy_(-1, idx, vals.to(logits.dtype))
    if temperature <= 0:
        return int(logits.argmax(dim=-1))
    logits = logits.float() / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.numel())
        v, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < v[-1], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        keep = cum - sorted_p < top_p
        sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        probs = torch.zeros_like(probs).scatter_(0, order, sorted_p)
        probs = probs / probs.sum().clamp_min(1e-12)
    return int(torch.multinomial(probs, 1))


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent element pairs as complex numbers; accepts [b,s,d] or [b,s,h,d].

    Used both for queries/keys and, with inverse=True, to un-rotate the attention output.
    """
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        f = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        f = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    out = torch.view_as_real(xc * f).flatten(-2)
    y.copy_(out)
    return y


def precompute_freqs_cis(
    dim: int, seqlen: int, original_seq_len: int, base: float, factor: float, beta_fast: float, beta_slow: float
) -> torch.Tensor:
    """Rotary frequencies (complex exponentials), one row per position; YaRN when original_seq_len > 0."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    mscale = 1.0
    if original_seq_len > 0:
        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
        # DeepSeek's own serving stack (DeepInfer rotary_embedding.py) applies
        # yarn_get_mscale(factor) = 0.1 * mscale * ln(factor) + 1 to the cos/sin cache,
        # with mscale defaulting to True and mscale_all_dim to 0. The model author is the
        # authority here, so the magnitude scaling is part of the reference semantics.
        # DeepInfer (the model author's serving stack) applies yarn_get_mscale:
        # 0.1*ln(factor)+1 = 1.277 at factor 16, and the author's stack is the authority
        # this engine aligns to. The plain HF reference table omits it, which is why
        # docs/parity.md records the DeepInfer comparison as the reference one.
        _ms = float(os.environ.get("TI_YARN_MSCALE", "1.0"))
        if factor > 1 and _ms:
            mscale = 0.1 * _ms * math.log(factor) + 1.0
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs) * mscale
