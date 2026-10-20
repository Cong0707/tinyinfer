"""Numerical tests for tinyinfer.nn against the official reference formulas."""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyinfer.nn import apply_rope, precompute_freqs_cis, rms_norm, sparse_attn_ref, window_idxs_prefill

# ---- official implementations copied verbatim from reference/official/model.py ----
def official_precompute(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def official_apply(x, freqs_cis, inverse=False):
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


def test_rope_matches_official():
    torch.manual_seed(0)
    for dim, seqlen, orig, base in [(64, 128, 0, 10000.0), (64, 256, 65536, 160000.0)]:
        f1 = precompute_freqs_cis(dim, seqlen, orig, base, 16, 32, 1)
        f2 = official_precompute(dim, seqlen, orig, base, 16, 32, 1)
        assert torch.allclose(f1, f2, atol=1e-5), (dim, orig)
        x = torch.randn(2, seqlen, 8, dim, dtype=torch.bfloat16)
        a, b = x.clone(), x.clone()
        apply_rope(a[..., -dim:], f1)
        official_apply(b[..., -dim:], f2)
        assert torch.allclose(a.float(), b.float(), atol=1e-2)
        a, b = x.clone(), x.clone()
        apply_rope(a[..., -dim:], f1, inverse=True)
        official_apply(b[..., -dim:], f2, inverse=True)
        assert torch.allclose(a.float(), b.float(), atol=1e-2)
    print("rope matches official")


def test_window_idxs():
    idxs = window_idxs_prefill(6, 4)
    assert idxs.shape == (6, 4)
    # query 0 sees only slot 0; query 5 sees slots 2..5
    assert idxs[0].tolist() == [0, -1, -1, -1]
    assert idxs[5].tolist() == [2, 3, 4, 5]
    print("window idxs ok")


def test_sparse_attn_sink():
    torch.manual_seed(1)
    b, s, h, d, win = 1, 4, 2, 8, 4
    q = torch.randn(b, s, h, d)
    kv = torch.randn(b, s, d)
    sink = torch.randn(h)
    idxs = window_idxs_prefill(s, win).unsqueeze(0)
    got = sparse_attn_ref(q, kv, sink, idxs, d ** -0.5)

    # naive reference
    for t in range(s):
        for hh in range(h):
            pos = [i for i in idxs[0, t].tolist() if i >= 0]
            kk = kv[0, pos]
            scores = (kk.float() @ q[0, t, hh].float()) / math.sqrt(d)
            m = max(scores.max().item(), sink[hh].item())
            w = torch.exp(scores - m)
            den = w.sum() + math.exp(sink[hh].item() - m)
            ref = (w / den) @ kk.float()
            assert torch.allclose(got[0, t, hh], ref, atol=1e-5), (t, hh)
    print("sparse_attn matches naive reference")


def test_rms_norm():
    x = torch.randn(2, 5, 16)
    w = torch.randn(16)
    y = rms_norm(x, w, 1e-6)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(y, ref, atol=1e-5)
    print("rms_norm ok")


if __name__ == "__main__":
    test_rope_matches_official()
    test_window_idxs()
    test_sparse_attn_sink()
    test_rms_norm()
    print("ALL OK")
