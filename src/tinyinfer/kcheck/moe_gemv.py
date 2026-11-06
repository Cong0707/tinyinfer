"""The fused MoE decode chain: gate_up -> act -> down -> acc.

These four kernels are what a decode step actually runs 40 times, and until now nothing
checked them against a reference. They take packed fp4 weights and e8m0 scales directly,
so a mistake here does not crash -- it produces a plausible activation that is wrong.
"""
from __future__ import annotations

import torch

from . import Case, check_close


def _fp4(vals: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] float codes 0..15 into the [N, K/2] uint8 layout the kernels read."""
    lo = vals[:, 0::2].to(torch.uint8)
    hi = vals[:, 1::2].to(torch.uint8)
    return (lo | (hi << 4)).contiguous()


def _lut() -> torch.Tensor:
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def register(report) -> None:
    try:
        from ..deepseek import moe_gemv
        from ..kernels.quant import act_quant as quant
    except Exception as exc:
        report.add(Case('moe_gemv', f'unavailable: {exc!r}', False))
        return
    if not torch.cuda.is_available():
        report.add(Case('moe_gemv', 'skipped (no CUDA)', True))
        return
    if not moe_gemv.available():
        report.add(Case('moe_gemv', 'kernels not built', False))
        return

    torch.manual_seed(30)
    # down's kernel walks k in 256-wide steps: the down projection's k_dim comes from
    # w2's shape, so pick sizes where that is a multiple of 256
    n_exp, k, n = 3, 512, 256
    lut = _lut().cuda()

    def codes(shape):
        return torch.randint(0, 16, shape, dtype=torch.uint8, device='cuda')

    w1c = codes((n_exp, n, k)); w3c = codes((n_exp, n, k)); w2c = codes((n_exp, k, n))
    w1 = [_fp4(w1c[e]) for e in range(n_exp)]
    w3 = [_fp4(w3c[e]) for e in range(n_exp)]
    w2 = [_fp4(w2c[e]) for e in range(n_exp)]
    s1 = [torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda') for _ in range(n_exp)]
    s3 = [torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda') for _ in range(n_exp)]
    s2 = [torch.full((k, n // 32), 127, dtype=torch.uint8, device='cuda') for _ in range(n_exp)]

    x = torch.randn(1, k, dtype=torch.bfloat16, device='cuda')
    aq, asc = quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
    aq = aq.view(torch.uint8).reshape(-1).contiguous()
    asc = asc.view(torch.uint8).reshape(-1).contiguous()

    # --- gate_up: [E, 2N] fp8-out of both projections ---
    try:
        got = moe_gemv.gate_up(aq, asc, w1, s1, w3, s3)
        af = aq.view(torch.float8_e4m3fn).float().reshape(-1, 32)
        ascf = torch.pow(2.0, (asc.to(torch.int32) - 127).float())
        def ref_proj(wc):
            # wc is [n_rows, k] of 4-bit codes -> dequantise with the same LUT
            wf = lut[wc.long()]
            out = torch.zeros(wc.shape[0], device='cuda')
            for kb in range(k // 32):
                out += (wf[:, kb * 32:(kb + 1) * 32] @ af[kb]) * ascf[kb]
            return out
        want = torch.stack([ref_proj(w1c[e]) for e in range(n_exp)])
        # output is [n_exp, 2*n_rows] bf16: gate rows first, then up
        gotf = got.float()[:, :n]
        check_close(report, 'moe_gemv gate_up', gotf, want, rtol=6e-2)
    except Exception as exc:
        report.add(Case('moe_gemv gate_up', f'raised {exc!r}', False))

    # --- act: silu(clamp(gate)) * clamp(up) * w ---
    try:
        gu = torch.randn(n_exp, 2 * n, dtype=torch.bfloat16, device='cuda') * 2
        we = torch.rand(n_exp, dtype=torch.float32, device='cuda') + 0.5
        limit = 3.0
        outa = moe_gemv.act(gu, we, limit)
        g = gu[:, :n].float().clamp(max=limit)
        u = gu[:, n:].float().clamp(-limit, limit)
        wanta = torch.nn.functional.silu(g) * u * we[:, None]
        check_close(report, 'moe_gemv act', outa, wanta, rtol=3e-2)
    except Exception as exc:
        report.add(Case('moe_gemv act', f'raised {exc!r}', False))

    # --- act_quant: the same, then block-32 fp8 quantisation in the same launch ---
    try:
        q, s = moe_gemv.act_quant(gu, we, limit)
        back = q.view(torch.float8_e4m3fn).float() * torch.pow(
            2.0, (s.view(torch.uint8).to(torch.int32) - 127).float()).repeat_interleave(32, dim=1)
        check_close(report, 'moe_gemv act_quant round-trip', back, wanta,
                    rtol=8e-2, note='(block-32 fp8: ~6% by construction)')
    except Exception as exc:
        report.add(Case('moe_gemv act_quant', f'raised {exc!r}', False))