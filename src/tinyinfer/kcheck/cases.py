"""Conformance cases for the native CUDA kernels.

Each case states what the kernel claims and checks it against an independent
implementation of the same maths. The references here are deliberately written in
plain torch rather than reusing the engine's own code, so a wrong assumption in the
engine cannot make a test agree with itself.
"""
from __future__ import annotations

import torch

from . import Case, check_bitwise, check_close


def _e8m0(b: torch.Tensor) -> torch.Tensor:
    """e8m0 byte -> float value. The stored byte IS the biased exponent."""
    return torch.pow(2.0, (b.view(torch.uint8).to(torch.int32) - 127).to(torch.float32))


def _ref_act_quant(x, k, block=32):
    """Reference fp8 quantisation: amax floored at 1e-4, power-of-two scale."""
    flat = x.reshape(-1, k).float()
    nb = k // block
    g = flat.view(-1, nb, block)
    amax = g.abs().amax(dim=-1).clamp_min(1e-4)
    e = torch.ceil(torch.log2((amax / 448.0).double())).to(torch.int32)
    s = torch.pow(2.0, e.to(torch.float32))
    q8 = (g / s[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    code = (e + 127).clamp(0, 255).to(torch.uint8)
    return q8.reshape(flat.shape), code.reshape(x.shape[:-1] + (nb,))


def _act_quant(report):
    try:
        from ..kernels.quant import act_quant
    except Exception as exc:
        report.add(Case('act_quant', f'unavailable: {exc!r}', False))
        return
    torch.manual_seed(0)
    for (rows, k) in ((1, 5120), (4, 5120), (2, 2304), (3, 512)):
        x = torch.randn(rows, k, dtype=torch.bfloat16, device='cuda')
        # a zeroed group exercises the amax floor, where the e8m0 byte comes from the
        # clamp rather than from the data
        x[0, :64] = 0.0
        got_q, got_s = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
        want_q, want_s = _ref_act_quant(x, k)
        check_bitwise(report, f'act_quant fp8 [{rows},{k}]', got_q, want_q)
        check_bitwise(report, f'act_quant scale [{rows},{k}]', got_s, want_s)


def _fp4_quant(report):
    try:
        from ..kernels.fp4q import fp4_act_quant
    except Exception as exc:
        report.add(Case('fp4_quant', f'unavailable: {exc!r}', False))
        return
    torch.manual_seed(1)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device='cuda')
    for (rows, k, block, dtype) in ((1, 512, 32, torch.float8_e8m0fnu),
                                    (2, 256, 16, torch.float8_e4m3fn)):
        x = torch.randn(rows, k, dtype=torch.bfloat16, device='cuda')
        x[0, :block] = 0.0
        got_q, got_s = fp4_act_quant(x, block, scale_dtype=dtype)
        packed = got_q.view(torch.uint8)
        lo = (packed & 0x0F).long()
        hi = ((packed >> 4) & 0x0F).long()
        codes = torch.stack((lo, hi), dim=-1).reshape(rows, k)
        mag = lut[codes & 0x07]
        vals = torch.where((codes & 0x08) != 0, -mag, mag)
        s = _e8m0(got_s) if dtype == torch.float8_e8m0fnu else got_s.float()
        back = vals.view(rows, k // block, block) * s[..., None]
        check_close(report, f'fp4_quant roundtrip [{rows},{k}] blk{block}',
                    back.reshape(rows, k), x, rtol=0.25,
                    note='(e2m1 is coarse; this catches a wrong scale, not rounding)')


def _fp8_gemv(report):
    try:
        from ..kernels import native
        from ..kernels.quant import act_quant
    except Exception as exc:
        report.add(Case('fp8_gemv_bs', f'unavailable: {exc!r}', False))
        return
    try:
        native.kernel('fp8_gemv_bs', 'fp8_gemv_bs')
    except Exception as exc:
        report.add(Case('fp8_gemv_bs', f'kernel not built: {exc!r}', False))
        return
    torch.manual_seed(2)
    for (n, k) in ((2304, 5120), (512, 512), (5120, 2304)):
        # randn spans +-4 and e4m3 tops out at 448, so the cast is safe; a uniform draw
        # scaled up would silently become inf and make the comparison meaningless.
        w = torch.randn(n, k, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        ws = torch.full((n // 32, k // 32), 127, dtype=torch.uint8, device='cuda')
        x = torch.randn(1, k, dtype=torch.bfloat16, device='cuda')
        aq, asc = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
        out = torch.empty(n, device='cuda', dtype=torch.bfloat16)
        # grid/block must match gemm.fp8_linear: 8 rows per block (one warp per row)
        native.launch(native.kernel('fp8_gemv_bs', 'fp8_gemv_bs'), ((n + 7) // 8,), (256,),
                      [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
                       w.view(torch.uint8).reshape(-1), ws.view(-1), out, n, k])
        # The scale sits INSIDE the k loop: it is per 32-element block, so each block's
        # dot product has to be finished before it is scaled. A plain fp8 matmul (the
        # first version of this reference) drops that and is off by ~3 orders of
        # magnitude, which looks like a broken kernel and is a broken test.
        af = aq.reshape(1, k).view(torch.float8_e4m3fn).float()
        wf = w.float()
        asf = _e8m0(asc).reshape(-1)
        wsf = _e8m0(ws)
        idx = torch.arange(n, device='cuda') // 32
        want = torch.zeros(n, device='cuda', dtype=torch.float32)
        for kb in range(k // 32):
            part = (af[0, kb * 32:(kb + 1) * 32] * wf[:, kb * 32:(kb + 1) * 32]).sum(dim=1)
            want += part * asf[kb] * wsf[idx, kb]
        check_close(report, f'fp8_gemv_bs [{n},{k}]', out, want, rtol=3e-2)


def register(report) -> None:
    _act_quant(report)
    _fp4_quant(report)
    _fp8_gemv(report)
