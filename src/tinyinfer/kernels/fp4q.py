"""fp4 block quantisation on our own CUDA kernel."""
from __future__ import annotations

import torch


def fp4_act_quant(x, block_size: int = 32, inplace: bool = False,
                  scale_dtype: torch.dtype = torch.float8_e8m0fnu):
    """Same contract as official_kernel_ref.fp4_act_quant."""
    from . import native

    k = int(x.size(-1))
    if (not x.is_cuda or x.dtype != torch.bfloat16 or block_size not in (16, 32)
            or k % block_size or scale_dtype not in
            (torch.float8_e8m0fnu, torch.float8_e4m3fn)):
        from ..official_kernel_ref import fp4_act_quant as _ref

        return _ref(x, block_size, inplace, scale_dtype)

    use_e8m0 = 1 if scale_dtype == torch.float8_e8m0fnu else 0
    z = x.contiguous()
    n = z.numel() // k
    flat = z.view(n, k)
    nb = k // block_size
    s = z.new_empty(*z.shape[:-1], nb, dtype=scale_dtype)
    s_flat = s.view(torch.uint8).view(n, nb)
    ngroups = n * nb
    grid = ((ngroups * 32 + 127) // 128,)
    if inplace:
        y = torch.empty_like(flat)
        native.launch(native.kernel("fp4_quant", "fp4_quant_dequant"), grid, (128,),
                      [flat, y, s_flat, int(n), int(k), int(block_size), use_e8m0])
        x.copy_(y.view_as(x))
        return x
    y = torch.empty((n, k // 2), dtype=torch.uint8, device=z.device)
    native.launch(native.kernel("fp4_quant", "fp4_quant"), grid, (128,),
                  [flat, y, s_flat, int(n), int(k), int(block_size), use_e8m0])
    return y.view(torch.float4_e2m1fn_x2).view(*z.shape[:-1], k // 2), s
