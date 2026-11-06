"""Block-32 fp8 -> bf16 dequantisation, on our own kernel."""

import torch


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """weight [O, I] float8_e4m3fn, scale [O/32, I/32] e8m0 bytes -> bf16."""
    from .kernels import native

    O, I = int(weight.shape[0]), int(weight.shape[1])
    if I % 32 or O % 32:
        raise ValueError(f'dequant_fp8 needs O%32==0 and I%32==0, got O={O} I={I}')
    out = torch.empty((O, I), dtype=torch.bfloat16, device=weight.device)
    # grid: x covers I in 16-element steps (256 threads), y covers O in 8-row tiles
    # 32x8 = 256 threads: 32 lanes cover 512 of I in 16-element steps, 8 rows of O
    grid = ((((I // 16) + 31) // 32), (O + 7) // 8)
    b = native.bind('dequant_fp8', 'dequant_fp8', grid, (32, 8), 'TTOii')
    b(weight.view(torch.uint8), scale.view(torch.uint8), out, int(O), int(I),
      stream=None)
    return out
