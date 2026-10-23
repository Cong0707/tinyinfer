"""Decode-time fp4 MoE fused across every routed expert of a layer.

The tilelang path launches one GEMV per expert per projection, converts each expert's
e8m0 scale block to fp32 before the kernel can read it, and then spends about seven
more launches on clamp/silu/scale. At decode that is ~34 launches per layer; the scale
copies alone measured 205ms of a 790ms decode step.

'native/moe_gemv6.cu' decodes the fp4 weights, the e8m0 scale bytes and the fp8 e4m3
activation inline, with the expert index carried by grid.y, so one layer is four
launches and no conversion pass at all:

    gu = gate_up(aq, asc, w1, s1, w3, s3)   # [E, 2N]  fp8 e4m3 activation
    a  = act(gu, w_exp, limit)              # [E,  N]  clamp+silu+routing weight
    d  = down(a, w2, s2)                    # [E, 2D]  bf16 activation, no quantise
    acc(d, y, ...)                          # y += sum_e d[e]

Measured on synthetic weights at the real shapes (6 experts, w1/w3 2304x5120):
fused gate_up 0.225-0.279ms against 1.011ms for the twelve tilelang launches it
replaces, and the result matches an fp32 reference to within one bf16 ulp.
"""
import torch

from . import native

MAX_EXPERTS = 8
_fn_cache: dict[str, object] = {}
_ones: dict[int, torch.Tensor] = {}


def available() -> bool:
    """True when the hand-written kernels are built."""
    try:
        _kernel("moe_gemv6")
        return True
    except Exception:
        return False


def _kernel(name: str):
    k = _fn_cache.get(name)
    if k is None:
        k = _fn_cache[name] = native.kernel("moe_gemv6", name)
    return k


def supported_shape(n_rows: int, k_dim: int) -> bool:
    """The GEMV walks k_dim in 256-wide steps and needs a whole 16-row tile."""
    return k_dim > 0 and k_dim % 256 == 0 and n_rows > 0


def _check(n_rows: int, k_dim: int) -> None:
    if not supported_shape(n_rows, k_dim):
        raise ValueError(f"moe_gemv: unsupported shape n_rows={n_rows} k_dim={k_dim}")


# Pointer arrays are staged through pinned memory, because building them with
# torch.tensor(list, device="cuda") is a pageable H2D that synchronises the stream first:
# at decode that is 8 arrays per layer (320 per step, measured 441us each, 141ms of an
# 805ms step). The staging arrays CANNOT be shared or recycled on a short cycle --
# copy_ from pinned memory only queues the DMA, so a reused host buffer is overwritten
# while the copy is still pending and the kernel reads a pointer array belonging to a
# different projection. The symptom is not a crash but decode collapsing into one
# repeated token. A pool large enough to cover far more calls than can be in flight is
# therefore used, and a slot is only ever rewritten after the stream has long passed it.
_PTR_RING = 16
_ptr_pool: dict[tuple, list] = {}
_ptr_next = [0]


def _ptrs(tensors) -> torch.Tensor:
    """Device pointer array for one launch (staged through pinned memory).

    Each slot is its own tensor, not a row of a bigger one: a slice of a pinned tensor
    loses the pinned flag on the copy path, so ``copy_`` fell back to a synchronising
    pageable copy. That was measurable -- the n=6 arrays (down projection) cost 923us
    each while the n=12 ones were 6us, and the difference is exactly one queue drain.
    """
    dev = tensors[0].device
    n = len(tensors)
    key = (dev, n)
    pool = _ptr_pool.get(key)
    if pool is None:
        if len(_ptr_pool) > 32:
            _ptr_pool.clear()
        pool = _ptr_pool[key] = [
            (torch.empty(n, dtype=torch.int64, pin_memory=True),
             torch.empty(n, dtype=torch.int64, device=dev))
            for _ in range(_PTR_RING)]
    i = _ptr_next[0]
    _ptr_next[0] = (i + 1) % _PTR_RING
    host, gpu = pool[i]
    for j, t in enumerate(tensors):
        host[j] = t.data_ptr()
    gpu.copy_(host, non_blocking=True)
    return gpu


def _e8m0_bytes(s: torch.Tensor) -> torch.Tensor:
    """Raw e8m0 scale bytes, without the fp32 widening the tilelang path needs."""
    if s.dtype == torch.uint8:
        return s
    return s.view(torch.uint8)


def gate_up(aq: torch.Tensor, asc: torch.Tensor, w1, s1, w3, s3) -> torch.Tensor:
    """[n_exp, 2*n_rows] bf16: gate rows first, then up rows, per expert."""
    n_exp = len(w1)
    n_rows, khalf = w1[0].shape
    k_dim = khalf * 2
    _check(n_rows, k_dim)
    out = torch.empty((n_exp, 2 * n_rows), dtype=torch.bfloat16, device=aq.device)
    wl: list = []
    sl: list = []
    for e in range(n_exp):
        wl.append(w1[e])
        wl.append(w3[e])
        sl.append(s1[e])
        sl.append(s3[e])
    native.launch(_kernel("moe_gemv6"), ((n_rows + 15) // 16, n_exp, 2), (512,),
                  [aq, _e8m0_bytes(asc), _ptrs(wl), _ptrs(sl), out, n_rows, k_dim, n_exp,
                   0, 0], 0)   # one activation row and one scale row shared by every expert
    return out


def act(gu: torch.Tensor, w_exp: torch.Tensor, limit: float = 0.0):
    """[n_exp, n] bf16: silu(clamp(gate)) * clamp(up) * routing weight."""
    n_exp, two_n = gu.shape
    n = two_n // 2
    out = torch.empty((n_exp, n), dtype=torch.bfloat16, device=gu.device)
    if w_exp.dtype != torch.float32:
        w_exp = w_exp.float()
    native.launch(_kernel("moe_act6"), ((n + 255) // 256, n_exp, 1), (256,),
                  [gu, w_exp.contiguous(), out, n, float(limit), n_exp], 0)
    return out


def act_quant(gu: torch.Tensor, w_exp: torch.Tensor, limit: float = 0.0):
    """([n_exp, n] fp8 e4m3, [n_exp, n//32] e8m0): silu, routing weight and the
    reference's block quantisation in one launch.

    The down projection feeds on this rather than on a bf16 activation: fp8 e4m3 has a
    3-bit mantissa, and substituting the exact bf16 value shifted the final logits by 52%
    over forty layers even with each individual matmul inside a bf16 ulp.
    """
    n_exp, two_n = gu.shape
    n = two_n // 2
    if n % 32:
        raise ValueError(f"moe_gemv: activation length {n} is not a multiple of 32")
    aq = torch.empty((n_exp, n), dtype=torch.uint8, device=gu.device)
    asc = torch.empty((n_exp, n // 32), dtype=torch.uint8, device=gu.device)
    if w_exp.dtype != torch.float32:
        w_exp = w_exp.float()
    native.launch(_kernel("moe_act_q6"), (n // 32, n_exp, 1), (32,),
                  [gu, w_exp.contiguous(), aq, asc, n, float(limit), n_exp], 0)
    return aq.view(torch.float8_e4m3fn), asc.view(torch.float8_e8m0fnu)


def down(aq: torch.Tensor, asc: torch.Tensor, w2, s2) -> torch.Tensor:
    """[n_exp, 2*n_rows] bf16: one down projection per expert, plane 0 used."""
    n_exp = len(w2)
    n_rows, khalf = w2[0].shape
    k_dim = khalf * 2
    _check(n_rows, k_dim)
    out = torch.empty((n_exp, 2 * n_rows), dtype=torch.bfloat16, device=aq.device)
    # grid.z stays two planes wide so the kernel's plane indexing is untouched; only
    # plane 0 is launched and only the first n_rows of each expert are consumed. The
    # pointers must be interleaved -- [w_e0, w_e0, w_e1, w_e1, ...] -- because the kernel
    # indexes them as w_ptrs[e*2 + which]; concatenating two copies instead paired expert
    # e with expert (2e)'s weights.
    wl = [t for w in w2 for t in (w, w)]
    sl = [t for s in s2 for t in (s, s)]
    native.launch(_kernel("moe_gemv6"), ((n_rows + 15) // 16, n_exp, 1), (512,),
                  [aq, _e8m0_bytes(asc), _ptrs(wl), _ptrs(sl), out, n_rows, k_dim, n_exp,
                   k_dim, k_dim // 32], 0)   # [n_exp, k_dim] activation and scale rows
    return out


def acc(src: torch.Tensor, y: torch.Tensor, n_exp: int) -> None:
    """y += sum over experts of src; src is [n_exp, stride_e] inside a wider buffer."""
    n = y.numel()
    stride_e = src.stride(0)
    native.launch(_kernel("moe_acc6"), ((n + 255) // 256, 1, 1), (256,),
                  [src, y, n, n_exp, stride_e], 0)
