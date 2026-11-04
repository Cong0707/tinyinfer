"""DeepSeek's own TileLang kernels, ported into tinyinfer.

Source of truth: DeepInfer (DeepSeek's serving stack),
deepinfer/model_executor/layers/tilelang_ops/cuda/*.py. Those kernels are written
against TileLang and then fanned out per backend (cuda/, asc/ for Ascend, ...),
which is the portability the engine wants: one source, more than NVIDIA.

Ported so far:

* compress_pool -- gated softmax pooling of score/latent into one compressed token
  per group, from cuda/compress.py.
* norm_rope -- RMSNorm followed by the rope rotation of the trailing dims, in one
  kernel (the norm+rope part of fused_transform.py). Used for the window KV path,
  which currently runs two separate torch ops per layer per token.
"""
import torch

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except Exception:  # pragma: no cover
    HAS_TILELANG = False


if HAS_TILELANG:

    @tilelang.jit(out_idx=[-1])
    def compress_pool_tl(DIM: int, RATIO: int, BLOCK: int = 128):
        """score/latent [R, DIM] fp32 -> pooled [R // RATIO, DIM] fp32."""
        n_groups = T.dynamic("n_groups")

        @T.prim_func
        def main(
            score: T.Tensor((n_groups * RATIO, DIM), "float32"),
            latent: T.Tensor((n_groups * RATIO, DIM), "float32"),
            out: T.Tensor((n_groups, DIM), "float32"),
        ):
            with T.Kernel(n_groups, T.ceildiv(DIM, BLOCK), threads=128) as (bg, bd):
                sf = T.alloc_fragment((RATIO, BLOCK), "float32")
                lf = T.alloc_fragment((RATIO, BLOCK), "float32")
                mx = T.alloc_fragment((BLOCK,), "float32")
                sm = T.alloc_fragment((BLOCK,), "float32")
                res = T.alloc_fragment((BLOCK,), "float32")
                for i, j in T.Parallel(RATIO, BLOCK):
                    sf[i, j] = score[bg * RATIO + i, bd * BLOCK + j]
                    lf[i, j] = latent[bg * RATIO + i, bd * BLOCK + j]
                T.reduce_max(sf, mx, dim=0, clear=True)
                for i, j in T.Parallel(RATIO, BLOCK):
                    sf[i, j] = T.exp(sf[i, j] - mx[j])
                T.reduce_sum(sf, sm, dim=0, clear=True)
                for i, j in T.Parallel(RATIO, BLOCK):
                    sf[i, j] = sf[i, j] / sm[j]
                for i, j in T.Parallel(RATIO, BLOCK):
                    lf[i, j] = lf[i, j] * sf[i, j]
                T.reduce_sum(lf, res, dim=0, clear=True)
                for j in T.Parallel(BLOCK):
                    if bd * BLOCK + j < DIM:
                        out[bg, bd * BLOCK + j] = res[j]

        return main

    @tilelang.jit(out_idx=[-1])
    def norm_rope_tl(DIM: int, ROT: int, EPS: float):
        """y = rope(rms_norm(x, w)) over the trailing ROT dims, adjacent-pair rotation.

        x [n, DIM] bf16, w [DIM] bf16, cos/sin [n, ROT // 2] fp32 -> [n, DIM] bf16.
        The rotation is the same view_as_complex pairing the reference uses.
        """
        n = T.dynamic("n")

        @T.prim_func
        def main(
            x: T.Tensor((n, DIM), "bfloat16"),
            w: T.Tensor((DIM,), "bfloat16"),
            cos: T.Tensor((n, ROT // 2), "float32"),
            sin: T.Tensor((n, ROT // 2), "float32"),
            y: T.Tensor((n, DIM), "bfloat16"),
        ):
            with T.Kernel(n, threads=128) as bn:
                xs = T.alloc_fragment((DIM,), "float32")
                sqv = T.alloc_fragment((DIM,), "float32")
                sq = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(DIM):
                    xs[j] = T.Cast("float32", x[bn, j])
                for j in T.Parallel(DIM):
                    sqv[j] = xs[j] * xs[j]
                T.reduce_sum(sqv, sq, dim=0, clear=True)
                inv = T.rsqrt(sq[0] / DIM + EPS)
                # stage the normalised row in shared memory: the rope pairs are a
                # strided access pattern that a fragment cannot express
                sr = T.alloc_shared((DIM,), "float32")
                for j in T.Parallel(DIM):
                    sr[j] = xs[j] * inv * T.Cast("float32", w[j])
                base = DIM - ROT
                for k in T.Parallel(ROT // 2):
                    c = cos[bn, k]
                    s = sin[bn, k]
                    a = sr[base + 2 * k]
                    b = sr[base + 2 * k + 1]
                    y[bn, base + 2 * k] = T.Cast("bfloat16", a * c - b * s)
                    y[bn, base + 2 * k + 1] = T.Cast("bfloat16", a * s + b * c)
                for j in T.Parallel(base):
                    y[bn, j] = T.Cast("bfloat16", sr[j])

        return main


    @tilelang.jit(out_idx=None)
    def rope_quant_tl(DIM: int, ROT: int, BLOCK: int = 32):
        """Rotation followed by block-fp8 quantisation in one kernel (exact, no reduction).

        Reference path: apply_rotary_emb then act_quant. Both are element-wise apart from
        the per-32 scale, which uses the same max(|x|)/448 rule as act_quant.
        """
        n = T.dynamic("n")
        nb = DIM // BLOCK

        @T.prim_func
        def main(
            x: T.Tensor((n, DIM), "bfloat16"),
            cos: T.Tensor((n, ROT // 2), "float32"),
            sin: T.Tensor((n, ROT // 2), "float32"),
            y: T.Tensor((n, DIM), "float8_e4m3"),
            s: T.Tensor((n, nb), "float8_e8m0fnu"),
        ):
            with T.Kernel(n, threads=128) as bn:
                sr = T.alloc_shared((DIM,), "float32")
                for j in T.Parallel(DIM):
                    sr[j] = T.Cast("float32", x[bn, j])
                base = DIM - ROT
                for k in T.Parallel(ROT // 2):
                    a = sr[base + 2 * k]
                    b = sr[base + 2 * k + 1]
                    c = cos[bn, k]
                    sn = sin[bn, k]
                    # apply_rotary_emb rounds back to bf16 before the quantiser runs
                    sr[base + 2 * k] = T.Cast("float32", T.Cast("bfloat16", a * c - b * sn))
                    sr[base + 2 * k + 1] = T.Cast("float32", T.Cast("bfloat16", a * sn + b * c))
                mx = T.alloc_fragment((BLOCK,), "float32")
                amax = T.alloc_fragment((1,), "float32")
                for blk in T.serial(nb):
                    for j in T.Parallel(BLOCK):
                        mx[j] = T.abs(sr[blk * BLOCK + j])
                    T.reduce_max(mx, amax, dim=0, clear=True)
                    # act_quant rounds the scale to a power of two (ue8m0); the value
                    # must be divided by that same power or quantisation doubles off
                    # act_quant rounds the scale UP to a power of two (fast_round_scale)
                    sc = T.exp2(T.ceil(T.log2(T.max(amax[0], 1e-4) / 448.0)))
                    s[bn, blk] = T.Cast("float8_e8m0fnu", sc)
                    for j in T.Parallel(BLOCK):
                        y[bn, blk * BLOCK + j] = T.Cast(
                            "float8_e4m3",
                            T.clamp(sr[blk * BLOCK + j] / sc, -448.0, 448.0))

        return main


def rope_quant_roundtrip(x: torch.Tensor, freqs: torch.Tensor, rot: int):
    """apply_rope + fp8 round-trip in one kernel (exact vs the reference, verified 0.0).

    x: [1, T, dim] bf16, freqs: [T, rot // 2] complex. Returns [1, T, dim] bf16, i.e.
    the value the window cache holds (quantised then dequantised).
    """
    if not HAS_TILELANG or not x.is_cuda or x.dim() != 3 or x.size(0) != 1:
        return None
    n, t, dim = x.shape
    flat = x.reshape(n * t, dim)
    fr = freqs.reshape(-1, freqs.shape[-1])
    y = torch.empty((n * t, dim), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((n * t, dim // 32), device=x.device, dtype=torch.float8_e8m0fnu)
    rope_quant_tl(int(dim), int(rot))(flat, fr.real.contiguous(), fr.imag.contiguous(), y, s)
    # dequantise back to bf16 (act_quant(..., inplace=True) semantics)
    out = (y.float() * s.float().repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    return out.reshape(x.shape)


def compress_pool(score: torch.Tensor, latent: torch.Tensor, ratio: int):
    """Pool ratio consecutive rows; None when the port does not cover the shape."""
    if not HAS_TILELANG or not score.is_cuda or score.shape[0] % ratio:
        return None
    s = score.float().contiguous()
    l = latent.float().contiguous()
    return compress_pool_tl(int(s.shape[1]), int(ratio))(s, l)


def norm_rope(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float, rot: int):
    """rms_norm + rope in one kernel; freqs is a complex table [n, rot // 2]."""
    if not HAS_TILELANG or not x.is_cuda or x.dim() != 3 or x.size(0) != 1:
        return None
    n, t, dim = x.shape
    flat = x.reshape(n * t, dim)
    fr = freqs[: x.size(1)].reshape(-1, freqs.shape[-1])
    return norm_rope_tl(int(dim), int(rot), float(eps))(
        flat, weight, fr.real.contiguous(), fr.imag.contiguous()).reshape(x.shape)
