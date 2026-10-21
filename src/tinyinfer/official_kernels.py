"""Official block-scaled GEMV algorithm, ported to TileLang.

Source of truth: NVIDIA b12x (archived; now flashinfer/experimental/b12x),
b12x/gemm/blockscaled/_block_fp8_gemv.py. It is not installable here (CuTe DSL,
CUTLASS DSL 4.7.1, torch 2.12), so the algorithm is ported into the tree instead.

The CuTe original is a GEMV, not a GEMM: at decode row counts the dense
block-scaled GEMM is latency bound (one weight tile per CTA plus a second
split-K pass). Its structure is:

  * a CTA owns a low-dimensional slice of output features;
  * the weights are the MMA A operand, the (few) activation rows the B operand;
  * each scale block's partial is scaled in FP32 and accumulated in FP32;
  * the block sums are reduced in a fixed order, so the output is deterministic,
    and rounded to bf16 once.

TileLang's T.gemm requires every warp to own at least 16 MMA rows, so the
official "16 features, 8 warps split K" layout cannot be written directly: a
16-row tile fits a single warp. The port keeps the same math with a wider CTA
tile -- 128 output features, which 8 warps split exactly -- and walks K block by
block inside the CTA instead of across warps. The per-block FP32 scale and the
single bf16 rounding are identical, and the wider tile removes the cross-warp
staging entirely.
"""
import torch

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except Exception:  # pragma: no cover
    HAS_TILELANG = False

MAX_ROWS = 8
# b12x measured this crossover on SM120: past 4096 output features the dense GEMM
# streams the same bytes and keeps the weights resident, so the GEMV stops paying.
MAX_OUT_FEATURES = 4096
BLOCK = 128       # one scale block (DeepSeek's fp8 weights are 128x128)
CTA_N = 128       # output features per CTA = 8 warps x 16 rows
COLS = 8          # activation rows padded to one MMA column tile


def supports(rows: int, n: int, k: int, block: int = BLOCK) -> bool:
    """True when the ported kernel accepts this shape."""
    return (
        1 <= int(rows) <= MAX_ROWS
        and int(k) > 0
        and int(k) % int(block) == 0
        and 0 < int(n) <= MAX_OUT_FEATURES
    )


if HAS_TILELANG:

    @tilelang.jit(out_idx=[4])
    def gemv_fp8_n128(N: int, K: int, ROWS: int = 1, KB: int = BLOCK,
                      threads: int = 256, stages: int = 2):
        """out[m,n] = sum_kb (W[n,kb] . X[m,kb]) * sw[n//kb_rows, kb] * sx[m, kb].

        X (ROWS, K) e4m3, W (N, K) e4m3 contiguous, SX (ROWS, K//KB) f32,
        SW (ceil(N/KB), K//KB) f32 -> Out (ROWS, N) bf16.
        """
        blocks = K // KB

        @T.prim_func
        def main(
            X: T.Tensor((COLS, K), "float8_e4m3"),
            W: T.Tensor((N, K), "float8_e4m3"),
            SX: T.Tensor((COLS, blocks), "float32"),
            SW: T.Tensor((T.ceildiv(N, KB), blocks), "float32"),
            Out: T.Tensor((ROWS, N), "bfloat16"),
        ):
            with T.Kernel(T.ceildiv(N, CTA_N), threads=threads) as bn:
                acc = T.alloc_fragment((CTA_N, COLS), "float32")
                part = T.alloc_fragment((CTA_N, COLS), "float32")
                a_s = T.alloc_shared((COLS, KB), "float8_e4m3")
                b_s = T.alloc_shared((CTA_N, KB), "float8_e4m3")
                T.clear(acc)
                for kb in T.serial(blocks):
                    T.clear(part)
                    T.copy(X[0, kb * KB], a_s)
                    T.copy(W[bn * CTA_N, kb * KB], b_s)
                    T.gemm(b_s, a_s, part, transpose_A=False, transpose_B=True)
                    # one weight scale per 128 features, one activation scale per row
                    w_scale = SW[(bn * CTA_N) // KB, kb]
                    for i, r in T.Parallel(CTA_N, COLS):
                        acc[i, r] += part[i, r] * w_scale * SX[r, kb]
                for i, r in T.Parallel(CTA_N, COLS):
                    if bn * CTA_N + i < N and r < ROWS:
                        Out[r, bn * CTA_N + i] = T.cast(acc[i, r], "bfloat16")

        return main


def gemv_fp8_block(x: torch.Tensor, w: torch.Tensor, sx: torch.Tensor,
                   sw: torch.Tensor, block: int = BLOCK):
    """Run the ported GEMV; None when the shape is outside its contract."""
    if not HAS_TILELANG:
        return None
    rows, k = x.shape
    n = w.shape[0]
    if not supports(rows, n, k, block):
        return None
    # pad the activation rows to the MMA column tile (the official kernel keeps its
    # B operand 8 columns wide and clamps the loads the same way)
    xp = x.new_zeros((COLS, k))
    xp[:rows] = x.contiguous()
    sp = sx.new_zeros((COLS, k // block))
    sp[:rows] = sx
    fn = gemv_fp8_n128(n, k, KB=block)
    out = fn(xp, w.contiguous(), sp, sw.contiguous())
    return out[:rows]