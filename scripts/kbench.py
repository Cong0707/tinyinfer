"""Kernel-level bandwidth benchmark.

End-to-end token time on this box is dominated by memcpy (421.9 ms of 509 ms), so it
cannot tell a faster kernel from a stalled one. This measures each kernel against the
bytes it actually moves, which is the only number that transfers to another card.

Reference: Marlin reaches 393 GB/s on an expert matrix of this exact shape, so that is
the figure the hand-written kernels are judged against.
"""
import sys, time, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')

rows = []


def bench(fn, n=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def add(name, secs, nbytes, calls=1, note=''):
    rows.append((nbytes / secs / 1e9, name, secs * 1e6, nbytes / 1e6, calls, note))


# ---- expert GEMV: gate/up, [2304, 5120] fp4 ----
N, K = 2304, 5120
w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')
s = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8, device='cuda')
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
nb = N * K // 2 + N * K // 32

from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast
gemv_fp4_fast(x, w, s)  # repack once
add('expert gemv  marlin', bench(lambda: gemv_fp4_fast(x, w, s)), nb, 240, 'w4a16')

from tinyinfer.deepseek import fp4_native
aq, asc = fp4_native.quantize_act(x, K)
add('expert gemv  native',
    bench(lambda: fp4_native.gemv_fp4_pq(aq[0], asc[0], w, s)), nb, 240, 'w4a8')

# ---- fp8 dense GEMV: the shared expert / attention projections ----
from tinyinfer.kernels import native
try:
    N8, K8 = 2304, 5120
    w8 = torch.randint(0, 120, (N8, K8), dtype=torch.uint8, device='cuda')
    sc8 = torch.full((N8, K8 // 32), 127, dtype=torch.uint8, device='cuda')
    x8 = torch.randn(1, K8, dtype=torch.bfloat16, device='cuda')
    from tinyinfer.gemm import fp8_linear
    out8 = fp8_linear(x8, w8, sc8.view(torch.float8_e8m0fnu))
    nb8 = N8 * K8 + N8 * K8 // 32
    add('dense  fp8   gemv_bs', bench(lambda: fp8_linear(x8, w8, sc8.view(torch.float8_e8m0fnu))),
        nb8, 248, '')
except Exception as e:
    print(f'RESULT skip fp8: {e!r}')

# ---- head GEMV: vocab 129280 x 5120 ----
print()
print('RESULT %-26s %8s %9s %9s %6s' % ('kernel', 'GB/s', 'us', 'MB', 'calls'))
print('RESULT ' + '-' * 64)
for bw, name, us, mb, calls, note in sorted(rows, reverse=True):
    print('RESULT %-26s %8.1f %9.1f %9.1f %6d %s' % (name, bw, us, mb, calls, note))
print()
print('RESULT reference: marlin 393 GB/s, HBM 384 GB/s theoretical')