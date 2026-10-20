"""Minimal tilelang JIT feasibility test on Windows + sm_120.

Run from a shell where nvcc/cl.exe are on PATH (see prototypes/kernels/README.md).
"""
import os, sys, time, traceback
import torch

print("torch", torch.__version__, "cap", torch.cuda.get_device_capability())
print("CUDA_HOME env:", os.environ.get("CUDA_HOME"))
try:
    import tilelang
    import tilelang.language as T
    print("tilelang", tilelang.__version__)
except Exception:
    traceback.print_exc(); sys.exit(1)

print("tilelang.env.CUDA_HOME =", tilelang.env.CUDA_HOME)

# ---------- 1) trivial elementwise kernel ----------
def build_add(n=1024):
    @tilelang.jit
    def add_kernel(N):
        @T.prim_func
        def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
            with T.Kernel(1, threads=128) as bx:
                for i in T.Parallel(N):
                    B[i] = A[i] + 1.0
        return main
    return add_kernel(n)

t0 = time.time()
try:
    k = build_add(1024)
    a = torch.arange(1024, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    k(a, b)
    torch.cuda.synchronize()
    ok = torch.allclose(b, a + 1.0)
    print(f"[OK] elementwise tilelang JIT: {ok}  ({time.time()-t0:.1f}s)")
    print("  src head:", k.get_kernel_source()[:120].replace("\n", " ") if hasattr(k, "get_kernel_source") else "n/a")
except Exception:
    print(f"[FAIL] elementwise tilelang JIT after {time.time()-t0:.1f}s")
    traceback.print_exc()

# ---------- 2) small fp16 GEMM ----------
def build_gemm(M, N, K):
    @tilelang.jit
    def gemm_kernel(M, N, K, block_M=64, block_N=64, block_K=32):
        @T.prim_func
        def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
                 C: T.Tensor((M, N), "float16")):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), "float16")
                B_s = T.alloc_shared((block_K, block_N), "float16")
                C_l = T.alloc_fragment((block_M, block_N), "float32")
                T.clear(C_l)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                    T.copy(A[by * block_M, k * block_K], A_s)
                    T.copy(B[k * block_K, bx * block_N], B_s)
                    T.gemm(A_s, B_s, C_l)
                T.copy(C_l, C[by * block_M, bx * block_N])
        return main
    return gemm_kernel(M, N, K)

t0 = time.time()
try:
    M = N = K = 256
    kg = build_gemm(M, N, K)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    kg(A, B, C)
    torch.cuda.synchronize()
    ref = (A.float() @ B.float()).half()
    print(f"[OK] fp16 GEMM tilelang JIT: max err {(C.float()-ref.float()).abs().max().item():.4f}  ({time.time()-t0:.1f}s)")
except Exception:
    print(f"[FAIL] fp16 GEMM tilelang JIT after {time.time()-t0:.1f}s")
    traceback.print_exc()
