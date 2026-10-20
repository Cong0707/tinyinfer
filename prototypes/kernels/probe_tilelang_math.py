import tilelang, tilelang.language as T, torch, re
names = [n for n in dir(T) if any(k in n.lower() for k in ("exp","sqrt","rsqrt","log","sin","cos","reinterpret","shift","ldexp","pow"))]
print("T symbols:", names)
@tilelang.jit
def k(N):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(N):
                Y[i] = T.exp(X[i]) + T.sqrt(X[i])
    return main
try:
    kk = k(256)
    src = kk.get_kernel_source()
    for line in src.splitlines():
        if any(s in line for s in ("exp", "sqrt", "rsqrt")):
            print("SRC:", line.strip()[:160])
except Exception as e:
    print("compile fail", type(e).__name__, str(e)[:300])
