import tilelang, tilelang.language as T, torch
print("has floor:", hasattr(T, "floor"), "has ceil:", hasattr(T, "ceil"), "has round:", hasattr(T, "round"))
print("has __exp:", hasattr(T, "__exp"), "has ieee_frsqrt:", hasattr(T, "ieee_frsqrt"))

@tilelang.jit
def k(N):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32"), Z: T.Tensor((N,), "float32"), W: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(N):
                Y[i] = T.exp(X[i])
                Z[i] = T.__exp(X[i]) if hasattr(T, "__exp") else T.exp(X[i])
                W[i] = T.rsqrt(T.abs(X[i]) + 1.0)
    return main
kk = k(256)
src = kk.get_kernel_source()
for line in src.splitlines():
    if any(s in line for s in ("exp", "rsqrt", "sqrt", "floor", "ldexp")):
        print("SRC:", line.strip()[:200])

# bit trick + ldexp test inside tilelang
@tilelang.jit
def k2(N):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(N):
                z = X[i] * 1.4426950408889634
                zi = T.floor(z)
                f = z - zi
                p = 1.0 + f * (0.6931471805599453 + f * (0.2402265069591007 + f * 0.0555041086648216))
                bits = T.Cast("int32", zi) + 127
                scale = T.reinterpret("float32", T.shift_left(bits, 23))
                Y[i] = p * scale
    return main
try:
    kk2 = k2(256)
    x = torch.linspace(-8, 8, 256, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x); kk2(x, y); torch.cuda.synchronize()
    import math
    ref = torch.exp(x)
    print("poly exp max rel err:", ((y-ref).abs()/ref).max().item())
    for line in kk2.get_kernel_source().splitlines():
        if any(s in line for s in ("exp", "ldexp", "reinterpret", "floor", "shift")):
            print("SRC2:", line.strip()[:200])
except Exception as e:
    import traceback; traceback.print_exc()
