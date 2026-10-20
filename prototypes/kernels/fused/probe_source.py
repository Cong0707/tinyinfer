import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels\fused")
import fused_ops as fo
k1 = fo.rmsnorm_rope_kernel(512, 64, 1e-20)
src1 = k1.get_kernel_source()
print("=== rmsnorm_rope source lines ===")
for line in src1.splitlines():
    s = line.strip()
    if any(x in s for x in ("float4", "uint4", "make_float", "cosf", "sinf", "rsqrtf", "expf", "__shfl", "__syncthreads")):
        print(s[:180])
print("total lines", len(src1.splitlines()))
k2 = fo.gemm_rope_kernel(512, 5120, 512, 64, 32)
src2 = k2.get_kernel_source()
print("\n=== gemm_rope source markers ===")
cnt = {}
for line in src2.splitlines():
    for m in ("mma", "cp.async", "float4", "uint4", "__syncthreads", "make_float4"):
        if m in line: cnt[m] = cnt.get(m,0)+1
print(cnt, "lines", len(src2.splitlines()))
for line in src2.splitlines()[:1]: pass
