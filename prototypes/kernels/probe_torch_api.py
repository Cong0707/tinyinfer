"""Probe torch 2.9 FP8/FP4 API surface on sm_120 (RTX 5060 Laptop)."""
import torch, inspect, sys

print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())

def has(name):
    return hasattr(torch, name)

for name in ["float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e8m0fnu",
             "float4_e2m1fn_x2", "uint4", "int4"]:
    print(f"dtype torch.{name}: {has(name)}")

print("_scaled_mm:", hasattr(torch, "_scaled_mm"))
try:
    print("scaled_mm sig:", inspect.signature(torch._scaled_mm))
except Exception as e:
    print("scaled_mm sig err:", e)

for fn in ["scaled_mm", "addmm", "matmul"]:
    print("F." + fn, hasattr(torch.nn.functional, fn))

import torch.backends.cuda as bc
print("cuda matmul allow_fp16_reduced_precision_reduction:", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
try:
    print("allow_fp8_reduced_precision_reduction:", torch.backends.cuda.matmul.allow_fp8_reduced_precision_reduction)
except Exception as e:
    print("fp8 red precision attr err:", e)

# torch version of cublaslt workspace config
for attr in ["allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction", "allow_fp8_reduced_precision_reduction"]:
    print(attr, getattr(torch.backends.cuda.matmul, attr, "N/A"))

print("cudnn sdp:", torch.backends.cuda.is_built(), torch.backends.cudnn.is_available())
print("flash sdp enabled:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient sdp enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math sdp enabled:", torch.backends.cuda.math_sdp_enabled())
print("cudnn sdp enabled:", torch.backends.cuda.cudnn_sdp_enabled())

print("torch.float4_e2m1fn_x2 itemsize:", getattr(torch.float4_e2m1fn_x2, "itemsize", "?"))
print("torch.float8_e4m3fn itemsize:", torch.float8_e4m3fn.itemsize)
print("torch.float8_e8m0fnu itemsize:", torch.float8_e8m0fnu.itemsize)
