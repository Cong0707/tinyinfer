#!/usr/bin/env bash
# Build the cutlass-based fp8 blockwise GEMM shared library.
#
# Two targets, because the blockwise kernel differs by architecture and cutlass only
# ships the scale-config helper for each separately:
#   sm90  (H100/H200)  KernelTmaWarpSpecializedCooperativeFP8Blockwise
#   sm120 (RTX 50xx)   Sm120BlockwiseScaleConfig with tile K = scale granularity K
#
# The resulting .so is loaded by tinyinfer/kernels/cutlass_gemm.py through ctypes; it has
# no torch dependency, unlike vLLM's _C.so.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
cutlass="${CUTLASS_PATH:-$root/third_party/cutlass}"
arch="${ARCH:-sm_90a}"
out="$root/build"
mkdir -p "$out"

if [ ! -d "$cutlass/include/cutlass" ]; then
  echo "cutlass headers not found at $cutlass/include/cutlass" >&2
  echo "set CUTLASS_PATH, or run: git submodule update --init third_party/cutlass" >&2
  exit 1
fi

NVCC="${NVCC:-nvcc}"
shift_arch="${arch//sm_/}"

echo "cutlass : $cutlass"
echo "arch    : $arch"
echo "output  : $out"

# The sm120 file uses Sm120BlockwiseScaleConfig; the sm90 file uses the FP8Blockwise
# kernel schedule. Compiling both is cheap and lets one checkout serve both.
for src in fp8_gemm fp8_gemm_sm90; do
  [ -f "$root/src/tinyinfer/csrc/$src.cu" ] || continue
  echo "--- $src"
  "$NVCC" -O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
    -arch="$arch" -I "$cutlass/include" -I "$cutlass/tools/util/include" \
    --shared -Xcompiler -fPIC \
    -o "$out/libtinyinfer_$src.so" "$root/src/tinyinfer/csrc/$src.cu"
done

echo "done"; ls -la "$out"/libtinyinfer_*.so 2>/dev/null || true