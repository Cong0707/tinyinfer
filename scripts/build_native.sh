#!/usr/bin/env bash
# Compile the hand-written decode kernels to cubin (Linux/macOS counterpart of
# build_native.ps1).
#
# Sources live under src/tinyinfer/kernels/cuda, one flat directory (plus marlin/, whose
# headers include each other); every cubin lands in src/tinyinfer/kernels/build because the
# loader addresses a module by base name alone.
#
#   ./scripts/build_native.sh                  # everything
#   ./scripts/build_native.sh rms_norm gemv_bf16   # a subset
#   ARCH=sm_90 ./scripts/build_native.sh       # override the detected arch
#
# Marlin's two translation units need CUTLASS headers; set CUTLASS_PATH to build them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA="$ROOT/src/tinyinfer/kernels/cuda"
OUT="$ROOT/src/tinyinfer/kernels/build"
mkdir -p "$OUT"

if [ -z "${ARCH:-}" ]; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  if [ -n "$CC" ]; then ARCH="sm_${CC/./}"; else ARCH="sm_120"; fi
fi
echo "arch: $ARCH"

CUTLASS="${CUTLASS_PATH:-}"
# Two Marlin units export module names that differ from their file name.
module_name() {
  case "$1" in
    instances) echo "marlin_fp4" ;;
    repack)    echo "marlin_repack" ;;
    *)         echo "$1" ;;
  esac
}

want=("$@")
mapfile -t all < <(find "$CUDA" -name '*.cu' | sort)
[ "${#all[@]}" -gt 0 ] || { echo "no .cu under $CUDA" >&2; exit 1; }

for cu in "${all[@]}"; do
  base="$(basename "$cu" .cu)"
  mod="$(module_name "$base")"
  if [ "${#want[@]}" -gt 0 ]; then
    keep=0
    for w in "${want[@]}"; do
      if [ "$w" = "$base" ] || [ "$w" = "$mod" ]; then keep=1; break; fi
    done
    [ "$keep" = "1" ] || continue
  fi
  dst="$OUT/$mod.cubin"
  inc=()
  if [[ "$cu" == *marlin* ]]; then
    if [ -z "$CUTLASS" ]; then
      printf '%-20s skipped (set CUTLASS_PATH)\n' "$base"
      continue
    fi
    inc=(-I "$(dirname "$cu")" -I "$CUTLASS/include" --expt-relaxed-constexpr -std=c++17)
  fi
  if ! nvcc -cubin -arch="$ARCH" "${inc[@]}" -o "$dst" "$cu"; then
    echo "nvcc failed for $base" >&2
    exit 1
  fi
  printf '%-20s %8d bytes\n' "$mod" "$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst")"
done
