#!/usr/bin/env bash
# tinyinfer bootstrap (Linux/macOS)
# 用法:
#   ./scripts/bootstrap.sh
#   TORCH_INDEX=cu130 ./scripts/bootstrap.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d .venv ]; then
  uv venv --python 3.13 .venv
fi
PY=".venv/bin/python"

TORCH_INDEX="${TORCH_INDEX:-}"
if [ -n "$TORCH_INDEX" ]; then
  "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
else
  "$PY" -m pip install torch || echo "torch 安装失败：可设置 TORCH_INDEX=cu130 重试"
fi

"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e . --no-deps

# The engine loads kernels from kernels/build/*.cubin; this is the step that turns the
# package from "works" into "fast". Without nvcc every dispatch falls back to torch and
# decode runs 5-10x slower, so a missing toolkit is a warning, not an install failure.
echo "==> 编译 kernel (cubin)"
if command -v nvcc >/dev/null 2>&1; then
  ./scripts/build_native.sh || echo "警告: kernel 编译失败"
else
  echo "警告: 找不到 nvcc，跳过 kernel 编译（会退回 torch 实现，慢 5-10 倍）"
  echo "      装好 CUDA toolkit 后再运行 scripts/build_native.sh"
fi

echo "==> 自检"
"$PY" -c "import torch, tinyinfer; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$PY" -c "from pathlib import Path; from tinyinfer.kernels import native; n=len(list((Path(native.__file__).parent/'build').glob('*.cubin'))); print('cubin:', n, '(0 means torch fallback)')"
echo ""
echo "完成。运行: $PY -m tinyinfer.cli --help"
