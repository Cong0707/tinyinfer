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
"$PY" -c "import torch, tinyinfer; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
