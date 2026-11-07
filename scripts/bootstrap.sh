#!/usr/bin/env bash
# tinyinfer bootstrap (Linux/macOS)
# 用法:
#   ./scripts/bootstrap.sh
#   TORCH_INDEX=cu130 ./scripts/bootstrap.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv
fi
PY=".venv/bin/python"

# uv 建的 venv 默认不含 pip，"$PY -m pip" 会直接 ModuleNotFoundError。
# 所有安装都通过 uv pip，并把 VIRTUAL_ENV 指向 .venv 让它落在这个环境里。
export VIRTUAL_ENV="$ROOT/.venv"

TORCH_INDEX="${TORCH_INDEX:-cu126}"
if ! uv pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"; then
  echo "torch 安装失败：换一个 index 重试，例如 TORCH_INDEX=cu124" >&2
  exit 1
fi

uv pip install -r requirements.txt
uv pip install -e . --no-deps

# The engine loads kernels from kernels/build/*.cubin; this is the step that turns the
# package from "works" into "fast". Without nvcc every dispatch falls back to torch and
# decode runs 5-10x slower, so a missing toolkit is a warning, not an install failure.
echo "==> 编译 kernel (cubin)"
if [ -x scripts/build_native.sh ] && command -v nvcc >/dev/null 2>&1; then
  bash scripts/build_native.sh || echo "警告: 部分 kernel 编译失败（看上面的 failed: 列表）"
else
  echo "警告: 找不到 nvcc，跳过 kernel 编译（会退回 torch 实现，慢 5-10 倍）"
  echo "      装好 CUDA toolkit 后再运行 bash scripts/build_native.sh"
fi

echo "==> 自检"
"$PY" - <<'PYEOF'
import torch
from pathlib import Path
from tinyinfer.kernels import native
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.0f} GB  {p.multi_processor_count} SM")
n = len(list((Path(native.__file__).parent / "build").glob("*.cubin")))
print("cubin:", n, "(0 means torch fallback)")
for m in ("fp8_gemm_mma", "fp8_gemv_bs", "gemv_fp4_w4a8"):
    print(f"  {m}: {'ok' if native.available(m, m) else 'MISSING'}")
PYEOF
echo ""
echo "完成。启动服务: ./start.sh --model <checkpoint-dir> --port 8080"
