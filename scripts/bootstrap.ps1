# tinyinfer bootstrap (Windows / PowerShell 7+)
# 用法:
#   .\scripts\bootstrap.ps1                 # 新建 .venv 并安装依赖（torch 走 cu130）
#   .\scripts\bootstrap.ps1 -ReuseEnv PATH  # 复用已存在的 venv（如本机预装 torch 的环境）
#   .\scripts\bootstrap.ps1 -TorchIndex cu132
param(
    [string]$ReuseEnv = "",
    [string]$TorchIndex = "cu130",
    [switch]$SkipTorch
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Get-Py {
    if ($ReuseEnv) {
        $py = Join-Path $ReuseEnv "Scripts\python.exe"
        if (-not (Test-Path $py)) { throw "ReuseEnv 中找不到 python.exe: $py" }
        return $py
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "未找到 uv。请先安装 uv: https://docs.astral.sh/uv/"
    }
    if (-not (Test-Path ".venv")) { uv venv --python 3.13 .venv | Out-Null }
    return (Join-Path $root ".venv\Scripts\python.exe")
}

$py = Get-Py
Write-Host "==> python: $py"
& $py -c "import sys; print(sys.version)"

if (-not $SkipTorch) {
    & $py -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> 安装 torch (index=$TorchIndex) ..."
        & $py -m pip install --upgrade pip | Out-Null
        & $py -m pip install "torch" --index-url "https://download.pytorch.org/whl/$TorchIndex"
    } else {
        Write-Host "==> 已有的 torch 可正常使用 CUDA"
    }
}

Write-Host "==> 安装 tinyinfer 基础依赖"
& $py -m pip install -r requirements.txt

Write-Host "==> 安装 tinyinfer 包 (editable)"
& $py -m pip install -e . --no-deps

Write-Host "==> 编译 kernel (cubin)"
# The engine loads kernels from kernels/build/*.cubin; this is the step that turns the
# package from "works" into "fast". Without nvcc every dispatch falls back to torch and
# decode runs 5-10x slower, so a missing toolkit is a warning, not an install failure.
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    try { & (Join-Path $PSScriptRoot "build_native.ps1") }
    catch { Write-Host "警告: kernel 编译失败: $_" }
} else {
    Write-Host "警告: 找不到 nvcc，跳过 kernel 编译（会退回 torch 实现，慢 5-10 倍）"
    Write-Host "      装好 CUDA toolkit 后再运行 scripts/build_native.ps1"
}

Write-Host "==> 自检"
& $py -c "import torch, tinyinfer; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('tinyinfer', tinyinfer.__version__)"
& $py -c "from pathlib import Path; from tinyinfer.kernels import native; n=len(list((Path(native.__file__).parent/'build').glob('*.cubin'))); print('cubin:', n, '(0 means torch fallback)')"

Write-Host ""
Write-Host "完成。运行: $py -m tinyinfer.cli --help"
