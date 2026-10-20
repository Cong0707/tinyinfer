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

Write-Host "==> 自检"
& $py -c "import torch, tinyinfer; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('tinyinfer', tinyinfer.__version__)"

Write-Host ""
Write-Host "完成。运行: $py -m tinyinfer.cli --help"
