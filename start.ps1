# start.ps1 - start the resident tinyinfer engine with its OpenAI-compatible API.
#
# The engine loads once and picks its runtime path from the checkpoint itself, so whoever
# starts it decides which model is served. Every client then just connects.
#
#   .\start.ps1                          # foreground, DeepSeek default
#   .\start.ps1 -Bg                      # background, logs to serve.log
#   .\start.ps1 -Model "E:\Models\Qwen3.5-0.8B"
#   .\start.ps1 -Model "D:\part1;E:\part2"   # one checkpoint striped over disks
param(
    [string]$Model = "E:\Models\DeepSeek-V4.1-Flash-1017",
    [string]$Tier = "aggressive",
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Bg
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $here "src"
$py = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

if (-not (Test-Path ($Model.Split(";")[0]))) {
    throw "checkpoint not found: $Model (pass -Model <dir>, or several dirs joined by ;)"
}

$argLine = @("-m", "tinyinfer.server", "--backend", "tiny", "--model-dir", $Model,
             "--tier", $Tier, "--host", $HostAddr, "--port", "$Port")
Write-Host "[tinyinfer] starting resident server on http://${HostAddr}:$Port (tier=$Tier)"
if ($Bg) {
    Start-Process -NoNewWindow -FilePath $py -ArgumentList $argLine `
        -RedirectStandardOutput (Join-Path $here "serve.log") `
        -RedirectStandardError (Join-Path $here "serve.err.log")
    Write-Host "[tinyinfer] background pid logged; tail serve.log for readiness"
} else {
    & $py @argLine
}