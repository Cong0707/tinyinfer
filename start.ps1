# start.ps1 - start the resident tinyinfer engine with its OpenAI-compatible API.
# The engine loads once; every request reuses the pool (no cold start between calls).
#
#   .\start.ps1            # foreground
#   .\start.ps1 -Bg        # background, logs to serve.log
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