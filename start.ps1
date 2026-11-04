# start.ps1 - start the resident tinyinfer engine with its OpenAI-compatible API.
#
# The engine loads once and picks its runtime path from the checkpoint itself, so whoever
# starts it decides which model is served. Every client then just connects.
#
#   .\start.ps1 -Model <dir>              # foreground
#   .\start.ps1 -Model <dir> -Bg          # background, logs to serve.log
#   $env:TINYINFER_MODEL = <dir>; .\start.ps1
#   .\start.ps1 -Model "<dirA>;<dirB>"     # one checkpoint striped over disks
param(
    # No baked-in default: the repository must not carry any one machine's checkpoint
    # path. Pass -Model, or export TINYINFER_MODEL in the shell that starts the engine.
    [string]$Model = $env:TINYINFER_MODEL,
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

if (-not $Model) {
    throw ("no checkpoint given: pass -Model <dir> (several dirs joined by ; for a " +
           "striped checkpoint), or set TINYINFER_MODEL")
}
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