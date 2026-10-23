# Compile the hand-written decode kernels to cubin.
#
# nvcc needs the MSVC environment on Windows, so each call is wrapped in vcvars64.
# Every *.cu under src/tinyinfer/native is built; pass names to build a subset.
# The arch defaults to this GPU's compute capability, overridable with -Arch sm_90.
param(
    [string]$Arch = "",
    [string[]]$Only = @()
)
$ErrorActionPreference = "Stop"
$dir = Join-Path $PSScriptRoot "..\src\tinyinfer\native"

if (-not $Arch) {
    $cc = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($cc) { $Arch = "sm_" + $cc.Trim().Replace(".", "") } else { $Arch = "sm_120" }
}
Write-Output "arch: $Arch"

$vc = $env:VCVARS
if (-not $vc) {
    foreach ($p in @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    )) { if (Test-Path $p) { $vc = $p; break } }
}
if (-not $vc) { Write-Output "warning: vcvars64.bat not found; assuming cl.exe is on PATH" }

$files = if ($Only.Count -gt 0) { $Only } else {
    Get-ChildItem $dir -Filter *.cu | ForEach-Object { $_.BaseName }
}
foreach ($f in $files) {
    $cu = Join-Path $dir "$f.cu"
    $out = Join-Path $dir "$f.cubin"
    if (-not (Test-Path $cu)) { throw "no such kernel: $cu" }
    if ($vc) {
        cmd /c "call `"$vc`" >nul 2>&1 && nvcc -cubin -arch=$Arch -o `"$out`" `"$cu`""
    } else {
        cmd /c "nvcc -cubin -arch=$Arch -o `"$out`" `"$cu`""
    }
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $f" }
    Write-Output ("{0,-16} {1,8} bytes" -f $f, (Get-Item $out).Length)
}