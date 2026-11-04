# Compile the hand-written decode kernels to cubin.
#
# nvcc needs the MSVC environment on Windows, so each call is wrapped in vcvars64.
# Sources live under src/tinyinfer/kernels/cuda, one sub-directory per domain (gemm, norm,
# act, moe, hc, linear_attn); every cubin lands flat in src/tinyinfer/kernels/build because
# the loader addresses a module by base name alone.
#
# Pass -Only <names> to build a subset. The arch defaults to this GPU compute capability.
param(
    [string]$Arch = "",
    [string[]]$Only = @()
)
$ErrorActionPreference = "Stop"
$cuda = Join-Path $PSScriptRoot "..\src\tinyinfer\kernels\cuda"
$out  = Join-Path $PSScriptRoot "..\src\tinyinfer\kernels\build"
New-Item -ItemType Directory -Force -Path $out | Out-Null

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

# Marlin templates need CUTLASS headers: point CUTLASS_PATH at a checkout to build them.
$cutlass = $env:CUTLASS_PATH
# Two Marlin translation units export module names that differ from their file name.
$rename = @{ "instances" = "marlin_fp4"; "repack" = "marlin_repack" }

# powershell -File passes "-Only a,b,c" as one string; split it so both call styles work.
$Only = @($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ })

$files = Get-ChildItem $cuda -Recurse -Filter *.cu
if ($Only.Count -gt 0) {
    $files = $files | Where-Object { $Only -contains $_.BaseName -or $Only -contains $rename[$_.BaseName] }
}
if (-not $files) { throw "no .cu under $cuda" }

foreach ($f in $files) {
    $name = $f.BaseName
    $mod = if ($rename.ContainsKey($name)) { $rename[$name] } else { $name }
    $dst = Join-Path $out "$mod.cubin"
    $inc = ""
    if ($f.DirectoryName -match "marlin") {
        if (-not $cutlass) { Write-Output ("{0,-20} skipped (set CUTLASS_PATH)" -f $name); continue }
        $inc = "-I `"$($f.DirectoryName)`" -I `"$cutlass\include`" --expt-relaxed-constexpr -std=c++17"
    }
    if ($vc) {
        $outTxt = cmd /c "call `"$vc`" >nul 2>&1 && nvcc -cubin -arch=$Arch $inc -o `"$dst`" `"$($f.FullName)`" 2>&1"
        $code = $LASTEXITCODE
    } else {
        $outTxt = cmd /c "nvcc -cubin -arch=$Arch $inc -o `"$dst`" `"$($f.FullName)`" 2>&1"
        $code = $LASTEXITCODE
    }
    if ($code -ne 0) { $outTxt | Select-Object -Last 4; throw "nvcc failed for $name" }
    Write-Output ("{0,-20} {1,8} bytes  ({2})" -f $mod, (Get-Item $dst).Length, $f.Directory.Name)
}
