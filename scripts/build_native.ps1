# Compile the hand-written decode kernels to cubin.
# nvcc needs the MSVC environment, so each call is wrapped in vcvars64.
$ErrorActionPreference = "Stop"
$vc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$dir = Join-Path $PSScriptRoot "..\src\tinyinfer\native"
$files = @("sinkhorn", "head_gemv", "rms_norm", "moe_act", "moe_w13", "gemv_fp4_e8m0", "moe_gemv6")
if ($args.Count -gt 0) { $files = $args }
foreach ($f in $files) {
    $cu = Join-Path $dir "$f.cu"
    $out = Join-Path $dir "$f.cubin"
    cmd /c "call ""$vc"" >nul 2>&1 && nvcc -cubin -arch=sm_120 -o ""$out"" ""$cu"""
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $f" }
    Write-Output ("{0,-16} {1,8} bytes" -f $f, (Get-Item $out).Length)
}
