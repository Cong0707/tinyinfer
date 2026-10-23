# One-shot check: syntax, tests, and (optionally) a benchmark.
#   powershell -File scripts/check.ps1
#   powershell -File scripts/check.ps1 -Benchmark -Model <checkpoint dir>
param(
    [switch]$Benchmark,
    [switch]$Kernels,
    [string]$Model = "",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"

if (-not $Python) {
    $candidate = Join-Path $root ".venv\Scripts\python.exe"
    $Python = if (Test-Path $candidate) { $candidate } else { "python" }
}
Write-Host "[check] python: $Python"

Write-Host "[check] syntax"
& $Python -c "import ast, pathlib; root = pathlib.Path(r'$root/src/tinyinfer'); bad = [f.name for f in root.glob('*.py') if not ast.parse(f.read_text(encoding='utf-8'))]; print('files', len(list(root.glob('*.py')))); assert not bad, bad"

Write-Host "[check] pytest"
# The checkpoint-backed tests load the 510GB model; opt-in via -Benchmark.
# Tests marked "kernel" compile tilelang kernels on first run (minutes, +16GB of compiler
# memory) -- that is a build step, not a routine gate, so it is opt-in via -Kernels.
# pytest.ini already excludes "slow"; a command-line -m would replace it, so the
# exclusions are combined here instead of passed through
$kfilter = if ($Kernels) { "not slow" } else { "not slow and not kernel" }
& $Python -m pytest (Join-Path $root "tests") -q -p no:cacheprovider --tb=short -m $kfilter
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

if ($Benchmark) {
    if (-not $Model) { throw "-Benchmark needs -Model <checkpoint dir>" }
    Write-Host "[check] benchmark"
    & $Python -m tinyinfer.bench --model $Model --new-tokens 3
}

Write-Host "[check] ok"