# One-shot check: syntax, tests, and (optionally) a benchmark.
#   powershell -File scripts/check.ps1
#   powershell -File scripts/check.ps1 -Benchmark -Model E:\Models\DeepSeek-V4.1-Flash-1017
param(
    [switch]$Benchmark,
    [string]$Model = "",
    [string]$Python = "E:\Develop\Deepseek\tinyinfer\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"

Write-Host "[check] python: $Python"
if (-not (Test-Path $Python)) { throw "python not found: $Python" }

Write-Host "[check] syntax"
& $Python -c "import ast, pathlib; root = pathlib.Path(r'$root/src/tinyinfer'); bad = [f.name for f in root.glob('*.py') if not (lambda p: (ast.parse(p.read_text(encoding='utf-8')) and True))(f)]; print('files', len(list(root.glob('*.py')))); assert not bad, bad"

Write-Host "[check] pytest"
# the checkpoint-backed tests load the 510GB model; they are opt-in via -Benchmark
& $Python -m pytest (Join-Path $root "tests") -q -p no:cacheprovider --tb=short -k "not streaming_session"
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

if ($Benchmark) {
    if (-not $Model) { throw "-Benchmark needs -Model <checkpoint dir>" }
    Write-Host "[check] benchmark"
    & $Python -m tinyinfer.bench --model $Model --new-tokens 3
}

Write-Host "[check] ok"
