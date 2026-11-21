<#
.SYNOPSIS
    Installs the Python packages scripts\score.py needs, at the versions in
    requirements.txt, into the given Python (a venv, or CI's Python).

.DESCRIPTION
    As in flake.nix, ranx is installed without ir_datasets, whose only use
    (Qrels.from_ir_datasets) is then made a lazy import; ir_datasets pulls in
    packages that do not build on Windows. Used by the Windows CI job and by
    setup-windows.ps1 -WithScoring.

.EXAMPLE
    pwsh -File tools/install-scoring.ps1 -Python .venv\Scripts\python.exe
#>
[CmdletBinding()]
param(
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$requirements = Join-Path (Split-Path -Parent $PSScriptRoot) 'requirements.txt'

function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "$Python $args exited with $LASTEXITCODE" }
}

$lines = Get-Content -LiteralPath $requirements
$ranx = $lines | Where-Object { $_ -match '^ranx==' }
if (-not $ranx) { throw "no ranx pin in $requirements" }

$others = Join-Path ([IO.Path]::GetTempPath()) 'qorfinder-scoring-requirements.txt'
$lines | Where-Object { $_ -notmatch '^ranx==' } | Set-Content -LiteralPath $others
Invoke-Python -m pip install -r $others
Invoke-Python -m pip install --no-deps $ranx

$site = (& $Python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
$qrels = Join-Path $site 'ranx/data_structures/qrels.py'
$source = [IO.File]::ReadAllText($qrels)
if ($source -match '(?m)^import ir_datasets\r?$') {
    $source = $source -replace '(?m)^import ir_datasets\r?\n', ''
    $call = '        qrels = Qrels.from_dict(ir_datasets'
    if (-not $source.Contains($call)) { throw "unexpected $qrels; cannot make ir_datasets lazy" }
    $source = $source.Replace($call, "        import ir_datasets`n`n$call")
    [IO.File]::WriteAllText($qrels, $source)
}

Invoke-Python -c "from ranx import Qrels, Run, evaluate; from scipy.stats import wilcoxon"
Write-Host "scoring packages installed for $Python"
