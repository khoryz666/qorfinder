# Benchmarks QorFinder execution: index duration, peak memory (qorfinder.exe),
# Qdrant storage size and memory, and end-to-end query latency.
#
# Requires: Qdrant running via Docker (docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant)
#
# Usage: .\scripts\bench.ps1 -CorpusDir data\quran\corpus [-Collection qorfinder-bench]
param(
    [Parameter(Mandatory = $true)][string]$CorpusDir,
    [string]$Collection = "qorfinder-bench",
    [string]$QdrantGrpcUrl = "http://localhost:6334",
    [string]$Binary = (Join-Path $PSScriptRoot "..\target\release\qorfinder.exe"),
    [int]$QueryRuns = 5,
    [string]$Query = "guide us to the straight path"
)
$ErrorActionPreference = "Stop"
$qdrantRest = "http://localhost:6333"

if (-not (Test-Path $Binary)) {
    throw "binary not found at $Binary (build it first: cargo build --release)"
}

Write-Host "== QorFinder benchmark =="
Write-Host "corpus:      $CorpusDir"
Write-Host "collection:  $Collection"
Write-Host "binary:      $Binary"

# 0. Qdrant health
try {
    Invoke-RestMethod "$qdrantRest/healthz" | Out-Null
} catch {
    throw "Qdrant not reachable at $qdrantRest (start it with docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant)"
}

# 1. Fresh collection
try { Invoke-RestMethod -Method Delete "$qdrantRest/collections/$Collection" | Out-Null } catch {}
Write-Host "`n[1/4] Indexing (memory sampled every 100 ms)..."
$outLog = Join-Path $env:TEMP "qorfinder-bench.out.log"
$errLog = Join-Path $env:TEMP "qorfinder-bench.err.log"
Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$argList = @("index", $CorpusDir, "--once", "--collection", $Collection, "--qdrant", $QdrantGrpcUrl)
$proc = Start-Process -FilePath $Binary -ArgumentList $argList -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -NoNewWindow
$peakBytes = 0
while (-not $proc.HasExited) {
    try { $ws = (Get-Process -Id $proc.Id).WorkingSet64 } catch { $ws = 0 }
    if ($ws -gt $peakBytes) { $peakBytes = $ws }
    Start-Sleep -Milliseconds 100
}
$proc.WaitForExit()
$sw.Stop()
if ($proc.ExitCode -ne 0) {
    Write-Host "index failed (see $errLog):"
    Get-Content $errLog -ErrorAction SilentlyContinue
    throw "index exited with code $($proc.ExitCode)"
}
$indexSeconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
$peakMb = [math]::Round($peakBytes / 1MB, 1)
Write-Host "  index time:     $indexSeconds s"
Write-Host "  peak working set: $peakMb MB"

# 2. Collection stats
$points = (& $Binary "stats" "--collection" $Collection "--qdrant" $QdrantGrpcUrl) -join " "
Write-Host "  $points"

# 3. Qdrant storage + memory
Write-Host "`n[2/4] Qdrant resource usage..."
$cid = docker ps -q --filter "ancestor=qdrant/qdrant" 2>$null | Select-Object -First 1
if ($cid) {
    $storage = (docker exec $cid du -sh /qdrant/storage 2>$null) -split "`t" | Select-Object -First 1
    Write-Host "  qdrant storage:  $storage (container $($cid.Substring(0,12)))"
    $mem = docker stats --no-stream --format "{{.MemUsage}}" $cid
    Write-Host "  qdrant memory:   $mem"
} else {
    Write-Host "  docker container not found; skipped qdrant storage/memory"
}

# 4. Query latency (full CLI invocation incl. model load + process startup)
Write-Host "`n[3/4] Query latency ($QueryRuns runs, full CLI invocation)..."
$times = @()
for ($i = 0; $i -lt $QueryRuns; $i++) {
    $t = Measure-Command {
        & $Binary "query" $Query "-k" "10" "--collection" $Collection "--qdrant" $QdrantGrpcUrl | Out-Null
    }
    $times += $t.TotalSeconds
}
$avg = ($times | Measure-Object -Average).Average
$max = ($times | Measure-Object -Maximum).Maximum
Write-Host "  avg: $([math]::Round($avg, 3)) s   max: $([math]::Round($max, 3)) s"

# 5. Summary
Write-Host "`n[4/4] Summary"
Write-Host "  index_time_s        = $indexSeconds"
Write-Host "  peak_memory_mb      = $peakMb"
Write-Host "  query_avg_s         = $([math]::Round($avg, 3))"
Write-Host "  query_max_s         = $([math]::Round($max, 3))"
Write-Host "  points              = $points"
