# One-command reproducible dev environment for Windows (native, MSVC).
# Pins: Rust 1.96.0 (rust-toolchain.toml), Qdrant v1.19.0 binary, pinned corpora.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$QdrantVersion = "1.19.0"
$QdrantSha256 = "980cb2e1ae771155cf211da8c0a8a9206b6482bd4effdc4db994d3adb707b087"
$RestUrl = if ($env:QORFINDER_QDRANT_REST_URL) { $env:QORFINDER_QDRANT_REST_URL } else { "http://localhost:6333" }
$QdrantUrl = if ($env:QORFINDER_QDRANT_URL) { $env:QORFINDER_QDRANT_URL } else { "http://localhost:6334" }

function Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "cargo not found. Install Rust first: https://rustup.rs"
}

# --- Qdrant server (pinned binary) -------------------------------------
$toolsDir = Join-Path (Get-Location) "tools\qdrant"
function Test-QdrantHealthy {
    try { Invoke-RestMethod "$RestUrl/healthz" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

if (-not (Test-QdrantHealthy)) {
    New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
    $exe = Join-Path $toolsDir "qdrant.exe"
    if (-not (Test-Path $exe)) {
        Step "Downloading Qdrant v$QdrantVersion (Windows msvc)"
        $zip = Join-Path $env:TEMP "qdrant-$QdrantVersion.zip"
        Invoke-WebRequest -Uri "https://github.com/qdrant/qdrant/releases/download/v$QdrantVersion/qdrant-x86_64-pc-windows-msvc.zip" -OutFile $zip
        $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $QdrantSha256) {
            throw "qdrant zip sha256 mismatch: expected $QdrantSha256, got $actual"
        }
        Expand-Archive -Path $zip -DestinationPath $toolsDir -Force
    }
    $env:QDRANT__STORAGE__STORAGE_PATH = Join-Path $toolsDir "storage"
    $proc = Start-Process -FilePath $exe -WorkingDirectory $toolsDir -WindowStyle Hidden -PassThru
    Set-Content -Path (Join-Path $toolsDir "qdrant.pid") -Value $proc.Id
    Step "Started Qdrant (pid $($proc.Id))"
    for ($i = 0; $i -lt 60; $i++) {
        if (Test-QdrantHealthy) { break }
        Start-Sleep -Seconds 1
    }
    if (-not (Test-QdrantHealthy)) { throw "Qdrant did not become healthy at $RestUrl" }
} else {
    Step "Qdrant already running at $RestUrl"
}

# --- Build (rust-toolchain.toml pins the compiler) ----------------------
Step "Building release binary"
cargo build --release
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

# --- Model + corpora (pinned URLs) --------------------------------------
Step "Warming embedding model cache"
.\target\release\qorfinder.exe warm

Step "Preparing corpora (BEIR scifact + Quran)"
.\target\release\qorfinder.exe corpus beir scifact
.\target\release\qorfinder.exe corpus quran

# --- End-to-end verification ---------------------------------------------
Step "Indexing SciFact and running evaluation smoke (30 queries)"
.\target\release\qorfinder.exe index .\data\scifact\corpus --once --collection scifact --qdrant $QdrantUrl
.\target\release\qorfinder.exe eval .\data\scifact\corpus .\data\scifact\queries.tsv .\data\scifact\qrels.tsv --collection scifact --qdrant $QdrantUrl --limit 30

Step "Dev environment ready."
Write-Host "  qdrant:   $QdrantUrl (stop with: Stop-Process -Id (Get-Content tools\qdrant\qdrant.pid))"
Write-Host "  corpora:  data\scifact, data\quran"
Write-Host "  try:      .\target\release\qorfinder.exe query 'what does the text say about zakat' --collection scifact --qdrant $QdrantUrl"
