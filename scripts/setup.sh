#!/usr/bin/env bash
# One-command reproducible dev environment for Linux (native, no Docker).
# Pins: Rust 1.96.0 (rust-toolchain.toml), Qdrant v1.19.0 binary, pinned corpora.
set -euo pipefail
cd "$(dirname "$0")/.."

QDRANT_VERSION="1.19.0"
QDRANT_SHA256="9ec667456443463eee390e43cd36988af6b730c6db807b4e39f57c303d0264a3"
REST_URL="${QORFINDER_QDRANT_REST_URL:-http://localhost:6333}"
QDRANT_URL="${QORFINDER_QDRANT_URL:-http://localhost:6334}"

step() { printf '\n==> %s\n' "$*"; }

command -v cargo >/dev/null 2>&1 || {
    echo "error: cargo not found. Install Rust: https://rustup.rs" >&2
    exit 1
}
command -v curl >/dev/null 2>&1 || {
    echo "error: curl is required" >&2
    exit 1
}

# --- Qdrant server (pinned binary) -------------------------------------
start_qdrant() {
    mkdir -p tools/qdrant
    if [ ! -x tools/qdrant/qdrant ]; then
        step "Downloading Qdrant v${QDRANT_VERSION} (Linux musl)"
        curl -fL -o /tmp/qdrant-${QDRANT_VERSION}.tar.gz \
            "https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz"
        echo "${QDRANT_SHA256}  /tmp/qdrant-${QDRANT_VERSION}.tar.gz" | sha256sum -c -
        tar -xzf "/tmp/qdrant-${QDRANT_VERSION}.tar.gz" -C tools/qdrant
    fi
    export QDRANT__STORAGE__STORAGE_PATH="$(pwd)/tools/qdrant/storage"
    nohup tools/qdrant/qdrant >tools/qdrant/qdrant.log 2>&1 &
    echo $! >tools/qdrant/qdrant.pid
    step "Started Qdrant (pid $(cat tools/qdrant/qdrant.pid), log: tools/qdrant/qdrant.log)"
}

wait_qdrant() {
    for _ in $(seq 1 60); do
        if curl -sf "${REST_URL}/healthz" >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "error: Qdrant did not become healthy at ${REST_URL}" >&2
    return 1
}

if curl -sf "${REST_URL}/healthz" >/dev/null 2>&1; then
    step "Qdrant already running at ${REST_URL}"
else
    start_qdrant
    wait_qdrant
fi

# --- Build (rust-toolchain.toml pins the compiler) ----------------------
step "Building release binary"
cargo build --release

# --- Model + corpora (pinned URLs) --------------------------------------
step "Warming embedding model cache"
./target/release/qorfinder warm

step "Preparing corpora (BEIR scifact + Quran)"
./target/release/qorfinder corpus beir scifact
./target/release/qorfinder corpus quran

# --- End-to-end verification ---------------------------------------------
step "Indexing SciFact and running evaluation smoke (30 queries)"
./target/release/qorfinder index data/scifact/corpus --once --collection scifact --qdrant "$QDRANT_URL"
./target/release/qorfinder eval data/scifact/corpus data/scifact/queries.tsv data/scifact/qrels.tsv --collection scifact --qdrant "$QDRANT_URL" --limit 30

step "Dev environment ready."
echo "  qdrant:   ${QDRANT_URL} (stop with: kill \$(cat tools/qdrant/qdrant.pid))"
echo "  corpora:  data/scifact, data/quran"
echo "  try:      ./target/release/qorfinder query 'what does the text say about zakat' --collection scifact --qdrant '${QDRANT_URL}'"
