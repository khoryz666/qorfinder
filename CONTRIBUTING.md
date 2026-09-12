# Contributing

This guide explains how to set up the reproducible development environment and test your code.

For architectural decisions, see [design/ARCHITECTURE.md](design/ARCHITECTURE.md). 
For benchmarking and evaluation details, see [design/EVALUATION.md](design/EVALUATION.md).

## Reproducible Environment

The dev environment is identical on Windows and Linux.

### Option A: Docker (Recommended)
```bash
# Start pinned Rust 1.96.0 toolchain + Qdrant v1.19.0
docker compose up -d

# Build, warm model, fetch corpora, and run eval smoke test
docker compose exec dev scripts/setup.sh
```

### Option B: Native
Run the native setup scripts based on your OS:
- Windows: `.\scripts\setup.ps1`
- Linux: `./scripts/setup.sh`

## Testing and Linting

All PRs must pass formatting, linting, and unit tests. Run these checks locally before committing:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib
```
Note: `cargo test --lib` runs fast, offline unit tests without requiring Qdrant or the model.
