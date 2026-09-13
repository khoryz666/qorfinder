# QorFinder

An efficient semantic search desktop app for local files, using natural language processing. QorFinder indexes a directory on your machine and answers natural-language queries with the most relevant chunks, entirely offline after the first run — no server, no Docker, no cloud API.

Targets Linux and WSL; there is no Windows-native build or CI coverage.

## Objectives

1. **Fast, accurate hybrid search.** Dense (sentence-embedding) and lexical (BM25 keyword) retrieval are run in parallel and fused with Reciprocal Rank Fusion, so exact terms/names that embeddings blur and paraphrases that keywords miss are both covered. Index size is a secondary concern versus speed and accuracy.
2. **Fully embedded, local-first architecture.** The whole store — lexical index, vector index, and per-file fingerprint bookkeeping — is embedded in-process (tantivy + usearch + redb). There is nothing to install, run, or keep alive besides the `qorfinder` binary itself; incremental re-indexing is O(files), not O(chunks).
3. **Desktop GUI integration** *(planned, not yet implemented)*. The current milestone ships the search engine as a CLI. A native desktop GUI wrapping this same engine is the next objective, so the tool becomes usable without a terminal.

## Quick start

```bash
# Build (see Develop below for the toolchain)
cargo build --release

# Index a directory (--once skips the file-watcher)
./target/release/qorfinder index ~/Documents --once

# Query it
./target/release/qorfinder query "what does the text say about zakat" -k 5
```

- First run downloads the ONNX embedding model (~120 MB) into `~/.cache/qorfinder/models`; everything after that is offline.
- The index lives under `~/.cache/qorfinder/indexes/default` by default (override with `--index-dir` or `QORFINDER_INDEX_DIR`).
- Supported file types: `txt`, `md`, `markdown`, `pdf`, `docx`. Files over 50 MB are skipped. PDF/DOCX extraction is text-only — scanned/image-only pages have no OCR and yield no text.
- To stop tracking a directory without deleting the whole index: `qorfinder forget <dir>`.

## Commands

| Command | Purpose |
|---|---|
| `index <dir> [--once] [--force] [--chunk-size N] [--chunk-overlap N]` | Index a directory, then watch it for changes (unless `--once`) |
| `query <text> [-k N]` | Search the index for the top-k matching chunks |
| `stats` | Show the number of indexed chunks |
| `forget <dir>` | Remove a directory's indexed files from the index, without touching disk |
| `eval <corpus> <queries> <qrels> [--mode dense\|lexical\|hybrid]` | Score retrieval quality (nDCG/Recall/MRR) against qrels — see [BENCHMARK.md](BENCHMARK.md) |
| `corpus beir <dataset>` / `corpus quran` | Download a benchmark corpus (no server needed) |
| `warm` | Download the embedding model and print its info, without indexing anything |

Global flags: `--index-dir` (or `QORFINDER_INDEX_DIR`) and `--model-cache` (or `QORFINDER_MODEL_CACHE`) work with every command.

## Develop

Toolchain (Rust, plus the C/C++ build tools `usearch` needs) is managed by a Nix flake; Rust crates themselves are managed by `cargo` as usual.

```bash
direnv allow      # or: nix develop
cargo build
```

## Test

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib
```

Unit tests are fast and fully offline — they never touch the embedding model or build a real index.

For an end-to-end check of the real pipeline (parse -> chunk -> embed -> index -> query) against the fixture corpus in `tests/fixtures/small_corpus`, run the ignored integration test — it needs the real embedding model, so it's excluded from the default `cargo test --lib` run:

```bash
cargo test --test integration -- --ignored
```

For retrieval-quality benchmarks (nDCG/Recall/MRR against BEIR-style corpora), see [BENCHMARK.md](BENCHMARK.md).

### Automation

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs fmt, clippy, unit tests, a release build, and both e2e checks (CLI smoke test + the ignored integration test) on every push and PR.
- [`.github/workflows/benchmark.yml`](.github/workflows/benchmark.yml) reproduces the full BEIR SciFact benchmark from BENCHMARK.md — dense/lexical/hybrid eval, plus a regression gate on hybrid nDCG@10 — on a weekly schedule and on demand (`workflow_dispatch`); it's not on every push since a full run takes ~15 minutes.
- [`.github/workflows/release.yml`](.github/workflows/release.yml) builds and publishes the release binary to GitHub Releases whenever a `v*` tag is pushed.

## Deploy

```bash
cargo build --release
```

The result is a single self-contained binary — no server, database, or container to deploy alongside it. Copy it wherever you want to run it.

To cut a release, push a `v*` tag (e.g. `git tag v0.1.0 && git push origin v0.1.0`); CI builds the release binary and publishes it to GitHub Releases automatically.

## More

- [ARCHITECTURE.drawio](ARCHITECTURE.drawio): pipeline diagram and component descriptions.
- [BENCHMARK.md](BENCHMARK.md): how QorFinder's retrieval quality and resource use compare to mainstream local-search tools.
- `ref/`: the original project proposal and background paper.

## License

MIT — see [LICENSE](LICENSE).
