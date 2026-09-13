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

New to this repo? This is the full path from a fresh clone to a working build — no prior familiarity with the project assumed.

**1. Get the toolchain.** Rust, plus the C/C++ build tools the `usearch` vector-search crate needs, are pinned by a Nix flake — you don't install any of it by hand.

```bash
# Install Nix first if you don't have it: https://nixos.org/download
direnv allow      # if you use direnv — auto-loads the dev shell whenever you cd into the repo
# without direnv, enter the shell manually instead:
nix develop
```

Everything below assumes you're inside that shell (prefix any command with `nix develop --command` otherwise). Rust crates themselves are managed by `cargo` as usual — the flake only provides the toolchain and native libraries.

**2. Build it.**

```bash
cargo build            # debug build, fast to compile, for day-to-day development
cargo build --release  # optimized build — use this one to actually index/search anything
```

The first build takes a few minutes (it compiles the ONNX runtime, tantivy, and usearch from scratch); later builds are incremental.

**3. Run it.**

```bash
cargo run --release -- index ~/Documents --once
cargo run --release -- query "something you know is in one of those files"
```

The first `index`/`query`/`warm` invocation downloads a ~120 MB embedding model into `~/.cache/qorfinder/models`; every run after that is fully offline. See [Commands](#commands) above for the full list.

## Test

There are three tiers, roughly fastest/most-frequently-run to slowest/least-frequently-run.

**1. Fast checks — run these before every commit:**

```bash
cargo fmt --all -- --check                  # formatting
cargo clippy --all-targets -- -D warnings   # lints
cargo test --tests                          # everything that doesn't need the embedding model
```

`cargo test --tests` finishes in well under a second: pure logic (chunking, RRF fusion math, nDCG/Recall/MRR, corpus-prep formatting) and the embedded store engines (tantivy/usearch/redb) exercised against real temp directories, but nothing that needs the actual embedding model, the network, or a download.

Test code lives in [`tests/`](tests/), one file per module it covers (`tests/chunker.rs` tests `src/chunker.rs`, `tests/store.rs` tests the `Store` facade, and so on) — for anything reachable through the crate's public API. A handful of tests exercising genuinely private internals — the individual engines behind the `Store` facade (`src/store/{lexical,vector,fingerprint}.rs`) and `src/cli.rs`'s own argument-parsing/lock-handling plumbing — stay as `#[cfg(test)]` modules next to the code they test instead: those internals are deliberately not public API, and exposing them just to relocate a test would be the wrong trade.

**2. Integration tests — need the real embedding model; run before pushing:**

```bash
cargo test --test integration -- --ignored
```

Marked `#[ignore]` so they don't slow down the fast tier above. They index and query a small sample corpus (fetched fresh from Wikipedia's public API at test time, not committed as fixture files), drive the file watcher end to end (edit a file, confirm it's re-indexed; delete one, confirm it's un-indexed; confirm queries still work while the watcher is idly running), and exercise the real CLI dispatch path for `index` and `forget`. The first run downloads the embedding model if `cargo build`/`run` hasn't already; every run after that is offline.

**3. Retrieval-quality benchmark — occasional, takes about 15 minutes:**

```bash
cargo run --release -- corpus beir scifact --out data
cargo run --release -- --index-dir /tmp/scifact-index index data/scifact/corpus --once
cargo run --release -- --index-dir /tmp/scifact-index eval data/scifact/corpus data/scifact/queries.tsv data/scifact/qrels.tsv --mode hybrid
```

See [BENCHMARK.md](BENCHMARK.md) for the full reproduction steps (dense/lexical/hybrid modes) and the current measured numbers.

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
