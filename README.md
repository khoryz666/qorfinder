# QorFinder

An efficient semantic search desktop app for local files, using natural language processing. QorFinder indexes a directory on your machine and answers natural-language queries with the most relevant chunks, entirely offline after the first run — no server, no Docker, no cloud API.

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
- Supported file types: `txt`, `md`, `markdown`, `pdf`, `docx`. Files over 50 MB are skipped.
- To stop tracking a directory without deleting the whole index: `qorfinder forget <dir>`.

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

## Deploy

```bash
cargo build --release
```

The result is a single self-contained binary — no server, database, or container to deploy alongside it. Copy it wherever you want to run it.

## More

- [ARCHITECTURE.drawio](ARCHITECTURE.drawio): pipeline diagram and component descriptions.
- [BENCHMARK.md](BENCHMARK.md): how QorFinder's retrieval quality and resource use compare to mainstream local-search tools.
- `ref/`: the original project proposal and background paper.
