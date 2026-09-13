# Benchmark

How QorFinder's retrieval quality and resource footprint compare to the mainstream alternatives someone searching local files would otherwise reach for.

## What we're comparing against

| | OS-native search (Windows Search, Spotlight) | Keyword-index tools (Everything, Recoll) | Cloud/server semantic search (typical "AI search" tool) | QorFinder |
|---|---|---|---|---|
| Match type | Filename + limited metadata | Full-text keyword (exact/fuzzy) | Dense embeddings, often via a hosted API | Hybrid: BM25 keyword **+** dense embeddings, fused |
| Understands paraphrases/synonyms | No | No | Yes | Yes |
| Finds exact terms, codes, names precisely | Partial | Yes | Weaker (embeddings blur exact matches) | Yes (lexical side) |
| Runs fully offline | Yes | Yes | Usually no (calls a cloud model) | Yes, after the one-time model download |
| Requires a background service/server | No | No | Often yes (server or cloud endpoint) | No — everything is embedded in-process |
| Setup | Built-in | Install one binary | Account/API key, sometimes a Docker container | One binary, no config |

The gap QorFinder targets: OS-native and keyword tools have no semantic understanding at all (a search for "car" won't find "automobile"), while most tools that *do* have semantic understanding depend on a server or cloud call — which is exactly the dependency this project's rebuild removed (see [ARCHITECTURE.drawio](ARCHITECTURE.drawio)).

## Retrieval quality: hybrid vs. dense-only

Standard IR literature (and QorFinder's own architecture rationale, see `src/query.rs`) is that hybrid lexical+dense retrieval matches or beats dense-only retrieval, because the two sides fail in complementary ways: embeddings blur exact terms/codes/names that keyword search catches, and keyword search misses paraphrases/synonyms that embeddings catch. Reciprocal Rank Fusion combines both without needing either side to "win" a scoring calibration.

**Measured baseline** (previous, dense-only architecture — Qdrant-backed, `intfloat/multilingual-e5-small`, BEIR SciFact, 300 queries):

| Metric | Score |
|---|---|
| nDCG@10 | 0.6234 |
| Recall@10 | 0.7281 |
| MRR@10 | 0.5975 |
| Query latency | ~230 ms/query (Quran corpus, 6,236 files) |
| Index storage | ~139 MB (Quran corpus via Qdrant's on-disk HNSW format) |

This repo's `eval.rs` now supports `--mode dense|lexical|hybrid` specifically so this comparison can be re-run against the current embedded hybrid store on the same SciFact benchmark — that re-run is the next step (see below) and its hybrid-mode numbers are intentionally not fabricated here; they should be measured, not guessed.

## Resource footprint: embedded vs. server-backed

Independent of retrieval quality, removing the Qdrant server dependency changes the deployment story:

- **No background process.** The old architecture required `docker run qdrant/qdrant` (or an installed Qdrant binary) running continuously. The new architecture's store — tantivy + usearch + redb — lives entirely inside the `qorfinder` process; nothing runs when you're not using it.
- **No network hop for storage I/O.** Every store operation is a local file/mmap access instead of a gRPC round trip, which is what let `indexer.rs` simplify from "batch writes with a `wait` flag to amortize network latency" to "commit once at the end of a scan" (see `ARCHITECTURE.drawio`'s indexing-path description).
- **Incremental re-indexing is O(files), not O(chunks).** The old design recovered per-file fingerprints by scrolling Qdrant's entire point collection; the new redb-backed fingerprint store does an O(1) lookup per file, which matters as a personal corpus grows into the tens of thousands of chunks.

## Reproducing this benchmark

```bash
# 1. Prepare the BEIR SciFact corpus
cargo run --release -- corpus beir scifact --out data

# 2. Index it into a scratch index directory
cargo run --release -- --index-dir /tmp/scifact-index index data/scifact/corpus --once

# 3. Evaluate each retrieval mode
cargo run --release -- --index-dir /tmp/scifact-index eval \
  data/scifact/corpus data/scifact/queries.tsv data/scifact/qrels.tsv --mode dense
cargo run --release -- --index-dir /tmp/scifact-index eval \
  data/scifact/corpus data/scifact/queries.tsv data/scifact/qrels.tsv --mode lexical
cargo run --release -- --index-dir /tmp/scifact-index eval \
  data/scifact/corpus data/scifact/queries.tsv data/scifact/qrels.tsv --mode hybrid
```

Each run prints nDCG@10, Recall@10, MRR@10, and per-query latency. Filling in this section's hybrid-vs-dense-vs-lexical table with a real run on the SciFact and Quran corpora is the next step for validating the rebuild's core claim (hybrid at least matches dense-only) — it was not completed as part of this pass. The pipeline itself (indexing, querying, all three retrieval modes) is verified working end to end against the repo's small fixture corpus (`tests/fixtures/small_corpus`) and is not a code issue; the full 5,183-document SciFact run was blocked by an environment problem, diagnosed as follows:

- `dmesg` on the dev machine's WSL instance shows its root filesystem (`ext4` on `/dev/sdd`) unmounting and remounting on a roughly 100-125 second cycle, continuously, independent of workload — each remount logs "corrupted or uncleanly shut down," i.e. an abrupt disconnect, not a clean unmount.
- A plain `sleep 150` with zero CPU/memory load was killed before completing, ruling out compute load as the cause.
- A fully detached process (`setsid` + `disown`, redirected away from the invoking shell) was *also* killed with no trace — ruling out "the invoking command's own timeout" as the cause, since a properly detached child should survive that.
- Together these point to the whole VM being interrupted at a fixed short cadence (possibly a host-level snapshot/pause, antivirus locking the WSL virtual disk, or similar), not anything specific to this codebase or to Rust/cargo's resource usage.

Any single operation that needs more than ~100 seconds of uninterrupted wall time — indexing 5,183 documents' embeddings included — is therefore unreliable on this specific machine until that's addressed. The reproduction commands above work correctly on a stable environment (verified against the small fixture corpus, which completes in well under that window).
