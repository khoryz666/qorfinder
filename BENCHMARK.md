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

**Measured** (current embedded architecture — tantivy + usearch + redb, `intfloat/multilingual-e5-small`, BEIR SciFact corpus, 5,183 documents, 300 judged queries):

| Metric | Dense only | Lexical only | Hybrid (RRF) | Previous dense-only baseline (Qdrant-backed) |
|---|---|---|---|---|
| nDCG@10 | 0.6237 | 0.6193 | **0.6654** | 0.6234 |
| Recall@10 | 0.7283 | 0.7325 | **0.7877** | 0.7281 |
| MRR@10 | 0.5972 | 0.5898 | **0.6338** | 0.5975 |
| Query latency | 11.9 ms/query | 0.6 ms/query | 13.1 ms/query | ~230 ms/query (Quran corpus, 6,236 files) |

Hybrid beats both single-mode retrieval paths on every metric (+0.042 nDCG, +0.059 Recall, +0.037 MRR over dense alone), confirming the rebuild's core architectural claim. The new dense-only path also reproduces the old Qdrant-backed baseline almost exactly (0.6237 vs. 0.6234 nDCG) — expected, since the retrieval math is unchanged and only the storage backend moved — and does so roughly 19x faster per query (11.9 ms vs. ~230 ms), consistent with removing the gRPC round trip to a separate server.

Index storage for this SciFact index (5,183 docs) is 48 MB (15 MB tantivy + 31 MB usearch + 2 MB redb) — smaller than the ~139 MB Qdrant baseline for a similarly-sized corpus (Quran, 6,236 docs), though the two aren't the same corpus so this is a directional comparison, not a controlled one.

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

Each run prints nDCG@10, Recall@10, MRR@10, and per-query latency; the numbers above are from this exact run against the full SciFact corpus (5,183 documents), indexed once and then evaluated in each mode against that same index.

The first attempt at this run was killed by the Linux OOM killer partway through indexing — not the VM-instability issue previously documented in this section (that was re-tested separately with a clean, uninterrupted 130-second `sleep` and did not reproduce, so it appears to have been transient to that earlier environment). `dmesg` showed the `qorfinder` process reaching several GB of resident memory before being killed. The cause: `indexer.rs` batched 256 chunks per embedding call (matching `fastembed`'s own default), and CPU transformer self-attention holds activations for the whole batch at once — memory that scales with `batch_size x sequence_length^2`. On this 384-dim/512-token model that meant 6-7 GB resident for a single batch, comfortably enough to OOM-kill an 8 GB machine. Lowering `EMBED_BATCH` to 32 (the conventional CPU sentence-embedding batch size) cut peak resident memory to ~1.9 GB for both a 1,000-document subset and the full 5,183-document corpus — confirming the memory ceiling is set by batch size, not corpus size — and the full run then completed cleanly in 14m8s.
