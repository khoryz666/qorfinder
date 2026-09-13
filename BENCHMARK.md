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

The gap QorFinder targets: OS-native and keyword tools have no semantic understanding at all (a search for "car" won't find "automobile"), while most tools that *do* have semantic understanding depend on a server or cloud call. QorFinder avoids that dependency entirely — see [ARCHITECTURE.drawio](ARCHITECTURE.drawio) for how the store is embedded in-process.

## Retrieval quality: hybrid vs. dense vs. lexical

Standard IR literature (and QorFinder's own architecture rationale, see `src/query.rs`) is that hybrid lexical+dense retrieval matches or beats either mode alone, because the two sides fail in complementary ways: embeddings blur exact terms/codes/names that keyword search catches, and keyword search misses paraphrases/synonyms that embeddings catch. Reciprocal Rank Fusion combines both without needing either side to "win" a scoring calibration.

**Measured** (`intfloat/multilingual-e5-small`, BEIR SciFact corpus, 5,183 documents, 300 judged queries):

| Metric | Dense only | Lexical only | Hybrid (RRF) |
|---|---|---|---|
| nDCG@10 | 0.6237 | 0.6193 | **0.6654** |
| Recall@10 | 0.7283 | 0.7325 | **0.7877** |
| MRR@10 | 0.5972 | 0.5898 | **0.6338** |
| Query latency | 11.9 ms/query | 0.6 ms/query | 13.1 ms/query |

Hybrid beats both single-mode retrieval paths on every metric (+0.042 nDCG, +0.059 Recall, +0.037 MRR over dense alone), confirming the fusion is worth its small latency cost over lexical-only search.

Index storage for this SciFact index (5,183 docs) is 48 MB: 15 MB tantivy (lexical), 31 MB usearch (vectors), 2 MB redb (fingerprints).

## Resource footprint

The store — tantivy (lexical) + usearch (vector) + redb (fingerprints) — is embedded entirely inside the `qorfinder` process:

- **No background process.** Nothing runs when you're not using it; there's no server or container to start, stop, or keep alive.
- **No network hop for storage I/O.** Every store operation is a local file/mmap access, which is what lets `indexer.rs` commit a whole indexing scan once at the end instead of amortizing round-trip latency across batched writes.
- **Incremental re-indexing is O(files), not O(chunks).** Per-file fingerprints are an O(1) lookup in the redb side-table, so re-scanning a directory only re-embeds what actually changed, and cost doesn't grow with total chunk count as a personal corpus scales into the tens of thousands of chunks.

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

Each run prints nDCG@10, Recall@10, MRR@10, and per-query latency; the numbers above are from this exact run against the full SciFact corpus.

Indexing the full corpus initially exhausted memory and was killed by the Linux OOM killer partway through: `indexer.rs` batched 256 chunks per embedding call, and CPU transformer self-attention holds activations for the whole batch at once — memory that scales with `batch_size x sequence_length^2`. On this 384-dim/512-token model that meant 6-7 GB resident for a single batch, enough to OOM-kill an 8 GB machine. Lowering `EMBED_BATCH` to 32 (the conventional CPU sentence-embedding batch size) cut peak resident memory to ~1.9 GB regardless of corpus size, and the full run then completed cleanly in 14m8s.
