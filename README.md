# QorFinder

Local-first semantic search CLI. Two jobs: index a user-chosen directory into a local vector DB (Qdrant), and answer queries as top-k matching chunks with file references.

## Quick start

```powershell
# 1. Start Qdrant (gRPC on 6334, REST on 6333)
docker run -d -p 6333:6333 -p 6334:6334 -v qorfinder_data:/qdrant/storage qdrant/qdrant

# 2. Build (Windows, MSVC toolchain)
cargo build --release

# 3. Index a directory (--once skips watching)
.\target\release\qorfinder.exe index C:\path\to\docs
.\target\release\qorfinder.exe index C:\path\to\docs --once

# 4. Query
.\target\release\qorfinder.exe query "what does the text say about zakat" -k 5
```

- First run downloads the ONNX embedding model (~120 MB) into `~/.cache/qorfinder/models`; afterwards everything works offline
- Supported file types: `txt`, `md`, `markdown`, `pdf`, `docx`

## Configuration (env vars or flags)

| Env var                 | Flag              | Default                  |
|-------------------------|-------------------|--------------------------|
| `QORFINDER_QDRANT_URL`  | `--qdrant`        | `http://localhost:6334`  |
| `QORFINDER_COLLECTION`  | `--collection`    | `qorfinder`              |
| `QORFINDER_MODEL_CACHE` | `--model-cache`   | `~/.cache/qorfinder/models` |

## Dependencies

| Crate            | Purpose                                    |
|------------------|--------------------------------------------|
| `clap`           | CLI argument parsing                       |
| `notify`         | file watching (debounced)                  |
| `lopdf`          | PDF text extraction                        |
| `docx-rs`        | DOCX text extraction                       |
| `fastembed`      | local ONNX sentence embeddings             |
| `qdrant-client`  | Qdrant gRPC client (upsert/delete/search)  |
| `tokio`          | async runtime                              |
| `walkdir`        | directory traversal                        |
| `uuid`           | deterministic point IDs                    |

## Next Steps

- Want to develop or run tests? See [CONTRIBUTING.md](CONTRIBUTING.md)
- Curious about internals? See [design/ARCHITECTURE.md](design/ARCHITECTURE.md)
- Interested in benchmarks? See [design/EVALUATION.md](design/EVALUATION.md)
