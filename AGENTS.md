# AGENTS.md

## What this is

QorFinder is a local-first semantic search engine, delivered as a CLI utility. It does exactly two things:

1. Indexes the files in a user-chosen "Target Directory" into a local vector DB
2. Converts each query into a sentence embedding and returns the top-k matching chunks with file references

## Architecture (pre-decided)

Hard invariants (silently wrong results if violated):

- The SAME embedding model must be used for indexing and querying
- The chosen model (`intfloat/multilingual-e5-small`, 384 dims, `MODEL_DIMS` in `src/embedder.rs`) fixes the collection's vector size; changing models means re-indexing everything
- e5 models require `query: ` / `passage: ` prompt prefixes on the two sides (see `src/embedder.rs`)
- Qdrant payload per point must carry file path, chunk index, and raw text — the Formatter needs them to display results without re-reading files
- Points use deterministic UUIDv5 IDs from `path:chunk_index` so re-indexing is idempotent

## Gotchas

- Qdrant gRPC is on **6334** (the Rust client's default), REST on 6333 — don't point the CLI at 6333
- `fastembed`'s default cache is CWD-relative (`.fastembed_cache`); the CLI pins it to `~/.cache/qorfinder/models`. First run needs network (~120 MB download), afterwards offline
- `qdrant-client` can't tell you how many points a delete removed (`UpdateResult` has no count) — see `Store::delete_file`
- Windows `canonicalize()` produces `\\?\`-prefixed paths; always use `dunce::canonicalize` (see `src/indexer.rs`) so payload paths stay clean

## Sources of truth

- [README.md](README.md): Quick start and CLI usage.
- [TARGETS.md](TARGETS.md): Clear objectives and scope boundaries. Do not propose features outside this scope.
- [design/ARCHITECTURE.md](design/ARCHITECTURE.md): Deep dive into system components and data flow.
- [design/EVALUATION.md](design/EVALUATION.md): Datasets and instructions for performance benchmarking.
- [CONTRIBUTING.md](CONTRIBUTING.md): Details on reproducible dev environments and testing.
