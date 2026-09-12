# Changelog

All notable changes to this project will be documented in this file.

## [v0.1.0] - MVP

### Added
- Core CLI shell with `index`, `query`, `stats`, and `eval` commands.
- File watchdog with 2-second debounce for create, modify, and delete events.
- Parsers for `txt`, `md`, `pdf` (via lopdf), and `docx` (via docx-rs).
- Text chunker using fixed 512-character windows with 64-character overlaps.
- Local ONNX embedding integration via `fastembed` using `intfloat/multilingual-e5-small`.
- Vector storage and retrieval via Qdrant over gRPC.
- Reproducible benchmarking scripts for BEIR/SciFact and Quran corpora.

### Boundaries (Existing Features)
- Qdrant instance must be managed externally by the user (e.g. via Docker).
- Changing embedding models requires manual collection deletion and full re-indexing.
- No GUI is provided; all interactions are through the CLI.
