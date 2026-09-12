# Targets

This document identifies the objectives and scope boundaries of QorFinder.

## Scope Boundaries
- **Local-first only**: No external APIs or cloud dependencies for execution.
- **CLI-centric**: The primary interface is the terminal. No GUI is planned within this repository.
- **Bring Your Own Qdrant**: For the current iteration, managing the Qdrant server is up to the user.

## Objectives
1. **Reduce Memory Usage**: Implement chunk-level batching instead of file-level batching during indexing to lower peak memory.
2. **Embedded Qdrant**: Integrate `qdrant-client` embedded mode to remove the Docker requirement for end-users.
3. **Advanced Chunking**: Implement sentence-boundary-aware and token-based chunking sizing.
4. **Hybrid Retrieval**: Combine dense embeddings with BM25 sparse vectors and/or a small cross-encoder reranker to improve top-k quality.
5. **Watchdog Improvements**: Handle edge cases such as directory moves, complex renames, and files locked by editors.
6. **Additional Parsers**: Add support for EPUB, XLSX, and HTML.
7. **Expanded Evaluation**: Test against additional BEIR sets (NFCorpus, ArguAna) and multilingual datasets.
8. **CI Infrastructure**: Add end-to-end tests in CI using Qdrant service containers and small pre-indexed fixtures.
