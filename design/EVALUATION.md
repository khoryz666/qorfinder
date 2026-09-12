# Evaluation

This document outlines how retrieval accuracy and system performance are measured.

## Corpora

1. **BEIR / SciFact**
   - Purpose: Accuracy benchmark.
   - Size: 5,183 scientific claims, 300 judged test queries.
   - Preparation: `scripts/prepare_beir.ps1 -Dataset scifact`

2. **Quran (Tanzil Uthmani + English)**
   - Purpose: Scale and storage domain testing.
   - Size: 6,236 files (one per ayah).
   - Preparation: `scripts/prepare_tanzil.ps1`

## Metrics

We evaluate retrieval quality using standard metrics against known judgments (qrels):
- **nDCG@10**: Normalized Discounted Cumulative Gain. Measures ranking quality.
- **Recall@10**: Proportion of relevant documents found in top 10 results.
- **MRR@10**: Mean Reciprocal Rank. Evaluates position of the first relevant document.

## Baseline Performance (v0.1.0)

**SciFact (300 queries)**
- nDCG@10: 0.6234
- Recall@10: 0.7281
- MRR@10: 0.5975

**Storage (Quran, 6,236 files)**
- Qdrant volume: ~139 MB (~5 KB/point including HNSW index).
- Query latency: ~230 ms per query average.

## Reproducing Benchmarks

Run the following scripts to reproduce evaluations:

```powershell
# 1. Prepare BEIR dataset
.\scripts\prepare_beir.ps1 -Dataset scifact

# 2. Index corpus
.\target\release\qorfinder.exe index .\data\scifact\corpus --once --collection scifact

# 3. Evaluate queries
.\target\release\qorfinder.exe eval .\data\scifact\corpus .\data\scifact\queries.tsv .\data\scifact\qrels.tsv --collection scifact
```
