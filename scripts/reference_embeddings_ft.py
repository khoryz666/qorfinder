#!/usr/bin/env python3
"""Write reference embeddings for the locally fine-tuned MiniLM.

The fine-tuned model (`all-MiniLM-L6-v2-ft-cuad`) is a bare BERT transformer
produced by scripts/finetune_minilm.py: it has no sentence-transformers
pooling module, so SentenceTransformer cannot load it. This script reproduces
QorFinder's own mean-pool + unit-normalize in plain transformers and writes the
same fixture format the Rust test `embeddings_match_sentence_transformers`
reads, on the same sentences and queries as every other model, so the Rust
embeddings can be checked against Python at cosine 1.0.

Usage: nix develop .#reference --command python3 scripts/reference_embeddings_ft.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from finetune_minilm import mean_pool_embeddings, sha256
from reference_embeddings import QUERIES, SENTENCES, rounded

MODEL_ID = "all-MiniLM-L6-v2-ft-cuad"
MODEL_DIR = Path("models/all-MiniLM-L6-v2-ft-cuad")
OUT = Path("crates/qf-core/tests/fixtures/reference/all-MiniLM-L6-v2-ft-cuad.json")


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR))
    model.eval()
    device = torch.device("cpu")
    with torch.no_grad():
        chunk_vectors = mean_pool_embeddings(model, tokenizer, SENTENCES, 256, device)
        query_vectors = mean_pool_embeddings(model, tokenizer, QUERIES, 256, device)

    reference = {
        "model": MODEL_ID,
        "revision": sha256(MODEL_DIR / "model.safetensors"),
        "max_seq_length": 256,
        "pooling": "mean",
        "dim": int(chunk_vectors.shape[1]),
        "query_prefix": "",
        "passage_prefix": "",
        "sentences": SENTENCES,
        "embeddings": rounded(chunk_vectors.tolist()),
        "queries": QUERIES,
        "query_embeddings": rounded(query_vectors.tolist()),
    }
    OUT.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    print(f"wrote {len(SENTENCES)} + {len(QUERIES)} embeddings of size {reference['dim']} to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
