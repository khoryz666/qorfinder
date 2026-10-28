#!/usr/bin/env python3
"""Write reference embeddings from Python sentence-transformers.

The Rust embedder must reproduce the original model: the test
`embeddings_match_sentence_transformers` embeds the same sentences with
candle and checks every cosine similarity is above 0.99 against the vectors
this script writes. Needs `pip install sentence-transformers` and the
network once; QorFinder itself needs neither.

Usage: python3 scripts/reference_embeddings.py
"""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # as in fetch_model.py
OUT = Path("crates/qf-core/tests/fixtures/minilm_reference.json")

SENTENCES = [
    "This Agreement shall be governed by the laws of the State of New York.",
    "Neither party may assign this Agreement without the prior written consent of the other party.",
    "During the Term and for two (2) years thereafter, Distributor shall not sell any competing products.",
    "Licensee shall pay Licensor a royalty of five percent (5%) of Net Sales, payable quarterly.",
    "Who owns the intellectual property created jointly under the collaboration?",
]


def main() -> int:
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    vectors = model.encode(SENTENCES, convert_to_numpy=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    reference = {
        "model": MODEL,
        "revision": REVISION,
        "max_seq_length": model.max_seq_length,
        "sentences": SENTENCES,
        "embeddings": [[round(float(x), 7) for x in v] for v in vectors],
    }
    OUT.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    print(f"wrote {len(SENTENCES)} embeddings of size {vectors.shape[1]} to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
