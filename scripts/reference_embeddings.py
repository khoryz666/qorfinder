#!/usr/bin/env python3
"""Write reference embeddings from Python sentence-transformers.

The Rust embedder must reproduce each original model: the test
`embeddings_match_sentence_transformers` embeds the same sentences (as
chunks) and queries with candle, checks every cosine similarity is above
0.99 against the vectors this script writes, and checks the model's
settings in crates/qf-core/src/embed/models.rs against the ones recorded
here. Needs sentence-transformers and the network once; QorFinder itself
needs neither.

Usage: nix develop .#reference --command python3 scripts/reference_embeddings.py [--model ID]
(every model by default)
"""

import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from fetch_model import MODELS

# The prefixes each model's card asks for, (query, passage), as in
# crates/qf-core/src/embed/models.rs.
PREFIXES = {
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
    "sentence-transformers/all-mpnet-base-v2": ("", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
}
OUT = Path("crates/qf-core/tests/fixtures/reference")

SENTENCES = [
    "This Agreement shall be governed by the laws of the State of New York.",
    "Neither party may assign this Agreement without the prior written consent of the other party.",
    "During the Term and for two (2) years thereafter, Distributor shall not sell any competing products.",
    "Licensee shall pay Licensor a royalty of five percent (5%) of Net Sales, payable quarterly.",
    "Who owns the intellectual property created jointly under the collaboration?",
]

QUERIES = [
    "Which law governs the contract?",
    "Can the agreement be transferred to another company?",
]


def rounded(vectors):
    return [[round(float(x), 7) for x in v] for v in vectors]


def write(model_id: str) -> None:
    revision = MODELS[model_id][0]
    query_prefix, passage_prefix = PREFIXES[model_id]
    model = SentenceTransformer(model_id, revision=revision, device="cpu")
    vectors = model.encode([passage_prefix + s for s in SENTENCES], convert_to_numpy=True)
    query_vectors = model.encode([query_prefix + q for q in QUERIES], convert_to_numpy=True)
    reference = {
        "model": model_id,
        "revision": revision,
        "max_seq_length": model.max_seq_length,
        "pooling": model[1].pooling_mode,
        "dim": int(vectors.shape[1]),
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "sentences": SENTENCES,
        "embeddings": rounded(vectors),
        "queries": QUERIES,
        "query_embeddings": rounded(query_vectors),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{model_id.split('/')[-1]}.json"
    out.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    print(f"wrote {len(SENTENCES)} + {len(QUERIES)} embeddings of size {vectors.shape[1]} to {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, help="one model (default: every model)")
    args = parser.parse_args()
    for model_id in [args.model] if args.model else MODELS:
        write(model_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
