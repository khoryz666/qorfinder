#!/usr/bin/env python3
"""Write reference embeddings from Python for every model in models/.

The Rust embedder must reproduce each original model: the test
`embeddings_match_sentence_transformers` embeds the same sentences (as
chunks) and queries with candle, checks every cosine similarity is above
0.99 against the vectors this script writes, and checks the model's
settings in crates/qf-core/src/embed/models.rs against the ones recorded
here. Needs sentence-transformers and the network once; QorFinder itself
needs neither.

The fine-tuned MiniLM (all-MiniLM-L6-v2-ft-cuad, from scripts/finetune_minilm.py)
is a bare BERT transformer with no sentence-transformers pooling module, so
it is embedded in plain transformers with QorFinder's own mean-pool and
unit-normalise instead. Models missing from models/ are skipped.

Usage: nix develop .#reference --command python3 scripts/reference_embeddings.py
"""

import hashlib
import json
import sys
from pathlib import Path

from fetch_model import MODELS

# The prefixes each model's card asks for, (query, passage), as in
# crates/qf-core/src/embed/models.rs.
PREFIXES = {
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
    "sentence-transformers/all-mpnet-base-v2": ("", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
}
FINE_TUNED = "all-MiniLM-L6-v2-ft-cuad"
# The fine-tuned model's input length, as the base MiniLM's.
FINE_TUNED_MAX_LEN = 256

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "models"
OUT = REPO / "crates/qf-core/tests/fixtures/reference"

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_pool_embeddings(model, tokenizer, texts: list[str], max_len: int):
    """Mean-pool over real tokens, then unit-normalise, as the Rust pipeline
    does. Left differentiable, so fine-tuning trains through it; wrap the
    call in torch.no_grad() for inference."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)


def save(name: str, reference: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.json"
    out.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    print(
        f"wrote {len(SENTENCES)} + {len(QUERIES)} embeddings of size "
        f"{reference['dim']} to {out}"
    )


def write_base(model_id: str) -> None:
    from sentence_transformers import SentenceTransformer

    revision = MODELS[model_id][0]
    query_prefix, passage_prefix = PREFIXES[model_id]
    model = SentenceTransformer(model_id, revision=revision, device="cpu")
    vectors = model.encode([passage_prefix + s for s in SENTENCES], convert_to_numpy=True)
    query_vectors = model.encode([query_prefix + q for q in QUERIES], convert_to_numpy=True)
    save(model_id.split("/")[-1], {
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
    })


def write_fine_tuned() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = MODELS_DIR / FINE_TUNED
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    model.eval()
    with torch.no_grad():
        vectors = mean_pool_embeddings(model, tokenizer, SENTENCES, FINE_TUNED_MAX_LEN)
        query_vectors = mean_pool_embeddings(model, tokenizer, QUERIES, FINE_TUNED_MAX_LEN)
    save(FINE_TUNED, {
        "model": FINE_TUNED,
        "revision": sha256(model_dir / "model.safetensors"),
        "max_seq_length": FINE_TUNED_MAX_LEN,
        "pooling": "mean",
        "dim": int(vectors.shape[1]),
        "query_prefix": "",
        "passage_prefix": "",
        "sentences": SENTENCES,
        "embeddings": rounded(vectors.tolist()),
        "queries": QUERIES,
        "query_embeddings": rounded(query_vectors.tolist()),
    })


def main() -> int:
    for model_id in MODELS:
        if (MODELS_DIR / model_id.split("/")[-1]).is_dir():
            write_base(model_id)
        else:
            print(f"skipped {model_id}: not in {MODELS_DIR}")
    if (MODELS_DIR / FINE_TUNED).is_dir():
        write_fine_tuned()
    else:
        print(f"skipped {FINE_TUNED}: not in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
