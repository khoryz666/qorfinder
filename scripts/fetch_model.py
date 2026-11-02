#!/usr/bin/env python3
"""Download an embedding model into models/ once.

QorFinder never touches the network itself (NFR1). Run this script once on
a connected machine; afterwards indexing and search work offline. Each
model's files are pinned to one Hugging Face revision and checked against
their SHA-256, so every machine embeds with exactly the same weights. The
models are those in crates/qf-core/src/embed/models.rs; the default is
QorFinder's default model.

Usage: python3 scripts/fetch_model.py [--model ID] [--out models]
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# model id -> (revision, {file name -> sha256 at that revision})
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        {
            "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
            "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
            "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        },
    ),
    "sentence-transformers/all-mpnet-base-v2": (
        "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        {
            "config.json": "d46a3e04ded82bba22528424480697d394eeda6a27484e08c5bb2bdf5906cfa0",
            "tokenizer.json": "b8be2c30ba5dd723a6d5ee26d013da103d5408d92ddcb23747622f9e48f1d842",
            "model.safetensors": "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658",
        },
    ),
    "BAAI/bge-base-en-v1.5": (
        "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        {
            "config.json": "bc00af31a4a31b74040d73370aa83b62da34c90b75eb77bfa7db039d90abd591",
            "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            "model.safetensors": "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7",
        },
    ),
    "intfloat/e5-base-v2": (
        "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        {
            "config.json": "01cc39aa39538a8179aa131c8a16adc03b506c92009f091502e4eb0c702f5f78",
            "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            "model.safetensors": "d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693",
        },
    ),
}
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repo: str, revision: str, name: str, dest: Path) -> None:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    revision, files = MODELS[args.model]
    # models/<name>, as qorfinder looks for it.
    model_dir = args.out / args.model.split("/")[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        dest = model_dir / name
        if not dest.exists():
            fetch(args.model, revision, name, dest)
        actual = sha256(dest)
        if actual != expected:
            print(f"{dest}: sha256 {actual}, expected {expected}", file=sys.stderr)
            return 1
        print(f"ok {dest} {actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
