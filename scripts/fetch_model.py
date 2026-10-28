#!/usr/bin/env python3
"""Download the embedding model into models/ once.

QorFinder never touches the network itself (NFR1). Run this script once on
a connected machine; afterwards indexing and search work offline. The files
are pinned to one Hugging Face revision and checked against their SHA-256,
so every machine embeds with exactly the same weights.

Usage: python3 scripts/fetch_model.py [--out models]
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# name -> sha256 at REVISION
FILES = {
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, dest: Path) -> None:
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    model_dir = args.out / REPO.split("/")[1]
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        dest = model_dir / name
        if not dest.exists():
            fetch(name, dest)
        actual = sha256(dest)
        if actual != expected:
            print(f"{dest}: sha256 {actual}, expected {expected}", file=sys.stderr)
            return 1
        print(f"ok {dest} {actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
