#!/usr/bin/env python3
"""Download every embedding model into models/ once.

QorFinder never touches the network itself (NFR1). Run this script once on
a connected machine; afterwards indexing and search work offline. Each
model's files are pinned to one Hugging Face revision and checked against
their SHA-256, so every machine embeds with exactly the same weights. The
models are those in crates/qf-core/src/embed/models.rs (about 1.4 GB in
all); files already present are only checked.

Usage: python3 scripts/fetch_model.py

Set QF_MODELS to a comma-separated list of short names (minilm, mpnet, bge,
e5) to fetch only those; CI sets QF_MODELS=minilm.
"""

import hashlib
import http.client
import itertools
import os
import socket
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
# QF_MODELS short name -> model id
SHORT_NAMES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "e5": "intfloat/e5-base-v2",
}
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# Seconds to wait for a connection to one address, and for any read.
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def connect(address, timeout=None, source_address=None, **_):
    """socket.create_connection, but trying IPv6 and IPv4 addresses in turn
    (RFC 8305) with a short timeout each. Python otherwise tries every IPv6
    address first and waits for the system's timeout on each, so on a network
    with broken IPv6 one download hung for over half an hour."""
    host, port = address
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    families = [[i for i in infos if i[0] == f] for f in (socket.AF_INET6, socket.AF_INET)]
    error = OSError(f"no address for {host}")
    for family, kind, proto, _, addr in filter(None, itertools.chain(*itertools.zip_longest(*families))):
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(CONNECT_TIMEOUT)
            if source_address:
                sock.bind(source_address)
            sock.connect(addr)
            sock.settimeout(timeout)
            return sock
        except OSError as e:
            error = e
            sock.close()
    raise error


class Connection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = connect


class Handler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(Connection, req, context=self._context)


def fetch(repo: str, revision: str, name: str, dest: Path) -> None:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", flush=True)
    opener = urllib.request.build_opener(Handler)
    with opener.open(url, timeout=READ_TIMEOUT) as response, tmp.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    tmp.replace(dest)


def fetch_model(model_id: str) -> bool:
    """Fetch the files of one model that are missing and check them all."""
    revision, files = MODELS[model_id]
    # models/<name>, as qorfinder looks for it.
    model_dir = MODELS_DIR / model_id.split("/")[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        dest = model_dir / name
        if not dest.exists():
            fetch(model_id, revision, name, dest)
        actual = sha256(dest)
        if actual != expected:
            # Deleted, or every rerun would only re-check the bad file.
            dest.unlink()
            print(f"{dest}: sha256 {actual}, expected {expected}; deleted, run again to re-download", file=sys.stderr)
            return False
        print(f"ok {dest} {actual}")
    return True


def main() -> int:
    names = os.environ.get("QF_MODELS", ",".join(SHORT_NAMES)).split(",")
    unknown = [n for n in names if n not in SHORT_NAMES]
    if unknown:
        print(f"QF_MODELS: unknown {unknown}; known {list(SHORT_NAMES)}", file=sys.stderr)
        return 1
    return 0 if all([fetch_model(SHORT_NAMES[n]) for n in names]) else 1


if __name__ == "__main__":
    sys.exit(main())
