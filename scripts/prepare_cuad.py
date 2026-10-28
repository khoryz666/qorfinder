#!/usr/bin/env python3
"""Build the QorFinder test collection from CUAD v1.

CUAD (Hendrycks et al., 2021; CC BY 4.0) has 510 commercial contracts with
lawyer-annotated clause spans in 41 categories. Download CUAD_v1.zip from
https://zenodo.org/records/4595826 and unzip it under data/cuad/ first; this
script never touches the network.

Usage: python3 scripts/prepare_cuad.py [--cuad data/cuad/CUAD_v1/CUAD_v1.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def load_contracts(cuad_json: Path) -> list[dict]:
    """One dict per contract: title, context and {category: [(start, end)]}."""
    data = json.loads(cuad_json.read_text(encoding="utf-8"))["data"]
    contracts = []
    for doc in data:
        (paragraph,) = doc["paragraphs"]  # CUAD has one context per contract
        spans = {}
        for qa in paragraph["qas"]:
            category = qa["id"].split("__")[-1]
            spans[category] = [
                (a["answer_start"], a["answer_start"] + len(a["text"]))
                for a in qa["answers"]
            ]
        contracts.append(
            {"title": doc["title"], "context": paragraph["context"], "spans": spans}
        )
    return contracts


def file_name(title: str) -> str:
    """A file name for a contract title that is valid on Linux and Windows."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(". ") + ".txt"


def write_texts(contracts: list[dict], out_dir: Path) -> None:
    """Write every contract's context to its own .txt file, exactly as is.

    newline="" stops Python turning \\n into \\r\\n on Windows, so character
    offsets in the files match CUAD's annotation offsets one for one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = set()
    for contract in contracts:
        name = file_name(contract["title"])
        if name in names:
            sys.exit(f"two contracts map to the file name {name}")
        names.add(name)
        contract["file"] = name
        with (out_dir / name).open("w", encoding="utf-8", newline="") as f:
            f.write(contract["context"])
    print(f"wrote {len(contracts)} contracts to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cuad", type=Path, default=Path("data/cuad/CUAD_v1/CUAD_v1.json")
    )
    parser.add_argument("--txt-dir", type=Path, default=Path("data/cuad/txt"))
    args = parser.parse_args()

    contracts = load_contracts(args.cuad)
    write_texts(contracts, args.txt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
