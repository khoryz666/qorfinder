#!/usr/bin/env python3
"""Build the QorFinder test collection from CUAD v1.

CUAD (Hendrycks et al., 2021; CC BY 4.0) has 510 commercial contracts with
lawyer-annotated clause spans in 41 categories. Download CUAD_v1.zip from
https://zenodo.org/records/4595826 and unzip it under data/cuad/ first; this
script never touches the network.

Usage: python3 scripts/prepare_cuad.py [-n 50] [-x 0.3] [--seed 2026]

Try a few values of -n and -x until 15-20 categories qualify (report
Section 4.1.3).
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Contract metadata rather than clauses a lawyer would search for.
METADATA_CATEGORIES = {
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
}


def load_contracts(cuad_json: Path) -> tuple[list[dict], dict[str, str]]:
    """Contracts (title, context and {category: [(start, end)]}) and the
    description CUAD gives for every category."""
    data = json.loads(cuad_json.read_text(encoding="utf-8"))["data"]
    contracts = []
    descriptions = {}
    for doc in data:
        (paragraph,) = doc["paragraphs"]  # CUAD has one context per contract
        spans = {}
        for qa in paragraph["qas"]:
            category = qa["id"].split("__")[-1]
            descriptions[category] = qa["question"].split("Details: ", 1)[-1].strip()
            spans[category] = [
                (a["answer_start"], a["answer_start"] + len(a["text"]))
                for a in qa["answers"]
            ]
        contracts.append(
            {"title": doc["title"], "context": paragraph["context"], "spans": spans}
        )
    return contracts, descriptions


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


def sample_and_count(
    contracts: list[dict], n: int, seed: int, max_share: float
) -> tuple[list[dict], dict[str, int], list[str]]:
    """Sample n contracts with a fixed seed and count, for every clause
    category, how many sampled contracts contain it. A category qualifies if
    it appears in at least one and at most max_share of the sample, so it is
    rare enough that finding it is a real search task."""
    sample = random.Random(seed).sample(contracts, n)
    sample.sort(key=lambda c: c["file"])
    categories = sorted(
        {c for contract in contracts for c in contract["spans"]} - METADATA_CATEGORIES
    )
    counts = {
        category: sum(1 for c in sample if c["spans"].get(category))
        for category in categories
    }
    kept = [c for c in categories if 1 <= counts[c] <= max_share * n]
    return sample, counts, kept


def write_selection(
    out_dir: Path,
    sample: list[dict],
    counts: dict[str, int],
    kept: list[str],
    descriptions: dict[str, str],
) -> None:
    """docs.txt lists the sampled files; categories.tsv the kept categories
    with their counts and CUAD descriptions, for writing the queries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "docs.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(c["file"] + "\n" for c in sample)
    with (out_dir / "categories.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("category\tcontracts\tdescription\n")
        for category in kept:
            f.write(f"{category}\t{counts[category]}\t{descriptions[category]}\n")


def write_spans(out_dir: Path, sample: list[dict], kept: list[str]) -> None:
    """spans.tsv: one line per annotated span of a kept category in a
    sampled contract, as category, file, char_start, char_end (end
    exclusive, in characters). Judgments stay at span level, so the chunk
    size can change without redoing them."""
    lines = 0
    with (out_dir / "spans.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for contract in sample:
            for category in kept:
                for start, end in sorted(contract["spans"].get(category, [])):
                    f.write(f"{category}\t{contract['file']}\t{start}\t{end}\n")
                    lines += 1
    print(f"wrote {lines} spans to {out_dir / 'spans.tsv'}")


def print_counts(counts: dict[str, int], kept: list[str], n: int, max_share: float) -> None:
    print(f"\nN = {n} contracts, X = {max_share:.0%} (at most {int(max_share * n)})")
    print(f"{'category':<40} {'contracts':>9}  kept")
    for category, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        mark = "yes" if category in kept else ""
        print(f"{category:<40} {count:>9}  {mark}")
    print(f"{len(kept)} of {len(counts)} categories qualify")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cuad", type=Path, default=Path("data/cuad/CUAD_v1/CUAD_v1.json")
    )
    parser.add_argument("--txt-dir", type=Path, default=Path("data/cuad/txt"))
    parser.add_argument("--testset", type=Path, default=Path("data/testset"))
    parser.add_argument("-n", type=int, default=50, help="contracts to sample")
    parser.add_argument(
        "-x",
        type=float,
        default=0.3,
        help="largest share of the sample a category may appear in",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    contracts, descriptions = load_contracts(args.cuad)
    write_texts(contracts, args.txt_dir)
    sample, counts, kept = sample_and_count(contracts, args.n, args.seed, args.x)
    print_counts(counts, kept, args.n, args.x)
    write_selection(args.testset, sample, counts, kept, descriptions)
    write_spans(args.testset, sample, kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
