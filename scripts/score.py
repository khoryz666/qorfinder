#!/usr/bin/env python3
"""Score the TREC run files written by `qorfinder eval`.

Prints nDCG@10, MRR@10 and R-Precision for BM25, semantic search and the
RRF hybrid, for wording-matched, paraphrased and all queries (report
Section 3.4), scored with ranx. Then runs the Wilcoxon signed-rank test
(SciPy) on the per-query nDCG@10 of each pair of methods.

Usage: python3 scripts/score.py [results/eval] [--testset data/testset]

Writes the printed tables to scores.txt and the per-query scores to
per_query.csv in the run folder, beside the TREC files.
"""

import argparse
import json
import sys
from pathlib import Path

from ranx import Qrels, Run, evaluate
from scipy.stats import wilcoxon

METHODS = ["bm25", "semantic", "hybrid"]
METRICS = ["ndcg@10", "mrr@10", "r-precision"]
GROUPS = ["matched", "paraphrased", "all"]
# (a, b): is a different from b?
PAIRS = [("semantic", "bm25"), ("hybrid", "bm25"), ("hybrid", "semantic")]


def query_types(testset: Path) -> dict[str, str]:
    """Query id -> matched or paraphrased, from queries.jsonl."""
    with (testset / "queries.jsonl").open(encoding="utf-8") as f:
        return {q["id"]: q["type"] for q in map(json.loads, f) if q}


def per_query_scores(qrels: Qrels, run: Run) -> dict[str, dict[str, float]]:
    """metric -> query id -> score. Queries the run has no hits for score 0."""
    evaluate(qrels, run, METRICS, make_comparable=True)
    return {m: dict(run.scores[m]) for m in METRICS}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def wilcoxon_p(a: list[float], b: list[float]) -> str:
    """Two-sided p-value, or why there is none. Pairs with equal scores are
    dropped (zero_method="wilcox"), as report Section 3.4 describes."""
    differing = sum(x != y for x, y in zip(a, b))
    if differing == 0:
        return "n/a (no query differs)"
    p = wilcoxon(a, b, zero_method="wilcox").pvalue
    return f"{p:.4f} ({differing} differing queries)"


def report(run_dir: Path, testset: Path) -> str:
    types = query_types(testset)
    qrels = Qrels.from_file(str(run_dir / "qrels.trec"), kind="trec")
    scores = {
        method: per_query_scores(
            qrels, Run.from_file(str(run_dir / f"run_{method}.trec"), kind="trec")
        )
        for method in METHODS
    }
    group_ids = {
        "matched": sorted(q for q, t in types.items() if t == "matched"),
        "paraphrased": sorted(q for q, t in types.items() if t == "paraphrased"),
        "all": sorted(types),
    }

    lines = []
    for group in GROUPS:
        ids = group_ids[group]
        lines.append(f"{group} queries ({len(ids)})")
        lines.append(f"  {'method':<10}" + "".join(f"{m:>13}" for m in METRICS))
        for method in METHODS:
            row = "".join(
                f"{mean([scores[method][m].get(q, 0.0) for q in ids]):>13.4f}"
                for m in METRICS
            )
            lines.append(f"  {method:<10}{row}")
        lines.append("  Wilcoxon signed-rank on nDCG@10, two-sided p:")
        for a, b in PAIRS:
            pa = [scores[a]["ndcg@10"].get(q, 0.0) for q in ids]
            pb = [scores[b]["ndcg@10"].get(q, 0.0) for q in ids]
            lines.append(f"    {a} vs {b}: {wilcoxon_p(pa, pb)}")
        lines.append("")

    with (run_dir / "per_query.csv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("query,type," + ",".join(f"{me}_{mt}" for me in METHODS for mt in METRICS) + "\n")
        for q in sorted(types):
            values = ",".join(
                f"{scores[me][mt].get(q, 0.0):.4f}" for me in METHODS for mt in METRICS
            )
            f.write(f"{q},{types[q]},{values}\n")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", type=Path, default=Path("results/eval"))
    parser.add_argument("--testset", type=Path, default=Path("data/testset"))
    args = parser.parse_args()
    text = report(args.run_dir, args.testset)
    print(text)
    (args.run_dir / "scores.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
