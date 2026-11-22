#!/usr/bin/env python3
"""Score the TREC run files written by `qorfinder eval`.

Prints nDCG@10, MRR@10 and R-Precision for BM25, semantic search and the
RRF hybrid, for wording-matched, paraphrased and all queries (report
Section 3.4), scored with ranx. Then runs the Wilcoxon signed-rank test
(SciPy) on the per-query nDCG@10 of each pair of methods, and counts the
queries with a relevant chunk in the top 10, 30 and 100.

Usage: python3 scripts/score.py

It first checks itself against a run scored by hand, before trusting ranx
with the real runs (report Section 3.4), then scores results/eval, where
`qorfinder eval` writes, if it holds a run. It writes the printed tables to
scores.txt and the per-query scores to per_query.csv beside the TREC files.
"""

import json
import math
import sys
import tempfile
import types
from pathlib import Path

# ranx imports ir_datasets at the top but uses it only in Qrels.from_ir_datasets,
# which this script never calls; requirements.txt leaves it out.
sys.modules.setdefault("ir_datasets", types.ModuleType("ir_datasets"))
from ranx import Qrels, Run, evaluate  # noqa: E402
from scipy.stats import wilcoxon

METHODS = ["bm25", "semantic", "hybrid"]
METRICS = ["ndcg@10", "mrr@10", "r-precision"]
GROUPS = ["matched", "paraphrased", "all"]
# (a, b): is a different from b?
PAIRS = [("semantic", "bm25"), ("hybrid", "bm25"), ("hybrid", "semantic")]
# Depths for counting queries with a relevant chunk in the top k: nDCG@10
# scores a hit at rank 11 the same as none, so these show the near misses.
# Added after the main analysis (doc/REMARK.md §3); nDCG@10 stays primary.
HIT_DEPTHS = [10, 30, 100]

REPO = Path(__file__).resolve().parent.parent
RUN_DIR = REPO / "results/eval"
TESTSET = REPO / "data/testset"


def query_types(testset: Path) -> dict[str, str]:
    """Query id -> matched or paraphrased, from queries.jsonl."""
    with (testset / "queries.jsonl").open(encoding="utf-8") as f:
        return {q["id"]: q["type"] for q in map(json.loads, f) if q}


def per_query_scores(qrels: Qrels, run: Run) -> dict[str, dict[str, float]]:
    """metric -> query id -> score. Queries the run has no hits for score 0."""
    hits = [f"hit_rate@{k}" for k in HIT_DEPTHS]
    evaluate(qrels, run, METRICS + hits, make_comparable=True)
    return {m: dict(run.scores[m]) for m in METRICS + hits}


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
        lines.append("  queries with a relevant chunk in the top k:")
        lines.append(f"    {'method':<10}" + "".join(f"{'top ' + str(k):>9}" for k in HIT_DEPTHS))
        for method in METHODS:
            counts = "".join(
                f"{sum(scores[method][f'hit_rate@{k}'].get(q, 0.0) for q in ids):>9.0f}"
                for k in HIT_DEPTHS
            )
            lines.append(f"    {method:<10}{counts}")
        lines.append("")

    with (run_dir / "per_query.csv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("query,type," + ",".join(f"{me}_{mt}" for me in METHODS for mt in METRICS) + "\n")
        for q in sorted(types):
            values = ",".join(
                f"{scores[me][mt].get(q, 0.0):.4f}" for me in METHODS for mt in METRICS
            )
            f.write(f"{q},{types[q]},{values}\n")
    return "\n".join(lines)


def check_hand_worked() -> None:
    """Score a two-query run worked out by hand and exit if score.py
    disagrees with it."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "queries.jsonl").write_text(
            '{"id":"Q01m","category":"C","type":"matched","text":"t"}\n'
            '{"id":"Q01p","category":"C","type":"paraphrased","text":"t"}\n'
        )
        # Chunks 1 and 2 are relevant to both queries.
        (d / "qrels.trec").write_text("Q01m 0 1 1\nQ01m 0 2 1\nQ01p 0 1 1\nQ01p 0 2 1\n")
        # Q01m: 3, 1, 2 -> relevant at ranks 2 and 3.
        # Q01p: 1, 2    -> relevant at ranks 1 and 2 (perfect).
        run = "Q01m Q0 3 1 0.9 {t}\nQ01m Q0 1 2 0.8 {t}\nQ01m Q0 2 3 0.7 {t}\n" \
              "Q01p Q0 1 1 0.9 {t}\nQ01p Q0 2 2 0.8 {t}\n"
        for method in METHODS:
            (d / f"run_{method}.trec").write_text(run.format(t=method))
        text = report(d, d)
        header, first = (d / "per_query.csv").read_text().splitlines()[:2]
    row = dict(zip(header.split(","), first.split(",")))

    # Q01m: DCG = 1/log2(3) + 1/log2(4); ideal DCG = 1 + 1/log2(3).
    ndcg = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    # MRR@10 = 1/2; R = 2 and the top 2 hold one relevant: 1/2.
    expected = {"bm25_ndcg@10": ndcg, "bm25_mrr@10": 0.5, "bm25_r-precision": 0.5}
    errors = [
        f"{column} = {row[column]}, by hand {value:.4f}"
        for column, value in expected.items()
        if abs(float(row[column]) - value) > 5e-5
    ]
    if row["query"] != "Q01m":
        errors.append(f"first row is {row['query']}, not Q01m")
    # The mean over both queries: (0.6934 + 1) / 2.
    if f"{(ndcg + 1) / 2:13.4f}" not in text:
        errors.append("mean nDCG@10 over both queries is not (0.6934 + 1) / 2")
    # Both queries have a relevant chunk in the top 10, so in the top 30 and 100.
    if f"    bm25{2:>15}{2:>9}{2:>9}" not in text:
        errors.append("top-k counts are not 2, 2, 2 for both queries")
    if "semantic vs bm25: n/a (no query differs)" not in text:
        errors.append("identical methods got a Wilcoxon p-value")
    if errors:
        sys.exit("score.py disagrees with the hand-worked run:\n" + "\n".join(errors))
    print("score.py agrees with the hand-worked run")


def main() -> int:
    check_hand_worked()
    if not (RUN_DIR / "qrels.trec").exists():
        print(f"no run in {RUN_DIR}; run `qorfinder eval` first to score one")
        return 0
    text = report(RUN_DIR, TESTSET)
    print(text)
    (RUN_DIR / "scores.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
