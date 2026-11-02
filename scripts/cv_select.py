#!/usr/bin/env python3
"""Choose a setting by leave-one-category-out cross-validation.

Each category's matched and paraphrased queries are held out together. The
setting with the best mean score on the other categories is chosen, and the
held-out pair is scored with it, so no query's score comes from a setting
chosen with that query. Prints the cross-validated mean per query group and
how often each setting was chosen.

Usage: python3 scripts/cv_select.py COLUMN RUN_DIR:LABEL [RUN_DIR:LABEL ...]

COLUMN is a per_query.csv column written by score.py, such as
hybrid_ndcg@10. Ties go to the setting listed first.
"""

import csv
import sys
from collections import Counter


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    column = sys.argv[1]
    runs: dict[str, dict[str, tuple[str, float]]] = {}
    for arg in sys.argv[2:]:
        run_dir, label = arg.rsplit(":", 1)
        with open(f"{run_dir}/per_query.csv", encoding="utf-8") as f:
            runs[label] = {r["query"]: (r["type"], float(r[column])) for r in csv.DictReader(f)}
    labels = list(runs)
    queries = sorted(runs[labels[0]])
    # Q01m and Q01p share category Q01.
    categories = sorted({q[:3] for q in queries})

    held_out: dict[str, tuple[str, float]] = {}
    chosen = Counter()
    for category in categories:
        train = [q for q in queries if q[:3] != category]
        best = max(labels, key=lambda l: (sum(runs[l][q][1] for q in train), -labels.index(l)))
        chosen[best] += 1
        for q in queries:
            if q[:3] == category:
                held_out[q] = runs[best][q]

    for group in ["matched", "paraphrased", "all"]:
        ids = [q for q in queries if group == "all" or held_out[q][0] == group]
        print(f"CV {column} {group:12} {sum(held_out[q][1] for q in ids) / len(ids):.4f}")
    print("chosen:", ", ".join(f"{label} in {n} of {len(categories)} folds" for label, n in chosen.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
