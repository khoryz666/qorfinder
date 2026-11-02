#!/usr/bin/env python3
"""Check score.py against numbers worked out by hand, before trusting ranx
with the real runs (report Section 3.4).

Usage: python3 scripts/test_score.py
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402


class HandWorkedRun(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        (d / "queries.jsonl").write_text(
            '{"id":"Q01m","category":"C","type":"matched","text":"t"}\n'
            '{"id":"Q01p","category":"C","type":"paraphrased","text":"t"}\n'
        )
        # Chunks 1 and 2 are relevant to both queries.
        (d / "qrels.trec").write_text(
            "Q01m 0 1 1\nQ01m 0 2 1\nQ01p 0 1 1\nQ01p 0 2 1\n"
        )
        # Q01m: 3, 1, 2 -> relevant at ranks 2 and 3.
        # Q01p: 1, 2    -> relevant at ranks 1 and 2 (perfect).
        run = "Q01m Q0 3 1 0.9 {t}\nQ01m Q0 1 2 0.8 {t}\nQ01m Q0 2 3 0.7 {t}\n" \
              "Q01p Q0 1 1 0.9 {t}\nQ01p Q0 2 2 0.8 {t}\n"
        for method in score.METHODS:
            (d / f"run_{method}.trec").write_text(run.format(t=method))
        self.text = score.report(d, d)
        self.per_query = (d / "per_query.csv").read_text().splitlines()

    def tearDown(self):
        self.dir.cleanup()

    def test_metrics_match_hand_calculation(self):
        # Q01m: DCG = 1/log2(3) + 1/log2(4); ideal DCG = 1 + 1/log2(3).
        ndcg = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg, 0.6934, places=4)
        # MRR@10 = 1/2; R = 2 and the top 2 hold one relevant: 1/2.
        row = dict(zip(self.per_query[0].split(","), self.per_query[1].split(",")))
        self.assertEqual(row["query"], "Q01m")
        self.assertAlmostEqual(float(row["bm25_ndcg@10"]), ndcg, places=4)
        self.assertAlmostEqual(float(row["bm25_mrr@10"]), 0.5, places=4)
        self.assertAlmostEqual(float(row["bm25_r-precision"]), 0.5, places=4)
        # The mean over both queries: (0.6934 + 1) / 2.
        self.assertIn(f"{(ndcg + 1) / 2:13.4f}", self.text)

    def test_identical_methods_have_no_wilcoxon_p(self):
        self.assertIn("semantic vs bm25: n/a (no query differs)", self.text)


if __name__ == "__main__":
    unittest.main()
