#!/usr/bin/env python3
"""Sanity-check the hand-written paraphrase queries used for fine-tuning.

The training queries are 3 hand-written paraphrases per category. Before any
training, this script verifies each one is a genuine, *load-bearing* query for
its category: under the **base** (un-fine-tuned) MiniLM, a template's own
category clauses should be closer than other categories' clauses, and its
nearest clause should belong to its own category far more often than chance
(1/17 ~ 6%). A template that fails this is a bad paraphrase and should be
rewritten before it poisons the training data.

Only held-out contracts are used (the 30 evaluation contracts are excluded);
the frozen test queries are read only for the phrase audit. Nothing is
downloaded.

Usage (reference devshell):

    nix develop .#reference --command python3 scripts/check_templates.py

Exits 0 if the templates pass (mean margin clearly positive, top-1 hit rate
well above chance), 1 otherwise. Writes nothing; prints a report.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from finetune_minilm import (
    audit_templates,
    QUERY_TEMPLATES,
    clean,
    load_contracts,
    mean_pool_embeddings,
    seed_everything,
    set_threads,
)


def embed_batched(model, tokenizer, texts: list[str], max_len: int, device, batch: int):
    rows = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            rows.append(
                mean_pool_embeddings(model, tokenizer, texts[i : i + batch], max_len, device)
            )
    return torch.cat(rows, dim=0)


def binomial_two_sided_p(k: int, n: int, p: float) -> float:
    """Two-sided p-value that P(Bin(n, p) is at least as extreme as k)."""
    if k > n * p:
        # upper tail: P(X >= k)
        tail = sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))
    else:
        tail = sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(0, k + 1))
    return min(2 * tail, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity-check the paraphrase templates.")
    parser.add_argument("--base-model", type=Path, default=Path("models/all-MiniLM-L6-v2"))
    parser.add_argument("--cuad", type=Path, default=Path("data/cuad/CUAD_v1/CUAD_v1.json"))
    parser.add_argument("--testset-docs", type=Path, default=Path("data/testset/docs.txt"))
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    set_threads(args.threads)
    seed_everything(2026)
    audit_templates(Path("data/testset/queries.jsonl"))

    excluded = {
        line.strip()
        for line in args.testset_docs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    contracts = load_contracts(args.cuad)
    held_out = [c for c in contracts if c["file"] not in excluded]

    # Clauses of each category in the held-out contracts.
    clauses_by_cat: dict[str, list[str]] = {c: [] for c in QUERY_TEMPLATES}
    for c in held_out:
        for cat, spans in c["spans"].items():
            if cat not in QUERY_TEMPLATES:
                continue
            for start, end in spans:
                text = clean(c["context"][start:end])
                if text:
                    clauses_by_cat[cat].append(text)

    total_clauses = sum(len(v) for v in clauses_by_cat.values())
    print(f"held-out contracts: {len(held_out)}")
    print(f"clauses per category (total {total_clauses}):")
    for cat in QUERY_TEMPLATES:
        print(f"  {cat:<34} {len(clauses_by_cat[cat])}")

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model))
    model = AutoModel.from_pretrained(str(args.base_model))
    model.eval()

    # Embed every clause once (shared across all templates).
    all_texts: list[str] = []
    cat_of_clause: list[str] = []
    for cat in QUERY_TEMPLATES:
        for text in clauses_by_cat[cat]:
            all_texts.append(text)
            cat_of_clause.append(cat)
    clause_emb = embed_batched(model, tokenizer, all_texts, args.max_len, device, args.batch)

    templates = [(cat, q) for cat, qs in QUERY_TEMPLATES.items() for q in qs]
    template_emb = embed_batched(
        model, tokenizer, [q for _, q in templates], args.max_len, device, args.batch
    )

    cat_index: dict[str, np.ndarray] = {
        cat: np.array([i for i, c in enumerate(cat_of_clause) if c == cat])
        for cat in QUERY_TEMPLATES
    }

    n_cats = len(QUERY_TEMPLATES)
    chance = 1.0 / n_cats

    print("\nper-category (mean over its 3 templates): margin = own-clause cosine "
          "minus other-clause cosine; rec@10 = share of top-10 clauses of its own category")
    print(f"{'category':<34} {'margin':>8} {'rec@10':>8} {'top-1 hits':>10}")
    margins = []
    recs = []
    top1_hits = 0
    n_templates = len(templates)
    for cat in QUERY_TEMPLATES:
        idx = cat_index[cat]
        others = np.ones(len(all_texts), dtype=bool)
        others[idx] = False
        cat_margin = []
        cat_rec = []
        for (c0, _), emb in zip(templates, template_emb):
            if c0 != cat:
                continue
            sim = (clause_emb @ emb).numpy()  # cosine to every clause
            margin = float(sim[idx].mean() - sim[others].mean())
            top10 = sim.argsort()[::-1][:10]
            rec = float(sum(1 for i in top10 if cat_of_clause[i] == cat) / len(top10))
            if sim.argmax() in idx:
                top1_hits += 1
            cat_margin.append(margin)
            cat_rec.append(rec)
            margins.append(margin)
            recs.append(rec)
        m = float(np.mean(cat_margin)) if cat_margin else float("nan")
        r = float(np.mean(cat_rec)) if cat_rec else float("nan")
        hits = sum(
            1 for (c0, _), emb in zip(templates, template_emb)
            if c0 == cat and int((clause_emb @ emb).argmax()) in idx
        )
        print(f"{cat:<34} {m:>8.4f} {r:>8.4f} {hits:>4}/3")

    mean_margin = float(np.mean(margins))
    positive = sum(1 for m in margins if m > 0)
    mean_rec = float(np.mean(recs))
    top1_rate = top1_hits / n_templates
    binom_p = binomial_two_sided_p(positive, n_templates, 0.5)

    print("\n--- summary ---")
    print(f"templates: {n_templates} ({n_cats} categories x 3)")
    print(f"mean margin: {mean_margin:.4f}")
    print(f"templates with positive margin: {positive}/{n_templates} "
          f"(sign-test p={binom_p:.2e})")
    print(f"mean recall@10 of own category: {mean_rec:.4f} (chance {chance:.3f})")
    print(f"top-1 hit rate: {top1_rate:.3f} (chance {chance:.3f}, "
          f"{top1_rate / chance:.1f}x chance)")

    ok = mean_margin > 0.01 and positive >= n_templates * 0.8 and top1_rate > chance * 3
    if ok:
        print("PASS: templates are meaningful, load-bearing queries for their categories.")
        return 0
    print("FAIL: one or more templates are not clearly tied to their category; "
          "rewrite them before training.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
