#!/usr/bin/env python3
"""Select the fine-tuning learning rate on the validation split (a small grid).

The fine-tuning script's default learning rate is just a sensible prior. To
make the final choice *evidence* rather than habit, this script trains a few
learning rates on a reduced copy of the training split and scores each on the
held-out validation split (which is split **by contract**, and shares no
contract with the training split or the frozen 34-query test set). The lowest
validation loss wins.

It is deliberately small and CPU-only: a few learning rates, a reduced number
of pairs per category, a couple of epochs. Epoch count, batch size (16),
temperature (0.05, i.e. sentence-transformers' MNRL scale 20) and sequence
length (128) are fixed at standard values and are not swept here; the number of
epochs for the final run is chosen by validation loss during that run
(`--save-best` + the per-epoch `val_loss` lines in the log).

Nothing is downloaded; only held-out contracts are used.

Usage (reference devshell):

    nix develop .#reference --command python3 scripts/tune_hyperparams.py

Writes `results/finetune-minilm/hyperparams.tsv` and prints a table with the
recommended learning rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from finetune_minilm import (
    audit_templates,
    build_epoch_batches,
    build_flat_batches,
    build_pairs,
    load_contracts,
    mean_pool_embeddings,
    multiple_negatives_loss,
    seed_everything,
    set_threads,
    split_contracts,
)


def encode(model, tokenizer, pairs, indices, max_len, device):
    a = mean_pool_embeddings(model, tokenizer, [pairs[i][1] for i in indices], max_len, device)
    p = mean_pool_embeddings(model, tokenizer, [pairs[i][2] for i in indices], max_len, device)
    return a, p


def val_metrics(model, tokenizer, val_pairs, batch, temp, max_len, device, seed):
    model.eval()
    total = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch_ids in build_flat_batches(val_pairs, batch, seed + 9_000_000):
            a, p = encode(model, tokenizer, val_pairs, batch_ids, max_len, device)
            total += multiple_negatives_loss(a, p, temp).item() * len(batch_ids)
            sim = a @ p.transpose(0, 1)
            pred = sim.argmax(dim=1)
            correct += (pred == torch.arange(len(batch_ids), device=device)).sum().item()
            count += len(batch_ids)
    model.train()
    return total / count if count else float("nan"), correct / count if count else float("nan")


def train_once(lr, base_model, train_pairs, val_pairs, epochs, batch, temp, max_len, device, seed):
    tokenizer = AutoTokenizer.from_pretrained(str(base_model))
    model = AutoModel.from_pretrained(str(base_model))
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = max(1, (len(train_pairs) + batch - 1) // batch)
    for epoch in range(epochs):
        batches = build_epoch_batches(train_pairs, batch, steps_per_epoch, seed + 1_000_000 + epoch)
        for batch_ids in batches:
            a, p = encode(model, tokenizer, train_pairs, batch_ids, max_len, device)
            loss = multiple_negatives_loss(a, p, temp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return val_metrics(model, tokenizer, val_pairs, batch, temp, max_len, device, seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the learning rate on the validation split.")
    parser.add_argument("--base-model", type=Path, default=Path("models/all-MiniLM-L6-v2"))
    parser.add_argument("--cuad", type=Path, default=Path("data/cuad/CUAD_v1/CUAD_v1.json"))
    parser.add_argument("--testset-docs", type=Path, default=Path("data/testset/docs.txt"))
    parser.add_argument("--out", type=Path, default=Path("results/finetune-minilm/hyperparams.tsv"))
    parser.add_argument("--lrs", type=str, default="1e-5,2e-5,5e-5")
    parser.add_argument("--tune-epochs", type=int, default=3)
    parser.add_argument("--max-pairs-per-category", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    set_threads(args.threads)
    seed_everything(args.seed)
    audit_templates(Path("data/testset/queries.jsonl"))

    excluded = {
        line.strip()
        for line in args.testset_docs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    contracts = load_contracts(args.cuad)
    held_out = [c for c in contracts if c["file"] not in excluded]
    train_contracts, val_contracts = split_contracts(held_out, 0.1, args.seed)
    train_pairs = build_pairs(train_contracts, args.max_pairs_per_category, args.seed)
    val_pairs = build_pairs(val_contracts, args.max_pairs_per_category, args.seed)
    print(
        f"train pairs={len(train_pairs)} val pairs={len(val_pairs)} "
        f"(reduced, max {args.max_pairs_per_category} per category); "
        f"{args.tune_epochs} epochs per learning rate"
    )

    device = torch.device("cpu")
    results = []
    for lr in [float(x) for x in args.lrs.split(",")]:
        val_loss, val_acc = train_once(
            lr, args.base_model, train_pairs, val_pairs, args.tune_epochs,
            args.batch_size, args.temperature, args.max_seq_len, device, args.seed,
        )
        results.append((lr, val_loss, val_acc))
        print(f"lr={lr:.1e}  val_loss={val_loss:.6f}  val_acc@1={val_acc:.4f}")

    results.sort(key=lambda r: r[1])
    best_lr, best_loss, best_acc = results[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("lr\tval_loss\tval_acc@1\n")
        for lr, loss, acc in results:
            f.write(f"{lr:.6e}\t{loss:.6f}\t{acc:.4f}\n")
    print(f"\nrecommended: --learning-rate {best_lr:.1e} (val_loss={best_loss:.6f}, "
          f"val_acc@1={best_acc:.4f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
