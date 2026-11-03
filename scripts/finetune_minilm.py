#!/usr/bin/env python3
"""Fine-tune all-MiniLM-L6-v2 for QorFinder semantic search.

QorFinder's semantic search is within 0.05 nDCG@10 of BM25 on the wording-
matched queries only if the MiniLM embeddings place the matched query close
to the clause passages that express it. This script teaches MiniLM exactly
that with a contrastive objective: every training example pairs a short,
paraphrased query with the clause span it should retrieve, and
MultipleNegativesRankingLoss pushes the query towards its own clause and
away from every other clause in the batch (the in-batch negatives).

LEAK PREVENTION (these rules are enforced here, not just documented):

  1. The 30 evaluation contracts are excluded. Only the other ~480 CUAD
     contracts provide training and validation data. The excluded files are
     read from data/testset/docs.txt and never touched.
  2. The 34 frozen test queries are never used. Training queries are built
     from scratch below, and the script refuses to train if any of them
     shares a 3-word phrase with any test query (matched or paraphrased).
  3. CUAD's category descriptions (and any near-verbatim form of them, i.e.
     the matched test queries) are NOT used as training queries. Training
     queries are hand-written paraphrases of each clause concept, written
     independently of the CUAD descriptions, and are paired with the
     held-out contracts' own clause text.
  4. The frozen test set is not shaped here: data/testset/queries.jsonl is
     read only for the phrase audit of rule 2, and its spans are never read.

DELIBERATE, EXPLAINABLE BIASES (everything not listed here is meant to be
neutral). The goal is a setup with no *unexplained* bias:

  1. Category focus. Training covers only the 17 CUAD clause categories the
     frozen test set uses. Those are the retrieval concepts being evaluated,
     so this is intentional. The model is not trained to be a general legal
     retriever, and no claim beyond these categories is made.
  2. Hand-written paraphrase queries. The 3 queries per category are written
     here by hand and audited (audit_templates) so no template shares a 3-word
     phrase with any of the 34 test queries; the only overlap left is one- or
     two-word concept terms ("non-compete", "intellectual property") that are
     the category's own vocabulary and cannot be reworded away. This stops
     either query group being inflated by near-verbatim wording while still
     teaching each concept.
  3. Register. Training queries are short declarative noun phrases, matching
     the register of the frozen queries (also declarative, not questions).
     Free-form question queries are out of scope and not expected to improve.
  4. Passage is the exact clause span, not the 200-token eval chunk. The span
     is the ground-truth relevant text; deployment mean-pools the chunk that
     contains it, which carries the same concept, so concept-to-clause
     learning transfers to chunk-level ranking.
  5. Uniform per-category batch sampling. One pair per category per batch, so
     rare categories get equal weight and no two same-category clauses are
     ever negatives (which would be false negatives). A deliberate rebalancing
     and a deliberate negative-labelling choice.
  6. Sequence length 128 (vs 256 at serve). A speed/consistency choice;
     clauses rarely exceed 128 tokens.

Run inside the reference devshell, which provides PyTorch and transformers:

    nix develop .#reference --command python3 scripts/finetune_minilm.py \
        --out-dir results/finetune-minilm

The base model is loaded from models/all-MiniLM-L6-v2 (offline). Nothing is
downloaded. The saved model is the bare BERT transformer (config.json,
tokenizer.json, model.safetensors): the Rust pipeline already mean-pools and
unit-normalizes BERT output itself, exactly as it does for the base MiniLM,
so no sentence-transformers pooling module is written.

Hibernation and reboot safety: the process is paused and resumed by the OS
without any wall-clock dependence; the log is flushed after every step and
every checkpoint is written to a temporary directory and atomically renamed,
so an interruption (including power loss) leaves the previous checkpoint
intact. On restart the script resumes from the newest checkpoint.

Usage summary (see --help for every option):

    python3 scripts/finetune_minilm.py [--out-dir DIR] [--epochs N] ...
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Training data: paraphrase templates, written here in my own words. They
# deliberately avoid the CUAD category descriptions and the matched test
# queries. Each is a short natural query a user might type; each clause span
# of that category in a held-out contract becomes its positive passage.
# Audited by audit_templates: no template shares a 3-word phrase with any test
# query (see the docstring's "biases" note).
# ---------------------------------------------------------------------------

QUERY_TEMPLATES: dict[str, list[str]] = {
    "Affiliate License-Licensee": [
        "the grant of rights covers the recipient's parent and sister companies as well",
        "related companies of the licensee may also exercise the license",
        "the license extends to subsidiaries and parent entities of the grantee",
    ],
    "Competitive Restriction Exception": [
        "situations that fall outside the ban on competing or chasing clients",
        "cases in which a party remains free to compete or pursue its partner's customers",
        "circumstances where the restrictive covenants do not apply",
    ],
    "Covenant Not To Sue": [
        "a party agrees never to attack the licensor's patents in court",
        "agreement to refrain from challenging the other party's IP or suing over it",
        "promise to refrain from litigation against the licensor on unrelated matters",
    ],
    "Irrevocable Or Perpetual License": [
        "a license that cannot be taken back and lasts without end",
        "rights granted on a permanent basis that the licensor may not revoke",
        "an everlasting, non-revocable grant of rights",
    ],
    "Joint Ip Ownership": [
        "co-owned inventions resulting from the joint development work",
        "intellectual property developed jointly is owned by both parties",
        "shared title to IP produced during the collaboration",
    ],
    "Liquidated Damages": [
        "a fixed amount to be paid if the deal is broken or ended early",
        "pre-agreed compensation for default or for terminating the contract",
        "the parties fix in advance how much a default will cost",
    ],
    "Most Favored Nation": [
        "if a rival customer gets better terms, this buyer gets them too",
        "the seller must pass on any better deal it gives to another party",
        "the buyer is entitled to the best terms offered to anyone else",
    ],
    "No-Solicit Of Customers": [
        "a party is forbidden from soliciting business from its partner's clientele",
        "ban on courting or signing up customers of its partner",
        "a party is barred from selling to existing accounts of its partner",
    ],
    "No-Solicit Of Employees": [
        "neither party will hire away personnel of its partner",
        "ban on poaching workers or freelancers from its partner",
        "no recruiting of the partner's employees during the term",
    ],
    "Non-Compete": [
        "a party is barred from running a competing business within a defined region",
        "ban on operating a rival enterprise within a given field or territory",
        "must not compete with its partner in a certain area or sector",
    ],
    "Non-Disparagement": [
        "neither party may speak ill of its partner",
        "obligation not to criticize or harm the partner's public image",
        "ban on making negative statements about the counterparty",
    ],
    "Non-Transferable License": [
        "the license may not be sold, assigned or sublicensed by the licensee",
        "the licensee is prohibited from passing its rights on to another entity",
        "restriction on giving the granted rights to any outside entity",
    ],
    "Rofr/Rofo/Rofn": [
        "one side gets first chance to buy before outsiders",
        "a party has priority to purchase before any sale to an outside buyer",
        "the partner must be offered the deal first",
    ],
    "Third Party Beneficiary": [
        "a non-signatory is entitled to enforce provisions of the contract",
        "a non-contracting party can hold the signers to their obligations",
        "a person outside the deal may hold the signers to what they promised",
    ],
    "Uncapped Liability": [
        "a party's liability is not subject to any ceiling",
        "unlimited responsibility for default, infringement or disclosure of confidential data",
        "exposure for damages is unlimited in certain cases",
    ],
    "Volume Restriction": [
        "additional fees or approval kick in once usage passes a set level",
        "consent or a higher price is required beyond a set quantity",
        "limits or charges apply if use exceeds a certain amount",
    ],
    "Warranty Duration": [
        "the goods are guaranteed free of defects for a stated period",
        "the length of the seller's promise against faults or errors",
        "period during which the provider warrants its deliverables",
    ],
}


def trigrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def audit_templates(queries_jsonl: Path) -> None:
    """Exit if any template shares a 3-word phrase with any frozen test query
    (leak rule 2), so a copied or near-copied test query can never train."""
    queries = [
        json.loads(line)
        for line in queries_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clashes = [
        f"{q['id']}: {template!r} shares {sorted(shared)}"
        for templates in QUERY_TEMPLATES.values()
        for template in templates
        for q in queries
        if (shared := trigrams(template) & trigrams(q["text"]))
    ]
    if clashes:
        sys.exit("templates overlap the test queries:\n" + "\n".join(clashes))


def file_name(title: str) -> str:
    """A file name for a contract title, as prepare_cuad.py writes it."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(". ") + ".txt"


def load_contracts(cuad_json: Path) -> list[dict]:
    """Every CUAD contract as {title, file, context, spans: {category: [(s, e)]}}."""
    data = json.loads(cuad_json.read_text(encoding="utf-8"))["data"]
    contracts = []
    for doc in data:
        (paragraph,) = doc["paragraphs"]
        context = paragraph["context"]
        spans: dict[str, list[tuple[int, int]]] = {}
        for qa in paragraph["qas"]:
            category = qa["id"].split("__")[-1]
            spans[category] = []
            for a in qa["answers"]:
                start = a["answer_start"]
                end = start + len(a["text"])
                if context[start:end] != a["text"]:
                    sys.exit(
                        f"{doc['title']}: {category} span {start}..{end} "
                        "does not match its text"
                    )
                spans[category].append((start, end))
        contracts.append(
            {
                "title": doc["title"],
                "file": file_name(doc["title"]),
                "context": context,
                "spans": spans,
            }
        )
    return contracts


def clean(text: str) -> str:
    """Collapse all whitespace so a span reads as one line."""
    return " ".join(text.split())


def build_pairs(
    contracts: list[dict],
    max_pairs_per_category: int | None,
    seed: int,
) -> list[tuple[str, str, str]]:
    """(category, query, passage) triples from the given (held-out) contracts.

    Every annotated clause span of a templated category is paired with each
    of that category's paraphrase queries. The passage is the span text
    itself, which is exactly what CUAD judged relevant, so a positive pair is
    a faithful (query, relevant-passage) example. `max_pairs_per_category`
    caps the total per category (useful for smoke tests).
    """
    by_cat: dict[str, list[tuple[str, str, str]]] = {}
    for contract in contracts:
        for category, spans in contract["spans"].items():
            if category not in QUERY_TEMPLATES:
                continue
            for start, end in spans:
                passage = clean(contract["context"][start:end])
                if not passage:
                    continue
                for query in QUERY_TEMPLATES[category]:
                    by_cat.setdefault(category, []).append((category, query, passage))
    pairs: list[tuple[str, str, str]] = []
    for category in sorted(by_cat):
        items = by_cat[category]
        if max_pairs_per_category is not None:
            items = items[:max_pairs_per_category]
        pairs.extend(items)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs


def split_contracts(
    contracts: list[dict], val_share: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Hold out a fixed share of *contracts* for validation, so no validation
    passage comes from a training contract (the model cannot memorise the
    exact clause and trivially match its paraphrase)."""
    ordered = sorted(contracts, key=lambda c: c["file"])
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = max(1, round(len(ordered) * val_share))
    return ordered[n_val:], ordered[:n_val]


# ---------------------------------------------------------------------------
# Model / encoding / loss
# ---------------------------------------------------------------------------

def mean_pool_embeddings(
    model, tokenizer, texts: list[str], max_len: int, device
) -> "torch.Tensor":
    """Mean-pool (over real tokens) then unit-normalise, as the Rust pipeline
    does, so training geometry matches the deployed cosine space. The forward
    pass is left differentiable so gradients flow to the model during
    training; callers that want inference only wrap the call in no_grad."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return pooled


def multiple_negatives_loss(anchors, positives, temperature: float) -> "torch.Tensor":
    """In-batch negatives: anchor i matches positive i and no other positive.

    `anchors` and `positives` are already unit-length, so their dot product is
    cosine similarity. Scaling by 1/temperature (temperature defaults to 0.05,
    i.e. sentence-transformers' scale 20) sharpens the softmax.
    """
    import torch.nn.functional as F

    logits = (anchors @ positives.transpose(0, 1)) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Batching: one pair per category per batch, so in-batch negatives are always
# other categories and a query is never paired against another clause of its
# own category (which would be a false negative).
# ---------------------------------------------------------------------------

def build_epoch_batches(
    pairs: list[tuple[str, str, str]],
    batch_size: int,
    steps_per_epoch: int,
    seed: int,
) -> list[list[int]]:
    by_cat: dict[str, list[int]] = {}
    for i, (category, _q, _p) in enumerate(pairs):
        by_cat.setdefault(category, []).append(i)
    categories = sorted(by_cat)
    if batch_size > len(categories):
        sys.exit(
            f"batch size {batch_size} exceeds the {len(categories)} template "
            "categories; a batch would contain two pairs of one category"
        )
    rng = random.Random(seed)
    batches: list[list[int]] = []
    for _ in range(steps_per_epoch):
        rng.shuffle(categories)
        batch = [rng.choice(by_cat[c]) for c in categories[:batch_size]]
        batches.append(batch)
    return batches


def build_flat_batches(
    pairs: list[tuple[str, str, str]], batch_size: int, seed: int
) -> list[list[int]]:
    """Shuffle and chunk into contiguous batches of `batch_size` (the last
    may be short). Used for validation, where a category may have no pair."""
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def encode_pair_batch(
    model, tokenizer, pairs, indices, max_len: int, device
) -> tuple["torch.Tensor", "torch.Tensor"]:
    anchors = [pairs[i][1] for i in indices]
    passages = [pairs[i][2] for i in indices]
    a = mean_pool_embeddings(model, tokenizer, anchors, max_len, device)
    p = mean_pool_embeddings(model, tokenizer, passages, max_len, device)
    return a, p


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Checkpoints (atomic) and resume
# ---------------------------------------------------------------------------

CHECKPOINT_PREFIX = "step_"


def checkpoint_steps(checkpoint_dir: Path) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []
    steps = []
    for name in checkpoint_dir.iterdir():
        if name.is_dir() and name.name.startswith(CHECKPOINT_PREFIX):
            try:
                steps.append(int(name.name[len(CHECKPOINT_PREFIX):]))
            except ValueError:
                continue
    return sorted(steps)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model,
    tokenizer,
    state: dict,
    logger: Logger,
) -> Path:
    """Write model.safetensors, config.json, tokenizer.json and trainer state
    into step_NNNN atomically (temp dir, then rename)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoint_dir / f"{CHECKPOINT_PREFIX}{step:07d}"
    tmp = checkpoint_dir / f".tmp_{step:07d}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    torch.save(state, tmp / "trainer.pt")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    logger.write(
        f"checkpoint saved step={step} -> {final} "
        f"(model.safetensors sha256={sha256(final / 'model.safetensors')[:16]}...)"
    )
    return final


def load_latest_checkpoint(
    checkpoint_dir: Path, logger: Logger
) -> tuple[int, dict, Path] | None:
    steps = checkpoint_steps(checkpoint_dir)
    if not steps:
        return None
    latest = checkpoint_dir / f"{CHECKPOINT_PREFIX}{steps[-1]:07d}"
    state_path = latest / "trainer.pt"
    if not state_path.is_file():
        logger.write(f"warning: {latest} has no trainer.pt; ignoring it")
        return None
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    logger.write(f"resumed from {latest} (global step {state['global_step']})")
    return steps[-1], state, latest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune all-MiniLM-L6-v2 for QorFinder (contrastive, in-batch negatives)."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("models/all-MiniLM-L6-v2"),
        help="folder holding the base MiniLM files (config.json, tokenizer.json, model.safetensors)",
    )
    parser.add_argument(
        "--cuad",
        type=Path,
        default=Path("data/cuad/CUAD_v1/CUAD_v1.json"),
    )
    parser.add_argument(
        "--testset-docs",
        type=Path,
        default=Path("data/testset/docs.txt"),
        help="the 30 evaluation contracts to EXCLUDE (leak rule 1)",
    )
    parser.add_argument(
        "--testset-queries",
        type=Path,
        default=Path("data/testset/queries.jsonl"),
        help="the 34 test queries no template may share a 3-word phrase with (leak rule 2)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/finetune-minilm"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-pairs-per-category", type=int, default=None)
    parser.add_argument("--val-share", type=float, default=0.1)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--steps-per-checkpoint",
        type=int,
        default=None,
        help="checkpoint every N steps (default: once per epoch)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many global steps (for smoke tests)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="also copy the lowest-validation-loss checkpoint to out-dir/model-best",
    )
    return parser.parse_args(argv)


def set_threads(n: int) -> None:
    n = max(1, n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_threads(args.threads)
    seed_everything(args.seed)

    from transformers import AutoModel, AutoTokenizer

    out_dir: Path = args.out_dir
    checkpoint_dir = out_dir / "checkpoints"
    log_path = out_dir / "train.log"
    logger = Logger(log_path)
    logger.write("--- finetune_minilm.py started ---")
    logger.write(
        f"base-model={args.base_model} cuad={args.cuad} testset-docs={args.testset_docs} "
        f"out-dir={out_dir} seed={args.seed} threads={args.threads}"
    )

    # --- Data (leak rule 2: no template echoes a test query) ---------------
    audit_templates(args.testset_queries)
    logger.write(f"templates share no 3-word phrase with {args.testset_queries}")

    # --- Data (leak rule 1: exclude the 30 evaluation contracts) ----------
    excluded = {
        line.strip()
        for line in args.testset_docs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    contracts = load_contracts(args.cuad)
    held_out = [c for c in contracts if c["file"] not in excluded]
    if len(held_out) + len(excluded) != len(contracts):
        sys.exit("held-out + excluded does not cover all contracts")
    logger.write(
        f"excluded {len(excluded)} evaluation contracts; {len(held_out)} held-out contracts"
    )
    train_contracts, val_contracts = split_contracts(
        held_out, args.val_share, args.seed
    )
    logger.write(
        f"train/val split by contract: {len(train_contracts)} train / "
        f"{len(val_contracts)} val (share {args.val_share})"
    )
    train_pairs = build_pairs(train_contracts, args.max_pairs_per_category, args.seed)
    val_pairs = build_pairs(val_contracts, args.max_pairs_per_category, args.seed)
    logger.write(
        f"training pairs={len(train_pairs)}  validation pairs={len(val_pairs)} "
        f"({len(QUERY_TEMPLATES)} categories, {len(QUERY_TEMPLATES) * 3} paraphrase queries)"
    )

    steps_per_epoch = args.steps_per_epoch or math.ceil(len(train_pairs) / args.batch_size)
    steps_per_checkpoint = args.steps_per_checkpoint or steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    logger.write(
        f"steps_per_epoch={steps_per_epoch} steps_per_checkpoint={steps_per_checkpoint} "
        f"epochs={args.epochs} total_steps={total_steps} batch_size={args.batch_size}"
    )

    # --- Model and optimizer ----------------------------------------------
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(str(args.base_model))
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_step = 0
    best_val_loss = float("inf")
    best_step: int | None = None
    if (latest := load_latest_checkpoint(checkpoint_dir, logger)) is not None:
        start_step, state, _ckpt_path = latest
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_val_loss = state.get("best_val_loss", float("inf"))
        best_step = state.get("best_step")
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        # Warn if the caller changed a data-affecting setting.
        for key in ("seed", "val_share", "max_pairs_per_category", "batch_size"):
            if state.get("args", {}).get(key) is not None and getattr(args, key) != state["args"][key]:
                logger.write(
                    f"warning: --{key} changed since checkpoint "
                    f"({state['args'][key]} -> {getattr(args, key)}); resume may misalign"
                )
    else:
        logger.write("no checkpoint found; training from scratch")

    # --- Training loop ------------------------------------------------------
    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / args.warmup_steps
        return args.learning_rate

    epoch_anchor = start_step // steps_per_epoch
    logger.write(f"training begins at global step {start_step} (epoch {epoch_anchor})")
    wall_start = time.time()
    last_train_loss = float("nan")
    val_loss = float("nan")

    global_step = start_step
    while global_step < total_steps:
        epoch = global_step // steps_per_epoch
        batches = build_epoch_batches(
            train_pairs, args.batch_size, steps_per_epoch, args.seed + 1_000_000 + epoch
        )
        in_epoch = global_step % steps_per_epoch
        for batch in batches[in_epoch:]:
            if global_step >= total_steps:
                break
            for g in optimizer.param_groups:
                g["lr"] = lr_at(global_step - start_step)
            anchors, positives = encode_pair_batch(
                model, tokenizer, train_pairs, batch, args.max_seq_len, device
            )
            loss = multiple_negatives_loss(anchors, positives, args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            last_train_loss = loss.item()
            elapsed = time.time() - wall_start
            logger.write(
                f"step={global_step} epoch={epoch} loss={last_train_loss:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} wall={elapsed:.1f}s"
            )
            if global_step % steps_per_checkpoint == 0 or global_step >= total_steps:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "rng_python": random.getstate(),
                    "rng_numpy": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(),
                    "args": vars(args),
                }
                save_checkpoint(checkpoint_dir, global_step, model, tokenizer, state, logger)

        # Validation once per epoch (fixed, reproducible val batches).
        val_batches = build_flat_batches(val_pairs, args.batch_size, args.seed + 9_000_000)
        model.eval()
        total = 0.0
        correct = 0
        count = 0
        with torch.no_grad():
            for batch in val_batches:
                a, p = encode_pair_batch(
                    model, tokenizer, val_pairs, batch, args.max_seq_len, device
                )
                loss = multiple_negatives_loss(a, p, args.temperature)
                total += loss.item() * len(batch)
                # in-batch accuracy@1: is each anchor's nearest neighbour its own positive?
                sim = a @ p.transpose(0, 1)
                pred = sim.argmax(dim=1)
                correct += (pred == torch.arange(len(batch), device=device)).sum().item()
                count += len(batch)
        model.train()
        val_loss = total / count if count else float("nan")
        val_acc = correct / count if count else float("nan")
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_step = global_step
        logger.write(
            f"epoch={epoch} done  val_loss={val_loss:.6f} val_acc@1={val_acc:.4f} "
            f"best_val_loss={best_val_loss:.6f}{' (new best)' if is_best else ''}"
        )

    # --- Final model and summary ------------------------------------------
    final_dir = out_dir / "model"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    final_safetensors = final_dir / "model.safetensors"

    best_dir: Path | None = None
    if args.save_best:
        src = None
        if best_step is not None:
            candidate = checkpoint_dir / f"{CHECKPOINT_PREFIX}{best_step:07d}"
            src = candidate if (candidate / "model.safetensors").is_file() else None
        if src is None:
            steps = checkpoint_steps(checkpoint_dir)
            if steps:
                src = checkpoint_dir / f"{CHECKPOINT_PREFIX}{steps[-1]:07d}"
        if src is not None:
            best_dir = out_dir / "model-best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            best_dir.mkdir(parents=True)
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                shutil.copyfile(src / name, best_dir / name)

    total_wall = time.time() - wall_start
    summary = [
        "--- training summary ---",
        f"seed={args.seed}",
        f"train/val split={len(train_contracts)}/{len(val_contracts)} contracts "
        f"({args.val_share:.0%} val), pairs train={len(train_pairs)} val={len(val_pairs)}",
        f"epochs requested={args.epochs}, steps run={global_step} "
        f"(steps_per_epoch={steps_per_epoch})",
        f"batch_size={args.batch_size}",
        f"learning_rate={args.learning_rate} (warmup={args.warmup_steps})",
        f"temperature={args.temperature}",
        f"max_seq_len={args.max_seq_len}, threads={args.threads}",
        f"final train loss={last_train_loss:.6f}",
        f"final val loss={val_loss:.6f}",
        f"best val loss={best_val_loss:.6f}",
        f"wall_time={total_wall:.1f}s ({total_wall / 60:.1f} min)",
        f"saved safetensors={final_safetensors} (sha256={sha256(final_safetensors)})",
    ]
    if best_dir is not None:
        summary.append(
            f"best checkpoint copy={best_dir / 'model.safetensors'} "
            f"(sha256={sha256(best_dir / 'model.safetensors')})"
        )
    for line in summary:
        logger.write(line)
    logger.write("--- finetune_minilm.py finished ---")
    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        print("\ninterrupted; the latest checkpoint is resumable", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
