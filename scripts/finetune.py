#!/usr/bin/env python3
"""Fine-tune all-MiniLM-L6-v2 for QorFinder semantic search.

QorFinder's semantic search is within 0.05 nDCG@10 of BM25 on the wording-
matched queries only if the MiniLM embeddings place the matched query close
to the clause passages that express it. This script teaches MiniLM exactly
that with a contrastive objective: every training example pairs a short,
paraphrased query with the clause span it should retrieve, and
MultipleNegativesRankingLoss pushes the query towards its own clause and
away from every other clause in the batch (the in-batch negatives).

It runs the whole pipeline, one section below per step, in this order:

  1. check the templates: under the base model, each paraphrase query is
     closer to its own category's clauses than to the others';
  2. tune the learning rate on the validation split;
  3. train at the best rate, resuming from the newest checkpoint;
  4. copy the epoch with the lowest validation loss to
     models/all-MiniLM-L6-v2-ft-cuad/;
  5. write its reference embeddings for the Rust test.

Usage (the reference shell provides PyTorch and transformers):

    nix develop .#reference --command python3 scripts/finetune.py

The base model is loaded from models/all-MiniLM-L6-v2 (offline); nothing is
downloaded. The log, the tuning table and the checkpoints go to
data/finetune/. The saved model is the bare BERT transformer (config.json,
tokenizer.json, model.safetensors): the Rust pipeline already mean-pools and
unit-normalizes BERT output itself, exactly as it does for the base MiniLM,
so no sentence-transformers pooling module is written. CPU training on
several threads is not bit-for-bit deterministic, so a rerun matches the
committed fixture within the test's cosine tolerance, not exactly.

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

Hibernation and reboot safety: the process is paused and resumed by the OS
without any wall-clock dependence; the log is flushed after every line and
every checkpoint is written to a temporary directory and atomically renamed,
so an interruption (including power loss) leaves the previous checkpoint
intact. On restart the script resumes from the newest checkpoint, and the
learning rates already tuned are not tuned again.
"""

from __future__ import annotations

import datetime
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

import reference_embeddings
from reference_embeddings import mean_pool_embeddings, sha256

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SEED = 2026
THREADS = os.cpu_count() or 1
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
# sentence-transformers' MultipleNegativesRankingLoss scale 20.
TEMPERATURE = 0.05
# Share of the held-out contracts kept for validation.
VAL_SHARE = 0.1

# Step 1: pass if the mean margin, the share of templates with a positive
# margin and the top-1 hit rate (as a multiple of chance) reach these.
CHECK_MIN_MARGIN = 0.01
CHECK_MIN_POSITIVE = 0.8
CHECK_MIN_TOP1_X_CHANCE = 3
CHECK_BATCH = 32

# Step 2: the grid, trained on a reduced copy of the training split.
TUNE_LEARNING_RATES = [1e-5, 2e-5, 5e-5]
TUNE_EPOCHS = 3
TUNE_MAX_PAIRS_PER_CATEGORY = 100

# Step 3.
EPOCHS = 3

REPO = Path(__file__).resolve().parent.parent
BASE_MODEL = REPO / "models/all-MiniLM-L6-v2"
FINE_TUNED_MODEL = REPO / "models" / reference_embeddings.FINE_TUNED
CUAD_JSON = REPO / "data/cuad/CUAD_v1/CUAD_v1.json"
# The 30 evaluation contracts to exclude (leak rule 1) and the 34 test
# queries no template may share a 3-word phrase with (leak rule 2).
TESTSET_DOCS = REPO / "data/testset/docs.txt"
TESTSET_QUERIES = REPO / "data/testset/queries.jsonl"
WORK = REPO / "data/finetune"
LOG = WORK / "finetune.log"
TUNING_TSV = WORK / "hyperparams.tsv"
CHECKPOINTS = WORK / "checkpoints"

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


def audit_templates() -> None:
    """Exit if any template shares a 3-word phrase with any frozen test query
    (leak rule 2), so a copied or near-copied test query can never train."""
    queries = [
        json.loads(line)
        for line in TESTSET_QUERIES.read_text(encoding="utf-8").splitlines()
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


def load_contracts() -> list[dict]:
    """Every CUAD contract as {title, file, context, spans: {category: [(s, e)]}}."""
    data = json.loads(CUAD_JSON.read_text(encoding="utf-8"))["data"]
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


def held_out_contracts() -> list[dict]:
    """The contracts outside the test sample (leak rule 1)."""
    excluded = {
        line.strip()
        for line in TESTSET_DOCS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    contracts = load_contracts()
    held_out = [c for c in contracts if c["file"] not in excluded]
    if len(held_out) + len(excluded) != len(contracts):
        sys.exit("held-out + excluded does not cover all contracts")
    return held_out


def clean(text: str) -> str:
    """Collapse all whitespace so a span reads as one line."""
    return " ".join(text.split())


def build_pairs(
    contracts: list[dict],
    max_pairs_per_category: int | None,
) -> list[tuple[str, str, str]]:
    """(category, query, passage) triples from the given (held-out) contracts.

    Every annotated clause span of a templated category is paired with each
    of that category's paraphrase queries. The passage is the span text
    itself, which is exactly what CUAD judged relevant, so a positive pair is
    a faithful (query, relevant-passage) example. `max_pairs_per_category`
    caps the total per category (the reduced copy that step 2 tunes on).
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
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    return pairs


def split_contracts(contracts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hold out a fixed share of *contracts* for validation, so no validation
    passage comes from a training contract (the model cannot memorise the
    exact clause and trivially match its paraphrase)."""
    ordered = sorted(contracts, key=lambda c: c["file"])
    rng = random.Random(SEED)
    rng.shuffle(ordered)
    n_val = max(1, round(len(ordered) * VAL_SHARE))
    return ordered[n_val:], ordered[:n_val]


# ---------------------------------------------------------------------------
# Loss and batching. One pair per category per batch, so in-batch negatives
# are always other categories and a query is never paired against another
# clause of its own category (which would be a false negative).
# ---------------------------------------------------------------------------

def multiple_negatives_loss(anchors, positives) -> "torch.Tensor":
    """In-batch negatives: anchor i matches positive i and no other positive.

    `anchors` and `positives` are already unit-length, so their dot product is
    cosine similarity. Scaling by 1/TEMPERATURE sharpens the softmax.
    """
    import torch.nn.functional as F

    logits = (anchors @ positives.transpose(0, 1)) / TEMPERATURE
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def build_epoch_batches(
    pairs: list[tuple[str, str, str]], steps: int, seed: int
) -> list[list[int]]:
    by_cat: dict[str, list[int]] = {}
    for i, (category, _q, _p) in enumerate(pairs):
        by_cat.setdefault(category, []).append(i)
    categories = sorted(by_cat)
    if BATCH_SIZE > len(categories):
        sys.exit(
            f"batch size {BATCH_SIZE} exceeds the {len(categories)} template "
            "categories; a batch would contain two pairs of one category"
        )
    rng = random.Random(seed)
    batches: list[list[int]] = []
    for _ in range(steps):
        rng.shuffle(categories)
        batches.append([rng.choice(by_cat[c]) for c in categories[:BATCH_SIZE]])
    return batches


def build_flat_batches(pairs: list[tuple[str, str, str]], seed: int) -> list[list[int]]:
    """Shuffle and chunk into contiguous batches (the last may be short).
    Used for validation, where a category may have no pair."""
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    return [indices[i : i + BATCH_SIZE] for i in range(0, len(indices), BATCH_SIZE)]


def encode_pair_batch(model, tokenizer, pairs, indices) -> tuple["torch.Tensor", "torch.Tensor"]:
    a = mean_pool_embeddings(model, tokenizer, [pairs[i][1] for i in indices], MAX_SEQ_LEN)
    p = mean_pool_embeddings(model, tokenizer, [pairs[i][2] for i in indices], MAX_SEQ_LEN)
    return a, p


def train_epoch(model, tokenizer, optimizer, pairs, steps: int, epoch: int, on_step=None) -> float:
    """One epoch of `steps` batches; returns the last batch's loss."""
    loss_value = float("nan")
    for step, batch in enumerate(build_epoch_batches(pairs, steps, SEED + 1_000_000 + epoch)):
        anchors, positives = encode_pair_batch(model, tokenizer, pairs, batch)
        loss = multiple_negatives_loss(anchors, positives)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if on_step is not None:
            on_step(step, loss_value)
    return loss_value


def validate(model, tokenizer, pairs) -> tuple[float, float]:
    """Mean loss and in-batch accuracy@1 (is each anchor's nearest neighbour
    its own positive?) on fixed, reproducible validation batches."""
    model.eval()
    total = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch in build_flat_batches(pairs, SEED + 9_000_000):
            a, p = encode_pair_batch(model, tokenizer, pairs, batch)
            total += multiple_negatives_loss(a, p).item() * len(batch)
            pred = (a @ p.transpose(0, 1)).argmax(dim=1)
            correct += (pred == torch.arange(len(batch))).sum().item()
            count += len(batch)
    model.train()
    return (total / count, correct / count) if count else (float("nan"), float("nan"))


def load_base_model():
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    model = AutoModel.from_pretrained(str(BASE_MODEL))
    return model, tokenizer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(message: str = "") -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    print(line, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Step 1: check the templates under the base model
# ---------------------------------------------------------------------------

def check_templates(held_out: list[dict]) -> None:
    """Exit unless each template is a genuine, load-bearing query for its
    category: under the base (un-fine-tuned) MiniLM, its own category's
    clauses should be closer than other categories' clauses, and its nearest
    clause should belong to its own category far more often than chance
    (1/17 ~ 6%). A template that fails is a bad paraphrase and must be
    rewritten before it poisons the training data."""
    # Grouped by category, as the recorded check embedded them.
    clauses: list[str] = []
    cat_of_clause: list[str] = []
    for cat in QUERY_TEMPLATES:
        for c in held_out:
            for start, end in c["spans"].get(cat, []):
                if text := clean(c["context"][start:end]):
                    clauses.append(text)
                    cat_of_clause.append(cat)
    log(f"step 1: {len(clauses)} clauses of {len(QUERY_TEMPLATES)} categories")

    model, tokenizer = load_base_model()
    model.eval()

    def embed(texts: list[str]):
        with torch.no_grad():
            return torch.cat([
                mean_pool_embeddings(model, tokenizer, texts[i : i + CHECK_BATCH], MAX_SEQ_LEN)
                for i in range(0, len(texts), CHECK_BATCH)
            ])

    clause_emb = embed(clauses)
    cat_of_clause_arr = np.array(cat_of_clause)
    # All templates in one call: batching them differently pads them
    # differently, which can flip a near-tie top-1 hit.
    template_emb = iter(embed([q for qs in QUERY_TEMPLATES.values() for q in qs]))
    chance = 1.0 / len(QUERY_TEMPLATES)

    log(f"{'category':<34} {'margin':>8} {'rec@10':>8} {'top-1 hits':>10}")
    margins = []
    recs = []
    top1_hits = 0
    for cat, templates in QUERY_TEMPLATES.items():
        own = cat_of_clause_arr == cat
        cat_margins = []
        cat_recs = []
        hits = 0
        for emb in (next(template_emb) for _ in templates):
            sim = (clause_emb @ emb).numpy()  # cosine to every clause
            cat_margins.append(float(sim[own].mean() - sim[~own].mean()))
            top10 = sim.argsort()[::-1][:10]
            cat_recs.append(float(own[top10].mean()))
            hits += bool(own[sim.argmax()])
        margins.extend(cat_margins)
        recs.extend(cat_recs)
        top1_hits += hits
        log(
            f"{cat:<34} {np.mean(cat_margins):>8.4f} {np.mean(cat_recs):>8.4f} "
            f"{hits:>4}/{len(templates)}"
        )

    n = len(margins)
    mean_margin = float(np.mean(margins))
    positive = sum(m > 0 for m in margins)
    top1_rate = top1_hits / n
    log(f"mean margin {mean_margin:.4f}; positive margin {positive}/{n}")
    log(f"mean recall@10 of own category {np.mean(recs):.4f} (chance {chance:.3f})")
    log(f"top-1 hit rate {top1_rate:.3f} (chance {chance:.3f}, {top1_rate / chance:.1f}x chance)")
    if (
        mean_margin <= CHECK_MIN_MARGIN
        or positive < n * CHECK_MIN_POSITIVE
        or top1_rate <= chance * CHECK_MIN_TOP1_X_CHANCE
    ):
        sys.exit("templates are not clearly tied to their categories; rewrite them before training")
    log("step 1: templates pass")


# ---------------------------------------------------------------------------
# Step 2: tune the learning rate on the validation split
# ---------------------------------------------------------------------------

def tune_learning_rate(train_contracts: list[dict], val_contracts: list[dict]) -> float:
    """Train each rate in TUNE_LEARNING_RATES on a reduced copy of the
    training split and return the one with the lowest validation loss.
    Epochs, batch size, temperature and sequence length stay at the
    standard values above and are not swept. Each rate's result is appended
    to TUNING_TSV as it finishes, so a rerun only trains the missing ones."""
    train_pairs = build_pairs(train_contracts, TUNE_MAX_PAIRS_PER_CATEGORY)
    val_pairs = build_pairs(val_contracts, TUNE_MAX_PAIRS_PER_CATEGORY)
    log(
        f"step 2: train pairs={len(train_pairs)} val pairs={len(val_pairs)} "
        f"(max {TUNE_MAX_PAIRS_PER_CATEGORY} per category), {TUNE_EPOCHS} epochs per rate"
    )
    done: dict[float, tuple[float, float]] = {}
    if TUNING_TSV.exists():
        for line in TUNING_TSV.read_text(encoding="utf-8").splitlines()[1:]:
            lr, loss, acc = line.split("\t")
            done[float(lr)] = (float(loss), float(acc))
    else:
        TUNING_TSV.write_text("lr\tval_loss\tval_acc@1\n", encoding="utf-8")

    steps = max(1, math.ceil(len(train_pairs) / BATCH_SIZE))
    for lr in TUNE_LEARNING_RATES:
        if lr not in done:
            seed_everything()
            model, tokenizer = load_base_model()
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            for epoch in range(TUNE_EPOCHS):
                train_epoch(model, tokenizer, optimizer, train_pairs, steps, epoch)
            done[lr] = validate(model, tokenizer, val_pairs)
            with TUNING_TSV.open("a", encoding="utf-8") as f:
                f.write(f"{lr:.6e}\t{done[lr][0]:.6f}\t{done[lr][1]:.4f}\n")
        log(f"lr={lr:.1e}  val_loss={done[lr][0]:.6f}  val_acc@1={done[lr][1]:.4f}")
    best = min(TUNE_LEARNING_RATES, key=lambda lr: done[lr][0])
    log(f"step 2: best learning rate {best:.1e}")
    return best


# ---------------------------------------------------------------------------
# Step 3: train at the best rate, checkpointing every epoch
# ---------------------------------------------------------------------------

CHECKPOINT_PREFIX = "epoch_"


def checkpoint_path(epoch: int) -> Path:
    return CHECKPOINTS / f"{CHECKPOINT_PREFIX}{epoch}"


def latest_checkpoint() -> Path | None:
    epochs = sorted(
        int(p.name[len(CHECKPOINT_PREFIX):])
        for p in CHECKPOINTS.glob(f"{CHECKPOINT_PREFIX}*")
        if (p / "trainer.pt").is_file()
    )
    return checkpoint_path(epochs[-1]) if epochs else None


def save_checkpoint(epoch: int, model, tokenizer, state: dict) -> None:
    """Write model.safetensors, config.json, tokenizer.json and the trainer
    state into epoch_N atomically (temp dir, then rename)."""
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    final = checkpoint_path(epoch)
    tmp = CHECKPOINTS / f".tmp_{epoch}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    torch.save(state, tmp / "trainer.pt")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    log(
        f"checkpoint saved epoch={epoch} -> {final} "
        f"(model.safetensors sha256={sha256(final / 'model.safetensors')[:16]}...)"
    )


def train(learning_rate: float, train_contracts: list[dict], val_contracts: list[dict]) -> Path:
    """Train EPOCHS epochs on the full training split, validating and
    checkpointing after each, and return the checkpoint with the lowest
    validation loss."""
    train_pairs = build_pairs(train_contracts, None)
    val_pairs = build_pairs(val_contracts, None)
    steps = math.ceil(len(train_pairs) / BATCH_SIZE)
    log(
        f"step 3: training pairs={len(train_pairs)} validation pairs={len(val_pairs)} "
        f"({len(QUERY_TEMPLATES)} categories, {len(QUERY_TEMPLATES) * 3} paraphrase queries)"
    )
    log(
        f"epochs={EPOCHS} steps_per_epoch={steps} batch_size={BATCH_SIZE} "
        f"learning_rate={learning_rate} temperature={TEMPERATURE} max_seq_len={MAX_SEQ_LEN} "
        f"threads={THREADS} seed={SEED}"
    )

    seed_everything()
    model, tokenizer = load_base_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    val_losses: list[float] = []
    if (latest := latest_checkpoint()) is not None:
        state = torch.load(latest / "trainer.pt", map_location="cpu", weights_only=False)
        if state["learning_rate"] != learning_rate:
            sys.exit(
                f"{latest} was trained at learning rate {state['learning_rate']}, not "
                f"{learning_rate}; delete {CHECKPOINTS} to train again"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        val_losses = state["val_losses"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        log(f"resumed from {latest} ({len(val_losses)} of {EPOCHS} epochs done)")
    else:
        log("no checkpoint found; training from scratch")

    wall_start = time.time()

    def on_step(step: int, loss: float) -> None:
        log(f"epoch={epoch} step={step + 1}/{steps} loss={loss:.6f} wall={time.time() - wall_start:.1f}s")

    for epoch in range(len(val_losses), EPOCHS):
        train_epoch(model, tokenizer, optimizer, train_pairs, steps, epoch, on_step)
        val_loss, val_acc = validate(model, tokenizer, val_pairs)
        val_losses.append(val_loss)
        log(f"epoch={epoch} done  val_loss={val_loss:.6f} val_acc@1={val_acc:.4f}")
        save_checkpoint(epoch, model, tokenizer, {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "val_losses": val_losses,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
        })

    best = min(range(EPOCHS), key=lambda e: val_losses[e])
    log(
        "step 3: validation loss by epoch "
        + " / ".join(f"{v:.4f}" for v in val_losses)
        + f"; epoch {best} kept"
    )
    return checkpoint_path(best)


# ---------------------------------------------------------------------------
# Steps 4 and 5: install the model and write its reference embeddings
# ---------------------------------------------------------------------------

def install(checkpoint: Path) -> None:
    """Copy the bare transformer files of `checkpoint` to models/, where
    qorfinder --model all-MiniLM-L6-v2-ft-cuad looks for them."""
    if FINE_TUNED_MODEL.exists():
        shutil.rmtree(FINE_TUNED_MODEL)
    FINE_TUNED_MODEL.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        shutil.copyfile(checkpoint / name, FINE_TUNED_MODEL / name)
    log(
        f"step 4: {checkpoint} -> {FINE_TUNED_MODEL} "
        f"(model.safetensors sha256={sha256(FINE_TUNED_MODEL / 'model.safetensors')})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_threads() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(THREADS)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def main() -> int:
    set_threads()
    log("--- finetune.py started ---")
    audit_templates()
    log(f"templates share no 3-word phrase with {TESTSET_QUERIES}")
    held_out = held_out_contracts()
    train_contracts, val_contracts = split_contracts(held_out)
    log(
        f"{len(held_out)} held-out contracts, split by contract "
        f"{len(train_contracts)} train / {len(val_contracts)} val"
    )
    try:
        check_templates(held_out)
        learning_rate = tune_learning_rate(train_contracts, val_contracts)
        best = train(learning_rate, train_contracts, val_contracts)
    except KeyboardInterrupt:
        print("\ninterrupted; rerun to resume from the latest checkpoint", file=sys.stderr)
        return 130
    install(best)
    reference_embeddings.write_fine_tuned()
    log("--- finetune.py finished ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
