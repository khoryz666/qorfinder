# QorFinder Learning Roadmap
*Assumes zero prior knowledge of NLP, embeddings, or information retrieval. Read top to bottom — each tier depends on the one before it.*

Project reference: the current implementation is Rust-only — `tantivy` (lexical/BM25) + `usearch` (vector/HNSW) + `redb` (embedded KV) + ONNX Runtime (embeddings), fused with Reciprocal Rank Fusion, evaluated on BEIR via nDCG/Recall/MRR. This roadmap is scoped to exactly that stack.

---

## Tier 0 — Math & Programming Prerequisites

| Source | Why |
|---|---|
| [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) | Vectors, dot products, cosine similarity — the literal math behind "semantic distance" |
| [3Blue1Brown — Neural Networks series](https://www.3blue1brown.com/topics/neural-networks) | Visual intuition for what a neural net actually computes, before touching transformers |
| [The Rust Book](https://doc.rust-lang.org/book/) | You're implementing in Rust — ownership, traits, and error handling (`Result`) show up constantly in `tantivy`/`usearch`/`ort` code |

---

## Tier 1 — NLP & Deep Learning Foundations

| Source | Why |
|---|---|
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) | Free, structured, zero-to-transformers. Do chapters 1–3 minimum before anything else below |
| [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Best visual explanation of self-attention that exists. Read before the actual paper |
| [Vaswani et al., "Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762) | The source paper. Read after the illustrated version above, not instead of it |
| [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) | Bridges "transformer" to "the thing that produces embeddings" |

---

## Tier 2 — Sentence Embeddings (the core of this project)

| Source | Why |
|---|---|
| [Reimers & Gurevych, "Sentence-BERT" (2019)](https://arxiv.org/abs/1908.10084) | Explains why raw BERT can't be pooled into one vector naively, and how SBERT fixes it — this is the lineage every embedding model you'll use descends from |
| [Sentence-Transformers docs](https://www.sbert.net/) | Practical, code-first companion to the paper above — pooling strategies, training losses, model zoo |
| [Gao et al., "SimCSE" (2021)](https://arxiv.org/abs/2104.08821) | Cleaner contrastive-learning framing than SBERT; modern models (BGE, E5, Qwen3-Embedding) build on this idea, not SBERT directly |

---

## Tier 3 — Classical Information Retrieval (the lexical/BM25 half)

| Source | Why |
|---|---|
| [Manning, Raghavan & Schütze — *Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/) (free, Stanford) | Read Ch. 1–2 (inverted index) and Ch. 6 (TF-IDF/ranking) before touching BM25 |
| [Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf) | The actual scoring formula `tantivy` implements under the hood |
| [tantivy repo/docs](https://github.com/quickwit-oss/tantivy) | Lucene-alike; once you understand inverted indexes conceptually, this is the concrete API you're calling |

---

## Tier 4 — Vector Search / Approximate Nearest Neighbor (the `usearch` half)

| Source | Why |
|---|---|
| [Malkov & Yashunin, "Efficient and Robust ANN Search Using HNSW Graphs" (2018)](https://arxiv.org/abs/1603.09320) | The algorithm `usearch` runs. Explains the recall/latency tradeoff you'll be tuning (`M`, `ef_construction`, `ef_search`) |
| [usearch repo/docs](https://github.com/unum-cloud/usearch) | Implementation-specific details: SIMD acceleration, quantized vector storage |

---

## Tier 5 — Hybrid Search & Evaluation

| Source | Why |
|---|---|
| [Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion" (2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Short paper — literally the fusion algorithm merging your dense + lexical results |
| [Manning et al., *IR-book*, Ch. 8](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf) | Defines Precision/Recall, MRR, and nDCG — the exact metrics your `eval` command reports |
| [Thakur et al., "BEIR" (2021)](https://arxiv.org/abs/2104.08663) | The benchmark your `corpus beir` command downloads and `BENCHMARK.md` reports against |
| [Muennighoff et al., "MTEB" (2022)](https://arxiv.org/abs/2210.07316) | Explains what public embedding-model leaderboard scores actually measure, for when you're choosing/comparing models |

---

## Tier 6 — Runtime, Quantization & Chunking (systems/engineering layer)

| Source | Why |
|---|---|
| [ONNX Runtime docs](https://onnxruntime.ai/docs/) | Read "Get Started" + "Performance" — execution providers, graph optimization levels, since you're CPU-bound |
| [Hugging Face Optimum — Quantization concepts](https://huggingface.co/docs/optimum/concept_guides/quantization) | Dynamic vs. static quantization, INT8 — concepts transfer even if you quantize via ONNX Runtime's own tools |
| [`ort` crate docs](https://docs.rs/ort) | Rust bindings to ONNX Runtime — the actual API you call for inference |
| [`redb` docs](https://docs.rs/redb) | Embedded KV store used for fingerprint/metadata bookkeeping |
| [Pinecone — Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/) | Fixed-size vs. semantic vs. recursive chunking, and why overlap affects recall at chunk boundaries |

---

## Suggested Reading Order

1. Tier 0 (skim — reference as needed)
2. Hugging Face NLP Course ch. 1–3 → Illustrated Transformer → Attention paper → Illustrated BERT
3. BM25 paper → HNSW paper → RRF paper *(these three explain your currently-running system)*
4. Sentence-BERT paper → SimCSE paper
5. IR-book Ch. 8 → BEIR paper → MTEB paper *(these explain what your `eval` numbers mean)*
6. ONNX Runtime docs + Optimum quantization *(as you get to optimization work)*

---

## Action Plan — Improving Current Performance (mapped to FYP sub-objectives)

Baseline: `intfloat/multilingual-e5-small` (ONNX, ~120 MB), corpus assumed mostly English.

### Step 1 — Establish a baseline number *(prerequisite for everything below)*
- Run `cargo run --release -- corpus beir scifact --out data` → index → `eval --mode dense`
- Record nDCG@10 / Recall / MRR for the **current** model before changing anything — without this, no later claim of "improved performance" is measurable.
- → Satisfies **Sub-objective 3** (benchmarking to verify retrieval quality).

### Step 2 — Swap to a stronger English encoder
- Export `BAAI/bge-large-en-v1.5` (or `intfloat/e5-large-v2` for a closer migration) to ONNX via `optimum-cli export onnx`.
- Re-run the same `eval --mode dense` command → compare nDCG@10 against Step 1's baseline.
- → Satisfies **Sub-objective 2** ("iterate and fine-tune the integration of sentence transformer models... to maximize accuracy").

### Step 3 — Quantize the new model, measure the accuracy/speed trade-off
- Use `onnxruntime.quantization.quantize_dynamic` (INT8) on the exported ONNX model — start from the full-precision export, not a pre-quantized download (see quantization discussion above for why this matters for it to count as your own work).
- Re-run `eval` again on the quantized model; record latency (ms/query) and peak RSS alongside nDCG@10.
- Produce a 3-row comparison table: baseline (e5-small) → bge-large fp32 → bge-large int8.
- → Satisfies **Sub-objective 2** (CPU-latency constraint) and the "Model Optimization" milestone (RAM usage reduced, inference speed improved, accuracy loss bounded).

### Step 4 — Tune hybrid fusion weighting
- Your RRF fusion currently combines dense + lexical results. With a stronger dense model, re-check whether the fusion still behaves well — run `eval --mode lexical`, `--mode dense`, `--mode hybrid` separately and compare, since a better embedding model can shift the optimal balance between the two signals.
- → Satisfies **Sub-objective 1** (loosely-coupled architecture efficiency) and **Sub-objective 3** (deterministic retrieval speed/integrity across configurations).

### Step 5 (stretch) — Fine-tune on your own eval failures
- From the Step 1–3 benchmarks, inspect queries where retrieval fails (`eval` output should show per-query scores). These misses are your hard-negative mining source.
- Fine-tune the chosen model with `sentence-transformers` using `MultipleNegativesRankingLoss` on a small set of (query, correct passage, wrong-but-similar passage) triplets drawn from your own failures.
- Re-benchmark. This step is optional but is the strongest "contribution" evidence: it shows model improvement specific to your corpus, not just picking a bigger off-the-shelf model.
- → Directly supports the project's stated **Contribution** (integration + optimization yielding novelty) beyond simply swapping models.

### Deliverable to keep throughout
A single running table (baseline → model swap → quantized → hybrid-tuned → fine-tuned) of nDCG@10 / Recall / MRR / latency / RAM. This table *is* the evidence for "improved performance" and doubles as the report's benchmarking section.
