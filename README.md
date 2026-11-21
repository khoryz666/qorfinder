# QorFinder

Offline semantic search for English legal documents. QorFinder indexes the
PDF, DOCX and TXT files in a folder and finds the passages closest in meaning
to a plain-English query, with the embedding model running locally on the
CPU. It is one Rust library (`qf-core`), a command-line front end
(`qorfinder`) and a browser UI that the same program serves on 127.0.0.1.

This README is for developing and testing QorFinder: setting up, finding
your way around, and rerunning every test and measurement recorded in
`results/`.

## 1. Set up

**With Nix** (Linux, macOS; flakes enabled). Every tool comes from the locked
flake, so there is nothing else to install:

```bash
nix develop               # or `direnv allow` once: Rust, Node.js 24, Python with ranx and scipy
nix develop .#reference   # only for fine-tuning and reference embeddings: PyTorch, transformers
```

**Windows without Nix.** `tools/setup-windows.ps1` installs the MSVC build
tools, the pinned Rust, Node.js 24, Python 3.14 and Git, and the scoring
packages into `.venv`:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup-windows.ps1 -SkipDocs -WithScoring -Build
```

**Elsewhere without Nix**, install the same versions by hand: rustup reads
`rust-toolchain.toml`, `ui/package.json` asks for Node.js 24, and
`requirements.txt` (scoring) and `requirements-finetune.txt` (fine-tuning)
pin the Python packages to the flake's (`python3 -m pip install -r
requirements.txt`; ranx also pulls in `ir_datasets`, which is never used).

Then fetch the model and build:

```bash
QF_MODELS=minilm python3 scripts/fetch_model.py   # the default model, about 90 MB
python3 scripts/fetch_model.py                    # or all four (about 1.4 GB), for the model comparison
(cd ui && npm ci && npm run build)
cargo build --release
```

Models are pinned by revision and SHA-256 and land in `models/`; nothing
downloads anything after this. Try the build on any folder:

```bash
./target/release/qorfinder index ~/contracts
./target/release/qorfinder search "can the employee work for a competitor after leaving?"
./target/release/qorfinder serve   # the browser UI
```

The examples below write `qorfinder` for `./target/release/qorfinder`, and
run from the repository root, since `eval`'s default folders (`data/testset`,
`data/cuad/sample`, `data/eval-index`, `results/eval`) are relative to it.

## 2. Find your way around

```
crates/qf-core/      the library, one module per pipeline step (section 3.2)
  tests/             integration tests; fixtures/reference/ holds Python reference vectors per model
crates/qf-cli/       the qorfinder binary: index, search, eval, serve and the local API
ui/                  React + TypeScript UI, built with Vite to ui/dist
scripts/             Python, one job per script, settings as constants (table below)
data/testset/        the frozen test set: sampled contracts, categories, spans, the 34 queries
data/fixtures/       hand-made files the T1 system test must skip
results/             the six curated records (table in section 4)
tools/               Windows setup without Nix, and the scoring installer it shares with CI
.github/workflows/   ci.yml (Linux and Windows, same checks), release.yml (zips on a tag)
flake.nix, rust-toolchain.toml, requirements*.txt, ui/package.json   the pinned environment
.cargo/              links the C runtime statically into the Windows executable
about.toml, about.hbs   cargo about: accepted licences and the THIRD_PARTY_LICENSES.html template
licenses/            licence of the bundled model (Apache-2.0)
ref/                 proposal and IIPSPW report
```

Everything else under `data/` (CUAD, indexes, the T1/T7 folders, fine-tuning
checkpoints), `models/` and `results/eval/` is generated and ignored by Git.

| Script | Does | Shell |
|---|---|---|
| `fetch_model.py` | downloads and checks the pinned models (`QF_MODELS` narrows the list) | default |
| `prepare_cuad.py` | checks `CUAD_v1.zip`, builds the frozen test set (N = 30, X = 15%, seed 2026) | default |
| `prepare_system_tests.py` | builds the T1 and T7 folders from CUAD and `data/fixtures/` | default |
| `score.py` | checks itself against a hand-worked run, then scores `results/eval/` | default |
| `finetune.py` | the whole fine-tune: template check, learning rate, training, install, reference vectors | reference |
| `reference_embeddings.py` | writes the reference vectors of every model in `models/` | reference |

No script takes flags: to change a setting, edit its constant, so Git
records it.

## 3. Develop

### 3.1 The check before every commit

The same checks CI runs on Linux and Windows:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                       # fast tests, no model needed
cargo test --workspace -- --include-ignored  # also the tests that load models/
python3 scripts/score.py                     # the scoring self-check
(cd ui && npm ci && npm run build)           # type-checks the UI
```

The ignored tests include `embeddings_match_sentence_transformers`: candle
must reproduce Python sentence-transformers (cosine > 0.99) for every model
in `models/`; models not fetched are skipped, except the default.

### 3.2 How it works

1. **Ingest** (`ingest/`): walk the folder; extract text per format
   (pdf-extract page by page, docx-rs paragraphs, UTF-8 text); skip
   encrypted PDFs, PDFs with no text layer and files over 50 MB; normalise
   to NFC, join hyphenated line breaks, collapse spaces and keep paragraph
   breaks, remembering where every character came from.
2. **Chunk** (`chunk/`): split on paragraphs and numbered clauses (`12.`,
   `12.3`, `(a)`), pack them up to 200 model tokens, and cut over-long
   paragraphs with a 40-token overlapping window. Every chunk keeps its
   character range in the original text.
3. **Embed** (`embed/`, driven by `index.rs`): the chosen model on candle,
   pooled as the model says and L2-normalised, with its query and passage
   prefixes; several batches are embedded at once.
4. **Store** (`store/`): documents, chunks and the build settings in SQLite;
   vectors in one flat little-endian `vectors.f32`.
5. **Search** (`search.rs`): embed the query the same way and take the exact
   top-k by dot product.
6. **Baseline and hybrid** (`bm25.rs`, `fusion.rs`, `eval.rs`): a tantivy
   BM25 index over the same chunks (English stemming, k1 = 1.2, b = 0.75),
   and Reciprocal Rank Fusion of the two full rankings (κ = 5).

### 3.3 Commands and options

| Command | What it does |
|---|---|
| `index <folder> [--dry-run]` | Extract, chunk and embed every PDF, DOCX and TXT file, replacing the previous index. Bad files are skipped with a reason. `--dry-run` prints every chunk with its offsets and page instead. |
| `search "<query>" [-k 10]` | Print the top passages with file, page, character range and score. |
| `eval [--timing] [--kappa 5] [--reindex]` | Run the test queries through BM25, semantic search and RRF and write TREC qrels and runs to `results/eval/`; with `--timing`, time every query against `--index-dir` instead. |
| `serve [--port 0]` | The browser UI on 127.0.0.1 only; every API call needs the random token in the address it opens. |

Global options: `--index-dir` (`QORFINDER_INDEX_DIR`; default
`qorfinder/index` in the user's local data folder), `--model`
(`QORFINDER_MODEL`), `--model-dir` (`QORFINDER_MODEL_DIR`; default
`models/<name>` next to the executable or in the current folder),
`--max-tokens` (200) and `--overlap` (40). An index is only searched with the
model and chunk settings it was built with.

### 3.4 Models

Each model is one entry in `crates/qf-core/src/embed/models.rs`
(architecture, size, input length, pooling, prefixes), one pin in
`scripts/fetch_model.py` (revision and SHA-256 of each file) and one
reference fixture. To add one, add the first two, fetch it and run
`reference_embeddings.py` for the third.

| `--model` | Architecture | Size | Pooling | Query / passage prefix |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (default) | BERT, 6 layers | 384 | mean | none |
| `sentence-transformers/all-mpnet-base-v2` | MPNet, 12 layers | 768 | mean | none |
| `BAAI/bge-base-en-v1.5` | BERT, 12 layers | 768 | first token | `Represent this sentence for searching relevant passages: ` / none |
| `intfloat/e5-base-v2` | BERT, 12 layers | 768 | mean | `query: ` / `passage: ` |
| `all-MiniLM-L6-v2-ft-cuad` | BERT, 6 layers | 384 | mean | none (trained by `finetune.py`, section 4.5) |

### 3.5 Releases

Pushing a tag `v*` runs `release.yml`: one zip per OS
(`qorfinder-<tag>-linux-x86_64.zip`, `qorfinder-<tag>-windows-x86_64.zip`)
with the executable, `models/all-MiniLM-L6-v2/`, the built UI in `ui/`, the
licences and `THIRD_PARTY_LICENSES.html`. The executable finds the model and
UI next to itself, so the unpacked folder runs as it is. The Linux binary is
built outside Nix so that it runs on Ubuntu 24.04 and Linux Mint 22 without
Nix.

## 4. Test and reproduce the results

Each file in `results/` opens every section with what it measures, the
commit, the machine and the exact commands; this section is the order to run
them in.

| Record | Holds | Report |
|---|---|---|
| `effectiveness.txt` | nDCG@10, MRR@10, R-Precision and Wilcoxon p for every model; the model choice | Tables 4.1, 4.2 |
| `per_query.csv` | every query's scores for every model | Table 4.2 |
| `tuning.txt` | the chunk-size and κ sweeps; how the fine-tune was set up | §3.7, §4.4 |
| `efficiency.txt` | 510-contract indexing and latency on both machines; per-model latency | Table 4.3 |
| `timing.csv` | every query's encode, scan and total time per run | Table 4.3 |
| `system-tests.txt` | release v0.0.0 on the testing machine: T1, T2, T5, T7, T8, T11, T12 | Table 4.4 |

### 4.1 The test collection

Download `CUAD_v1.zip` (Hendrycks et al., 2021, CC BY 4.0) from
<https://zenodo.org/records/4595826> into `data/cuad/` (no need to unzip it),
then:

```bash
python3 scripts/prepare_cuad.py
```

It checks the zip's SHA-256, writes all 510 contracts to `data/cuad/txt/`
with offsets matching the annotations, samples 30 into `data/cuad/sample/`
and writes `docs.txt`, `categories.tsv` and `spans.tsv` to `data/testset/`,
reproducing the committed files exactly. The 34 queries in
`data/testset/queries.jsonl` are frozen. Evaluation indexes the sample only,
since the spans cover the sample alone; `data/cuad/txt/` is for the timing
run.

### 4.2 Retrieval effectiveness

```bash
qorfinder eval              # indexes data/cuad/sample into data/eval-index, writes results/eval/
python3 scripts/score.py    # writes scores.txt and per_query.csv beside the runs
```

`results/eval/scores.txt` should match the all-MiniLM-L6-v2 section of
`effectiveness.txt`: the TREC files are regenerated byte for byte. For
another model, give it its own index, and score it before the next run
overwrites `results/eval/`:

```bash
qorfinder --model intfloat/e5-base-v2 --index-dir data/eval-e5 eval
python3 scripts/score.py
```

The chunk-size and κ sweeps in `tuning.txt` are the same pair of commands in
a loop over `eval --reindex --max-tokens M --overlap O` and `eval --kappa K`.
Settings were chosen by leave-one-category-out cross-validation over the
per-query scores, as `effectiveness.txt` describes.

### 4.3 Efficiency

```bash
qorfinder --index-dir data/full-index index data/cuad/txt   # all 510 contracts; prints time and peak memory
qorfinder --index-dir data/full-index eval --timing         # 34 queries x 5 runs, writes results/eval/timing.csv
```

The reported figures come from the testing machine (a ThinkPad T590 on
Windows 10, with the release zip) and the dev machine; `efficiency.txt` has
both.

### 4.4 System tests

```bash
python3 scripts/prepare_system_tests.py   # data/t1-mixed/ and data/t7-change/ (its state before the change)
qorfinder --index-dir <tmp> index data/t1-mixed
```

The T1 folder holds 10 PDF, 9 DOCX and 10 TXT contracts plus five files that
must be skipped with a reason. The T7 changes, the serve checks (T5, T8) and
the T12 page check are done by hand, step by step as `system-tests.txt`
describes.

### 4.5 Fine-tuning

```bash
nix develop .#reference --command python3 scripts/finetune.py
qorfinder --model all-MiniLM-L6-v2-ft-cuad --index-dir data/eval-ft eval --reindex
python3 scripts/score.py
```

`finetune.py` trains on the clauses of the 480 CUAD contracts outside the
test sample, and refuses to run if any training query shares a three-word
phrase with a test query. It checks the templates, picks the learning rate
by validation loss, trains (it resumes from the newest checkpoint in
`data/finetune/`), copies the best epoch to
`models/all-MiniLM-L6-v2-ft-cuad/` and rewrites its reference fixture. The
whole script took 3 h 44 min on the dev machine in the last rerun.

The weights are not published, so the model is reproduced only by
retraining, and training is not deterministic even on the same machine: a
rerun on the dev machine chose a different learning rate and gave different
weights from the reported ones (SHA-256 `2a164e0e…`), with nDCG@10 within
0.01 of the record and no significant difference (`results/tuning.txt`). The
Rust embedder is then checked against the new model's own fixture.

## License

MIT. The bundled model is Apache-2.0; see `licenses/all-MiniLM-L6-v2.txt`.
