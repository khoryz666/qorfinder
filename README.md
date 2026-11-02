# QorFinder

Offline semantic search for English legal documents. QorFinder indexes the
PDF, DOCX and TXT files in a folder on your machine and finds the passages
closest in meaning to a plain-English query. The model runs locally on the
CPU, so no document, query or result ever leaves the machine.

It is one Rust library (`qf-core`) with a command-line front end
(`qorfinder`) and a browser UI that the same program serves on 127.0.0.1.

## Quick start

```bash
# 1. Download the embedding model once (about 90 MB, pinned revision)
python3 scripts/fetch_model.py

# 2. Build (the UI once, then the program)
(cd ui && npm ci && npm run build)
cargo build --release

# 3. Index a folder, then search it. No network is needed from here on.
./target/release/qorfinder index ~/contracts
./target/release/qorfinder search "can the employee work for a competitor after leaving?"

# Or do both in the browser
./target/release/qorfinder serve
```

A tagged release instead ships one zip per OS (`qorfinder-<tag>-linux-x86_64.zip`,
`qorfinder-<tag>-windows-x86_64.zip`) holding the executable with the model in
`models/all-MiniLM-L6-v2/` and the UI in `ui/` beside it, so it runs as
unpacked with no download. `THIRD_PARTY_LICENSES.html` lists the licences of
the Rust libraries built into it.

`index --dry-run` prints every chunk with its character offsets and page
instead of embedding anything, which is useful for checking extraction.

## Commands

| Command | What it does |
|---|---|
| `index <folder> [--dry-run]` | Extract, chunk and embed every PDF, DOCX and TXT file, replacing the previous index. Bad files are skipped with a reason. |
| `search "<query>" [-k 10]` | Print the top passages with file, page, character range and score. |
| `eval [--timing]` | Run the CUAD test queries through BM25, semantic search and their RRF fusion and write TREC qrels and run files; with `--timing`, time every query instead. |
| `serve [--port 0]` | Open the browser UI: choose a folder, watch indexing, search and open files. Listens on 127.0.0.1 only, and every API call needs the random token in the address it opens. |

Global options: `--index-dir` (or `QORFINDER_INDEX_DIR`; default
`qorfinder/index` in the user's local data folder) and `--model-dir` (or
`QORFINDER_MODEL_DIR`; default `models/all-MiniLM-L6-v2` next to the
executable, or in the current folder). `--max-tokens`
(200) and `--overlap` (40) set the chunk size; an index is only searched with
the settings it was built with.

## How it works

1. **Ingest**: walk the folder; extract text per format (pdf-extract page by
   page, docx-rs paragraphs, UTF-8 text); skip encrypted PDFs, PDFs with no
   text layer and files over 50 MB; normalise to NFC, join hyphenated line
   breaks, collapse spaces and keep paragraph breaks, remembering where every
   character came from.
2. **Chunk**: split on paragraphs and numbered clauses (`12.`, `12.3`,
   `(a)`), pack them up to 200 model tokens, and cut over-long paragraphs
   with a 40-token overlapping window. Every chunk keeps its character range
   in the original text.
3. **Embed**: all-MiniLM-L6-v2 on candle, mean-pooled and L2-normalised,
   matching Python sentence-transformers (cosine > 0.99, checked by a test).
4. **Store**: documents, chunks and the build settings in SQLite; vectors
   in one flat little-endian `vectors.f32`.
5. **Search**: embed the query the same way and take the exact top-k by
   dot product.
6. **Baseline and hybrid** (evaluation): a tantivy BM25 index over the same
   chunks (English stemming, k1 = 1.2, b = 0.75), and Reciprocal Rank Fusion
   of the two full rankings (κ = 60, untuned).

## Test collection

The evaluation uses CUAD v1 (Hendrycks et al., 2021, CC BY 4.0). Download
`CUAD_v1.zip` from <https://zenodo.org/records/4595826>, unzip it under
`data/cuad/`, then:

```bash
python3 scripts/prepare_cuad.py -n 30 -x 0.15
```

This writes all 510 contracts to `data/cuad/txt/` with offsets matching the
annotations, samples `-n` contracts with a fixed seed and copies them to
`data/cuad/sample/`, prints how many contain each clause category, keeps the
categories found in at most `-x` of them, and writes `docs.txt`,
`categories.tsv` and `spans.tsv` to `data/testset/`. The evaluation indexes
`data/cuad/sample/` only, since the spans cover the sample alone; the full
`data/cuad/txt/` is for the timing run.

The queries in `data/testset/queries.jsonl` are written before the first
run and then frozen. To score:

```bash
qorfinder eval                       # indexes data/cuad/sample into data/eval-index
python3 scripts/score.py results/eval   # nDCG@10, MRR@10, R-Precision, Wilcoxon
qorfinder --index-dir <index of data/cuad/txt> eval --timing   # latency
```

`score.py` writes `scores.txt` and `per_query.csv` beside the run files;
`scripts/test_score.py` checks it against a hand-worked example.

## Develop

```bash
nix develop          # or `direnv allow`: Rust, Node.js, Python with ranx and scipy
cargo test           # fast tests, no model needed
cargo test -- --include-ignored   # also the tests that load models/
```

Every tool comes from the locked flake, so any machine with Nix (flakes
enabled) gets the same environment with no other setup. Regenerating the
reference embeddings needs sentence-transformers and PyTorch, which live in a
separate shell:

```bash
nix develop .#reference --command python3 scripts/reference_embeddings.py
```

Layout:

```
crates/qf-core/   library: ingest, chunk, embed, store, index, search, bm25, fusion, eval
crates/qf-cli/    the `qorfinder` binary and the local API (serve)
ui/               React + TypeScript UI, built with Vite to ui/dist
scripts/          model download, CUAD preparation, scoring, reference embeddings
licenses/         licence of the bundled model (Apache-2.0)
ref/              proposal and IIPSPW report
```

## License

MIT. The bundled model is Apache-2.0; see `licenses/all-MiniLM-L6-v2.txt`.
