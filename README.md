# QorFinder

Offline semantic search for English legal documents. QorFinder indexes the
PDF, DOCX and TXT files in a folder on your machine and finds the passages
closest in meaning to a plain-English query. The model runs locally on the
CPU, so no document, query or result ever leaves the machine.

It is one Rust library (`qf-core`) with a command-line front end
(`qorfinder`). The same library will also back a browser UI served on
127.0.0.1.

## Quick start

```bash
# 1. Download the embedding model once (about 90 MB, pinned revision)
python3 scripts/fetch_model.py

# 2. Build
cargo build --release

# 3. Index a folder, then search it. No network is needed from here on.
./target/release/qorfinder index ~/contracts
./target/release/qorfinder search "can the employee work for a competitor after leaving?"
```

`index --dry-run` prints every chunk with its character offsets and page
instead of embedding anything, which is useful for checking extraction.

## Commands

| Command | What it does |
|---|---|
| `index <folder> [--dry-run]` | Extract, chunk and embed every PDF, DOCX and TXT file, replacing the previous index. Bad files are skipped with a reason. |
| `search "<query>" [-k 10]` | Print the top passages with file, page, character range and score. |

Global options: `--index-dir` (or `QORFINDER_INDEX_DIR`; default
`qorfinder/index` in the user's local data folder) and `--model-dir` (or
`QORFINDER_MODEL_DIR`; default `models/all-MiniLM-L6-v2`). `--max-tokens`
(200) and `--overlap` (40) set the chunk size; an index is only searched with
the settings it was built with.

## How it works

1. **Ingest**: walk the folder; extract text per format (lopdf page by
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

## Test collection

The evaluation uses CUAD v1 (Hendrycks et al., 2021, CC BY 4.0). Download
`CUAD_v1.zip` from <https://zenodo.org/records/4595826>, unzip it under
`data/cuad/`, then:

```bash
python3 scripts/prepare_cuad.py -n 15 -x 0.3
```

This writes all 510 contracts to `data/cuad/txt/` with offsets matching the
annotations, samples `-n` contracts with a fixed seed, prints how many
contain each clause category, keeps the categories found in at most `-x` of
them, and writes `docs.txt`, `categories.tsv` and `spans.tsv` to
`data/testset/`.

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
crates/qf-core/   library: ingest, chunk, embed, store, index, search
crates/qf-cli/    the `qorfinder` binary
scripts/          model download, CUAD preparation, reference embeddings
ref/              proposal and IIPSPW report
```

## License

MIT
