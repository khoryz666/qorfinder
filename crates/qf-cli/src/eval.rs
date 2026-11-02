//! `qorfinder eval`: run the test queries through BM25, semantic search and
//! their RRF fusion and write TREC files for `scripts/score.py`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use qf_core::bm25::Bm25;
use qf_core::chunk::ChunkConfig;
use qf_core::eval::{judge, read_queries, read_spans, write_qrels, write_run};
use qf_core::fusion::{RRF_KAPPA, reciprocal_rank_fusion};
use qf_core::index::{build_index, check_index_dir, index_meta};
use qf_core::search::Searcher;
use qf_core::store::Store;
use qf_core::{Hit, IndexMeta};

use crate::cli::{load_embedder, load_tokenizer};
use crate::progress::Bars;

/// Hits written per query to each run file. Deeper than any query's number
/// of relevant chunks, so R-Precision sees the whole top R.
const RUN_DEPTH: usize = 1000;

/// One method's hits for every query, by query id.
type Run = Vec<(String, Vec<Hit>)>;

/// Index `docs` into `index_dir` (unless an index of the same folder with
/// the same settings is already there), judge every chunk against the
/// spans, and write `qrels.trec` and one run file per method to `out`.
pub fn effectiveness(
    testset: &Path,
    docs: &Path,
    out: &Path,
    index_dir: &Path,
    model_dir: &Path,
    config: ChunkConfig,
    reindex: bool,
) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    let queries = read_queries(&testset.join("queries.jsonl"))?;
    let spans = read_spans(&testset.join("spans.tsv"))?;
    // Before the model loads, so a wrong --index-dir fails at once.
    check_index_dir(docs, index_dir)?;
    let embedder = load_embedder(model_dir)?;
    let meta = index_meta(&embedder, config);
    if reindex || !is_index_of(index_dir, docs, &meta) {
        writeln!(
            stdout,
            "indexing {} into {}",
            docs.display(),
            index_dir.display()
        )?;
        let tokenizer = load_tokenizer(model_dir)?;
        let report = build_index(
            docs,
            index_dir,
            &embedder,
            &tokenizer,
            config,
            &Bars::default(),
        )?;
        for s in &report.skipped {
            writeln!(stdout, "skipped {}: {}", s.path.display(), s.reason)?;
        }
    } else {
        writeln!(
            stdout,
            "using the index of {} in {}",
            docs.display(),
            index_dir.display()
        )?;
    }

    let searcher = Searcher::open(index_dir, &embedder, &meta)?;
    let documents = searcher.store().documents()?;
    let chunks = searcher.store().chunks()?;
    let bm25 = Bm25::build(&chunks)?;
    let qrels = judge(&queries, &spans, &documents, &chunks);

    // Full rankings, so RRF fuses true ranks; the run files keep the top
    // RUN_DEPTH of each.
    let all = chunks.len();
    let mut runs: [(&str, Run); 3] = [
        ("bm25", Vec::new()),
        ("semantic", Vec::new()),
        ("hybrid", Vec::new()),
    ];
    for query in &queries {
        let lexical = bm25.search(&query.text, all)?;
        let semantic = searcher.search(&query.text, all)?.hits;
        let hybrid = reciprocal_rank_fusion(&ids(&lexical), &ids(&semantic), all, RRF_KAPPA);
        for ((_, run), mut hits) in runs.iter_mut().zip([lexical, semantic, hybrid]) {
            hits.truncate(RUN_DEPTH);
            run.push((query.id.clone(), hits));
        }
    }

    std::fs::create_dir_all(out).with_context(|| format!("cannot create {}", out.display()))?;
    write_file(&out.join("qrels.trec"), |f| {
        write_qrels(f, &queries, &qrels)
    })?;
    for (tag, run) in &runs {
        write_file(&out.join(format!("run_{tag}.trec")), |f| {
            write_run(f, tag, run)
        })?;
    }

    let relevant: usize = qrels.values().map(|ids| ids.len()).sum();
    writeln!(
        stdout,
        "{} queries over {} documents, {} chunks; {relevant} relevant (query, chunk) pairs",
        queries.len(),
        documents.len(),
        chunks.len(),
    )?;
    for query in &queries {
        if qrels[&query.id].is_empty() {
            writeln!(stdout, "warning: no chunk is relevant to {}", query.id)?;
        }
    }
    writeln!(
        stdout,
        "wrote qrels.trec and run_{{bm25,semantic,hybrid}}.trec to {}",
        out.display()
    )?;
    Ok(())
}

/// Whether `index_dir` holds a complete index of `docs` built with `meta`.
fn is_index_of(index_dir: &Path, docs: &Path, meta: &IndexMeta) -> bool {
    let Ok(store) = Store::open(index_dir) else {
        return false;
    };
    let source = store.source_folder().ok().flatten();
    store.check_meta(meta).is_ok() && source == std::path::absolute(docs).ok()
}

fn ids(hits: &[Hit]) -> Vec<u32> {
    hits.iter().map(|h| h.chunk_id).collect()
}

fn write_file(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
) -> Result<()> {
    let mut file = BufWriter::new(
        File::create(path).with_context(|| format!("cannot create {}", path.display()))?,
    );
    write(&mut file)
        .and_then(|()| file.flush())
        .with_context(|| format!("cannot write {}", path.display()))
}
