//! `qorfinder eval`: run the test queries through BM25, semantic search and
//! their RRF fusion and write TREC files for `scripts/score.py`, or time
//! the queries for the efficiency results.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;

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

/// Timed runs per query after one warm-up.
const TIMED_RUNS: usize = 5;

/// One method's hits for every query, by query id.
type Run = Vec<(String, Vec<Hit>)>;

/// One query's median times over its timed runs.
struct Timed<'a> {
    id: &'a str,
    encode: Duration,
    scan: Duration,
    total: Duration,
}

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

/// Time every query against the index in `index_dir`: one warm-up run, then
/// TIMED_RUNS timed runs, and write each query's median encode, scan and
/// total time to `out/timing.csv`. Prints the median and 95th percentile
/// over the queries.
pub fn timing(
    testset: &Path,
    out: &Path,
    index_dir: &Path,
    model_dir: &Path,
    config: ChunkConfig,
) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    let queries = read_queries(&testset.join("queries.jsonl"))?;
    let embedder = load_embedder(model_dir)?;
    let searcher = Searcher::open(index_dir, &embedder, &index_meta(&embedder, config))?;
    let chunks = searcher.store().chunk_count()?;

    let mut rows = Vec::with_capacity(queries.len());
    for query in &queries {
        searcher.search(&query.text, 10)?;
        let mut encode = Vec::with_capacity(TIMED_RUNS);
        let mut scan = Vec::with_capacity(TIMED_RUNS);
        let mut total = Vec::with_capacity(TIMED_RUNS);
        for _ in 0..TIMED_RUNS {
            let result = searcher.search(&query.text, 10)?;
            encode.push(result.encode_time);
            scan.push(result.scan_time);
            total.push(result.encode_time + result.scan_time);
        }
        rows.push(Timed {
            id: &query.id,
            encode: median(encode),
            scan: median(scan),
            total: median(total),
        });
    }

    std::fs::create_dir_all(out).with_context(|| format!("cannot create {}", out.display()))?;
    write_file(&out.join("timing.csv"), |f| {
        writeln!(f, "query,encode_ms,scan_ms,total_ms")?;
        for row in &rows {
            writeln!(
                f,
                "{},{:.3},{:.3},{:.3}",
                row.id,
                ms(row.encode),
                ms(row.scan),
                ms(row.total)
            )?;
        }
        Ok(())
    })?;

    writeln!(
        stdout,
        "{} queries over {chunks} chunks, median of {TIMED_RUNS} runs each",
        rows.len()
    )?;
    for (name, values) in [
        ("total", rows.iter().map(|r| r.total).collect::<Vec<_>>()),
        ("encode", rows.iter().map(|r| r.encode).collect()),
        ("scan", rows.iter().map(|r| r.scan).collect()),
    ] {
        let mut values = values;
        values.sort();
        writeln!(
            stdout,
            "{name:>6}: median {:.1} ms, p95 {:.1} ms",
            ms(median(values.clone())),
            ms(percentile(&values, 95)),
        )?;
    }
    writeln!(stdout, "wrote {}", out.join("timing.csv").display())?;
    if let Some(line) = crate::memory::peak_line() {
        writeln!(stdout, "{line}")?;
    }
    Ok(())
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

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The middle value, or the mean of the two middle values.
fn median(mut values: Vec<Duration>) -> Duration {
    values.sort();
    let n = values.len();
    if n == 0 {
        return Duration::ZERO;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2
    }
}

/// Nearest-rank percentile of sorted `values`.
fn percentile(sorted: &[Duration], p: usize) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = (p * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}
