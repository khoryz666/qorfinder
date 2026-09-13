use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use walkdir::WalkDir;

use crate::embedder::Embedder;
use crate::store::Store;

pub struct EvalReport {
    pub evaluated: usize,
    pub skipped_no_qrels: usize,
    pub k: usize,
    pub ndcg: f64,
    pub recall: f64,
    pub mrr: f64,
    pub total_seconds: f64,
}

/// Parse a TSV of `qid \t query text`.
pub fn parse_queries(path: &Path) -> Result<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.splitn(2, '\t');
        let qid = fields.next().unwrap_or_default().trim().to_string();
        let query = fields.next().unwrap_or_default().trim().to_string();
        if qid.is_empty() || query.is_empty() {
            bail!(
                "{}:{}: expected 'qid\\tquery', got {:?}",
                path.display(),
                i + 1,
                line
            );
        }
        out.push((qid, query));
    }
    Ok(out)
}

/// Parse qrels as either TREC format (`qid \t 0 \t docid \t rel`) or the
/// BEIR 3-column format (`qid \t docid \t rel`). Returns qid -> set of
/// relevant doc ids (rel > 0).
pub fn parse_qrels(path: &Path) -> Result<HashMap<String, HashSet<String>>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields
            .first()
            .is_some_and(|f| *f == "query-id" || *f == "qid" || *f == "query_id")
        {
            continue;
        }
        let (qid, docid, rel) = match fields.len() {
            len if len >= 4 => (fields[0], fields[2], fields[3]),
            3 => (fields[0], fields[1], fields[2]),
            _ => {
                bail!(
                    "{}:{}: expected TREC or BEIR qrels format, got {:?}",
                    path.display(),
                    i + 1,
                    line
                );
            }
        };
        let rel: i64 = rel.trim().parse().with_context(|| {
            format!("{}:{}: bad relevance value {rel:?}", path.display(), i + 1)
        })?;
        if rel > 0 {
            out.entry(qid.trim().to_string())
                .or_default()
                .insert(docid.trim().to_string());
        }
    }
    Ok(out)
}

/// Keep the first occurrence of each doc id (a document can contribute
/// several chunks to the top-k result).
pub fn dedupe(mut ranked: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    ranked.retain(|d| seen.insert(d.clone()));
    ranked
}

/// Discounted cumulative gain at k with binary relevance.
pub fn dcg_at_k(relevance: &[bool], k: usize) -> f64 {
    relevance
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, rel)| {
            if *rel {
                1.0 / ((i as f64 + 2.0).log2())
            } else {
                0.0
            }
        })
        .sum()
}

/// Ideal DCG at k for `num_relevant` relevant documents.
pub fn idcg_at_k(num_relevant: usize, k: usize) -> f64 {
    (0..num_relevant.min(k))
        .map(|i| 1.0 / ((i as f64 + 2.0).log2()))
        .sum()
}

pub fn ndcg_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    let idcg = idcg_at_k(relevant.len(), k);
    if idcg == 0.0 {
        return 0.0;
    }
    let rels: Vec<bool> = ranked
        .iter()
        .take(k)
        .map(|d| relevant.contains(d))
        .collect();
    dcg_at_k(&rels, k) / idcg
}

pub fn recall_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let found = ranked
        .iter()
        .take(k)
        .filter(|d| relevant.contains(*d))
        .count();
    found as f64 / relevant.len() as f64
}

pub fn reciprocal_rank(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    ranked
        .iter()
        .take(k)
        .position(|d| relevant.contains(d))
        .map(|i| 1.0 / (i as f64 + 1.0))
        .unwrap_or(0.0)
}

/// Map canonical file path strings in the corpus to doc ids (file stems).
pub fn build_doc_map(corpus: &Path) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for entry in WalkDir::new(corpus) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if !crate::parser::is_supported(path) {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        if let Ok(canonical) = std::fs::canonicalize(path) {
            map.insert(canonical.display().to_string(), stem);
        }
    }
    map
}

/// Which retrieval path to score. `Dense` and `Lexical` isolate one side of
/// the store for comparison; `Hybrid` is what the CLI's `query` command
/// actually uses in production (RRF fusion of both).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum EvalMode {
    Dense,
    Lexical,
    Hybrid,
}

/// Run options for `run_eval`, grouped so the function stays under clippy's
/// argument-count lint: `k` is the metrics cut-off, `limit` caps how many
/// *judged* queries are actually embedded and searched, `mode` selects which
/// retrieval path to score.
#[derive(Debug, Clone, Copy)]
pub struct EvalOptions {
    pub k: usize,
    pub limit: Option<usize>,
    pub mode: EvalMode,
}

/// Run every query against the store and score the results against qrels.
/// The corpus must already be indexed. Queries without qrels are skipped
/// (BEIR ships train + test queries together).
pub fn run_eval(
    store: &Store,
    embedder: &Embedder,
    corpus: &Path,
    queries_path: &Path,
    qrels_path: &Path,
    options: EvalOptions,
) -> Result<EvalReport> {
    let EvalOptions { k, limit, mode } = options;
    let queries = parse_queries(queries_path)?;
    let qrels = parse_qrels(qrels_path)?;
    let doc_map = build_doc_map(corpus);

    let mut sum_ndcg = 0.0;
    let mut sum_recall = 0.0;
    let mut sum_mrr = 0.0;
    let mut evaluated = 0usize;
    let mut skipped_no_qrels = 0usize;
    let started = Instant::now();

    for (qid, query) in &queries {
        if limit.is_some_and(|l| evaluated >= l) {
            break;
        }
        let Some(relevant) = qrels.get(qid) else {
            skipped_no_qrels += 1;
            continue;
        };
        let hits: Vec<crate::store::SearchHit> = match mode {
            EvalMode::Dense => {
                let vector = embedder.embed_query(query)?;
                store.search(vector, k as u64)?
            }
            EvalMode::Lexical => store
                .search_lexical(query, k)?
                .into_iter()
                .map(|h| crate::store::SearchHit {
                    file_path: h.file_path,
                    chunk_index: h.chunk_index,
                    score: h.score,
                    text: h.text,
                })
                .collect(),
            EvalMode::Hybrid => crate::query::run_query(store, embedder, query, k as u64)?,
        };
        let ranked: Vec<String> = dedupe(
            hits.iter()
                .filter_map(|h| doc_map.get(&h.file_path).cloned())
                .collect(),
        );
        sum_ndcg += ndcg_at_k(&ranked, relevant, k);
        sum_recall += recall_at_k(&ranked, relevant, k);
        sum_mrr += reciprocal_rank(&ranked, relevant, k);
        evaluated += 1;
    }

    let n = evaluated.max(1) as f64;
    Ok(EvalReport {
        evaluated,
        skipped_no_qrels,
        k,
        ndcg: sum_ndcg / n,
        recall: sum_recall / n,
        mrr: sum_mrr / n,
        total_seconds: started.elapsed().as_secs_f64(),
    })
}
