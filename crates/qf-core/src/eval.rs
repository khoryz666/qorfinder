//! Evaluation on the CUAD test collection: read the queries and annotated
//! spans, judge which chunks are relevant to each query, and write TREC
//! qrels and run files for ranx to score.

use std::collections::{BTreeSet, HashMap};
use std::io::{self, Write};
use std::ops::Range;
use std::path::Path;

use crate::{Chunk, Document, Hit};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("{file} line {line}: {reason}")]
    Parse {
        file: String,
        line: usize,
        reason: String,
    },
}

/// One test query from `queries.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    /// For example `Q01m`.
    pub id: String,
    /// The CUAD clause category whose spans are its relevant passages.
    pub category: String,
    /// `matched` (wording taken from the category description) or
    /// `paraphrased`.
    pub kind: String,
    pub text: String,
}

/// One annotated clause from `spans.tsv`: characters `[start, end)` of
/// `file` belong to `category`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub category: String,
    pub file: String,
    pub start: usize,
    pub end: usize,
}

/// Read `queries.jsonl`: one JSON object per line with `id`, `category`,
/// `type` and `text`.
pub fn read_queries(path: &Path) -> Result<Vec<Query>, EvalError> {
    let text = std::fs::read_to_string(path)?;
    let mut queries = Vec::new();
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let error = |reason: String| EvalError::Parse {
            file: path.display().to_string(),
            line: n + 1,
            reason,
        };
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|e| error(e.to_string()))?;
        let field = |key: &str| {
            value[key]
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| error(format!("missing text field {key:?}")))
        };
        let query = Query {
            id: field("id")?,
            category: field("category")?,
            kind: field("type")?,
            text: field("text")?,
        };
        if query.text.trim().is_empty() {
            return Err(error(format!("query {} has no text", query.id)));
        }
        queries.push(query);
    }
    Ok(queries)
}

/// Read `spans.tsv`: category, file, start and end, tab-separated.
pub fn read_spans(path: &Path) -> Result<Vec<Span>, EvalError> {
    let text = std::fs::read_to_string(path)?;
    let mut spans = Vec::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.is_empty()) {
        let error = |reason: &str| EvalError::Parse {
            file: path.display().to_string(),
            line: n + 1,
            reason: reason.to_string(),
        };
        let [category, file, start, end] = line
            .split('\t')
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| error("expected 4 tab-separated fields"))?;
        let start = start.parse().map_err(|_| error("start is not a number"))?;
        let end = end.parse().map_err(|_| error("end is not a number"))?;
        if start >= end {
            return Err(error("span is empty"));
        }
        spans.push(Span {
            category: category.to_string(),
            file: file.to_string(),
            start,
            end,
        });
    }
    Ok(spans)
}

/// The chunks relevant to each query (report 4.1.3): a chunk is relevant
/// if at least half of its characters fall inside spans of the query's
/// category in the same file, or if it contains a whole span. Relevance is
/// binary. Documents are matched to spans by file name. Every query gets
/// an entry, empty if nothing is relevant.
pub fn judge(
    queries: &[Query],
    spans: &[Span],
    documents: &[Document],
    chunks: &[Chunk],
) -> HashMap<String, BTreeSet<u32>> {
    // Spans of each category, by document id.
    let doc_ids: HashMap<&str, u32> = documents
        .iter()
        .filter_map(|d| Some((d.path.file_name()?.to_str()?, d.id)))
        .collect();
    let mut by_category: HashMap<(&str, u32), Vec<Range<usize>>> = HashMap::new();
    for span in spans {
        if let Some(&doc) = doc_ids.get(span.file.as_str()) {
            by_category
                .entry((span.category.as_str(), doc))
                .or_default()
                .push(span.start..span.end);
        }
    }
    queries
        .iter()
        .map(|query| {
            let relevant = chunks
                .iter()
                .filter(|chunk| {
                    by_category
                        .get(&(query.category.as_str(), chunk.doc_id))
                        .is_some_and(|spans| is_relevant(chunk.char_start..chunk.char_end, spans))
                })
                .map(|chunk| chunk.id)
                .collect();
            (query.id.clone(), relevant)
        })
        .collect()
}

/// The relevance rule for one chunk against the spans of one category in
/// its document. Overlapping spans are merged first, so no character is
/// counted twice.
pub fn is_relevant(chunk: Range<usize>, spans: &[Range<usize>]) -> bool {
    if spans
        .iter()
        .any(|s| chunk.start <= s.start && s.end <= chunk.end)
    {
        return true;
    }
    let inside: usize = merge(spans)
        .iter()
        .map(|s| {
            chunk
                .end
                .min(s.end)
                .saturating_sub(chunk.start.max(s.start))
        })
        .sum();
    let len = chunk.end - chunk.start;
    len > 0 && 2 * inside >= len
}

/// `spans` sorted and with overlapping ranges joined.
fn merge(spans: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut sorted = spans.to_vec();
    sorted.sort_by_key(|s| s.start);
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(sorted.len());
    for span in sorted {
        match merged.last_mut() {
            Some(last) if span.start <= last.end => last.end = last.end.max(span.end),
            _ => merged.push(span),
        }
    }
    merged
}

/// Write TREC qrels: `qid 0 chunk_id 1` for every relevant chunk, in query
/// order.
pub fn write_qrels(
    out: &mut impl Write,
    queries: &[Query],
    qrels: &HashMap<String, BTreeSet<u32>>,
) -> io::Result<()> {
    for query in queries {
        for id in qrels.get(&query.id).into_iter().flatten() {
            writeln!(out, "{} 0 {id} 1", query.id)?;
        }
    }
    Ok(())
}

/// Write a TREC run: `qid Q0 chunk_id rank score tag` for each query's
/// hits, ranks from 1. Scores are printed with enough digits that ranx
/// sorts them in the same order as the hits.
pub fn write_run(out: &mut impl Write, tag: &str, runs: &[(String, Vec<Hit>)]) -> io::Result<()> {
    for (qid, hits) in runs {
        for (rank, hit) in hits.iter().enumerate() {
            writeln!(
                out,
                "{qid} Q0 {} {} {:.9} {tag}",
                hit.chunk_id,
                rank + 1,
                hit.score
            )?;
        }
    }
    Ok(())
}
