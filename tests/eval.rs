use std::collections::HashSet;

use qorfinder::eval::{
    dcg_at_k, dedupe, idcg_at_k, ndcg_at_k, parse_qrels, parse_queries, recall_at_k,
    reciprocal_rank,
};

fn set(items: &[&str]) -> HashSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn ranked(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn parses_queries_tsv() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("queries.tsv");
    std::fs::write(&path, "q1\tfirst query\n\nq2\tsecond query\n").unwrap();
    let parsed = parse_queries(&path).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0], ("q1".to_string(), "first query".to_string()));
    assert_eq!(parsed[1], ("q2".to_string(), "second query".to_string()));
}

#[test]
fn parses_qrels_trec_and_beir() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qrels.tsv");
    std::fs::write(&path, "q1\t0\td1\t1\nq1\t0\td2\t0\nq2\td3\t2\n").unwrap();
    let parsed = parse_qrels(&path).unwrap();
    assert_eq!(parsed["q1"], set(&["d1"]));
    assert_eq!(parsed["q2"], set(&["d3"]));
}

#[test]
fn dcg_binary_relevance() {
    assert_eq!(dcg_at_k(&[true, false, true], 3), 1.0 + 0.5);
    assert_eq!(dcg_at_k(&[true, true], 1), 1.0);
}

#[test]
fn idcg_is_perfect_ranking() {
    assert_eq!(idcg_at_k(2, 3), 1.0 + 1.0 / (3.0f64).log2());
    assert_eq!(idcg_at_k(5, 2), 1.0 + 1.0 / (3.0f64).log2());
}

#[test]
fn ndcg_is_one_for_perfect_ranking() {
    let relevant = set(&["a", "b"]);
    let perfect = ranked(&["a", "b", "c"]);
    assert!((ndcg_at_k(&perfect, &relevant, 3) - 1.0).abs() < 1e-9);
}

#[test]
fn ndcg_zero_when_no_relevant() {
    assert_eq!(ndcg_at_k(&ranked(&["a"]), &set(&[]), 3), 0.0);
}

#[test]
fn recall_fraction_found() {
    let relevant = set(&["a", "b"]);
    assert!((recall_at_k(&ranked(&["a", "x"]), &relevant, 2) - 0.5).abs() < 1e-9);
    assert!((recall_at_k(&ranked(&["a", "b"]), &relevant, 2) - 1.0).abs() < 1e-9);
}

#[test]
fn mrr_uses_first_relevant_rank() {
    let relevant = set(&["b"]);
    assert!((reciprocal_rank(&ranked(&["x", "b"]), &relevant, 3) - 0.5).abs() < 1e-9);
    assert_eq!(reciprocal_rank(&ranked(&["x", "y"]), &relevant, 3), 0.0);
}

#[test]
fn dedupe_keeps_first_occurrence() {
    assert_eq!(
        dedupe(ranked(&["a", "b", "a", "c", "b"])),
        ranked(&["a", "b", "c"])
    );
}

#[test]
fn ndcg_never_exceeds_one_with_duplicate_chunks() {
    let relevant = set(&["a"]);
    let duped = ranked(&["a", "a", "b"]);
    let deduped = dedupe(duped);
    assert!(ndcg_at_k(&deduped, &relevant, 3) <= 1.0);
}

#[test]
fn qrels_header_line_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qrels.tsv");
    std::fs::write(&path, "query-id\tcorpus-id\tscore\nq1\td1\t1\n").unwrap();
    let parsed = parse_qrels(&path).unwrap();
    assert_eq!(parsed["q1"], set(&["d1"]));
}
