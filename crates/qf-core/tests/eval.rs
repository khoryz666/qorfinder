// Slices of one span are what the relevance rule takes.
#![allow(clippy::single_range_in_vec_init)]

use std::path::PathBuf;

use qf_core::eval::{
    Query, Span, is_relevant, judge, read_queries, read_spans, write_qrels, write_run,
};
use qf_core::{Chunk, Document, Format, Hit};

#[test]
fn half_a_chunk_inside_a_span_is_relevant() {
    // Chunk of 10 characters, [100, 110).
    assert!(is_relevant(100..110, &[105..200])); // 5 of 10 inside
    assert!(!is_relevant(100..110, &[106..200])); // 4 of 10
    assert!(is_relevant(100..110, &[0..300])); // all inside
    assert!(!is_relevant(100..110, &[0..100, 110..120])); // touching only
}

#[test]
fn a_chunk_holding_a_whole_span_is_relevant() {
    // A 2-character span inside a 100-character chunk.
    assert!(is_relevant(0..100, &[40..42]));
    assert!(!is_relevant(0..100, &[90..120]));
}

#[test]
fn overlapping_spans_are_not_counted_twice() {
    // Chunk [10, 110), no span wholly inside it. Merged, characters 10..40
    // and 105..110 are inside: 35 of 100, though the spans add up to 60.
    assert!(!is_relevant(10..110, &[0..35, 5..40, 105..120]));
    assert!(is_relevant(10..110, &[0..40, 30..70])); // 10..70: 60 of 100
}

fn document(id: u32, name: &str) -> Document {
    Document {
        id,
        path: PathBuf::from("/sample").join(name),
        format: Format::Txt,
        hash: [0; 32],
        mtime: 0,
    }
}

fn chunk(id: u32, doc_id: u32, range: std::ops::Range<usize>) -> Chunk {
    Chunk {
        id,
        doc_id,
        ord: id,
        text: String::new(),
        char_start: range.start,
        char_end: range.end,
        page: None,
    }
}

fn query(id: &str, category: &str) -> Query {
    Query {
        id: id.into(),
        category: category.into(),
        kind: "matched".into(),
        text: "text".into(),
    }
}

fn span(category: &str, file: &str, range: std::ops::Range<usize>) -> Span {
    Span {
        category: category.into(),
        file: file.into(),
        start: range.start,
        end: range.end,
    }
}

#[test]
fn judgements_follow_category_and_file() {
    let documents = [document(0, "a.txt"), document(1, "b.txt")];
    let chunks = [
        chunk(0, 0, 0..100),
        chunk(1, 0, 100..200),
        chunk(2, 1, 0..100),
        chunk(3, 1, 100..200),
    ];
    let spans = [
        span("Non-Compete", "a.txt", 120..190),
        span("Non-Compete", "b.txt", 10..20),
        span("Warranty Duration", "a.txt", 0..100),
        span("Non-Compete", "not-indexed.txt", 0..1000),
    ];
    let queries = [
        query("Q01m", "Non-Compete"),
        query("Q02m", "Warranty Duration"),
        query("Q03m", "Liquidated Damages"),
    ];
    let qrels = judge(&queries, &spans, &documents, &chunks);
    assert_eq!(qrels["Q01m"].iter().copied().collect::<Vec<_>>(), [1, 2]);
    assert_eq!(qrels["Q02m"].iter().copied().collect::<Vec<_>>(), [0]);
    assert!(qrels["Q03m"].is_empty());

    let mut out = Vec::new();
    write_qrels(&mut out, &queries, &qrels).unwrap();
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "Q01m 0 1 1\nQ01m 0 2 1\nQ02m 0 0 1\n"
    );
}

#[test]
fn runs_are_written_in_trec_format() {
    let runs = vec![(
        "Q01m".to_string(),
        vec![
            Hit {
                chunk_id: 7,
                score: 0.5,
            },
            Hit {
                chunk_id: 3,
                score: 0.25,
            },
        ],
    )];
    let mut out = Vec::new();
    write_run(&mut out, "bm25", &runs).unwrap();
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "Q01m Q0 7 1 0.500000000 bm25\nQ01m Q0 3 2 0.250000000 bm25\n"
    );
}

#[test]
fn queries_and_spans_are_read_from_their_files() {
    let dir = tempfile::tempdir().unwrap();
    let queries = dir.path().join("queries.jsonl");
    std::fs::write(
        &queries,
        "{\"id\":\"Q01m\",\"category\":\"Non-Compete\",\"type\":\"matched\",\"text\":\"no rival business\"}\n\n",
    )
    .unwrap();
    assert_eq!(
        read_queries(&queries).unwrap(),
        [Query {
            id: "Q01m".into(),
            category: "Non-Compete".into(),
            kind: "matched".into(),
            text: "no rival business".into(),
        }]
    );
    std::fs::write(
        &queries,
        "{\"id\":\"Q01m\",\"category\":\"C\",\"type\":\"matched\",\"text\":\"\"}\n",
    )
    .unwrap();
    assert!(read_queries(&queries).is_err());

    let spans = dir.path().join("spans.tsv");
    std::fs::write(&spans, "Non-Compete\ta b.txt\t10\t20\n").unwrap();
    assert_eq!(
        read_spans(&spans).unwrap(),
        [span("Non-Compete", "a b.txt", 10..20)]
    );
    std::fs::write(&spans, "Non-Compete\ta.txt\t20\t10\n").unwrap();
    let err = read_spans(&spans).unwrap_err();
    assert!(err.to_string().contains("line 1"), "{err}");
}
