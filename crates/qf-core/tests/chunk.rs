use std::ops::Range;

use qf_core::chunk::{
    ChunkConfig, Span, Tokenize, TokenizeError, chunk_text, is_clause_start, split_units,
};

/// One token per whitespace-separated word, so tests need no model files.
struct Words;

impl Tokenize for Words {
    fn token_offsets(&self, text: &str) -> Result<Vec<Range<usize>>, TokenizeError> {
        let mut out = Vec::new();
        let mut start = None;
        for (i, c) in text.char_indices().chain([(text.len(), ' ')]) {
            match (c.is_whitespace(), start) {
                (false, None) => start = Some(i),
                (true, Some(s)) => {
                    out.push(s..i);
                    start = None;
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

fn config(max_tokens: usize, overlap: usize) -> ChunkConfig {
    ChunkConfig {
        max_tokens,
        overlap,
    }
}

fn slice(text: &str, span: Span) -> String {
    text.chars()
        .skip(span.start)
        .take(span.end - span.start)
        .collect()
}

fn chunk_strings(text: &str, cfg: ChunkConfig) -> Vec<String> {
    chunk_text(text, &Words, cfg)
        .unwrap()
        .into_iter()
        .map(|span| slice(text, span))
        .collect()
}

fn unit_texts(text: &str) -> Vec<&str> {
    split_units(text).into_iter().map(|r| &text[r]).collect()
}

#[test]
fn recognises_clause_numbers() {
    for line in [
        "12. Term",
        "12.3 Fees",
        "1.2.3. Notice",
        "4) Remedies",
        "(a) the",
        "(iv) any",
        "(2) each",
    ] {
        assert!(is_clause_start(line), "{line}");
    }
    for line in [
        "2023 was",
        "12.",
        "Section 5",
        "(abcde) x",
        "1.2345 x",
        "a. b",
        "(a)b",
        "12,3 x",
    ] {
        assert!(!is_clause_start(line), "{line}");
    }
}

#[test]
fn splits_on_paragraphs() {
    assert_eq!(
        unit_texts("First paragraph\nstill first.\n\nSecond paragraph."),
        ["First paragraph\nstill first.", "Second paragraph."]
    );
}

#[test]
fn splits_before_clause_lines_inside_a_paragraph() {
    assert_eq!(
        unit_texts(
            "Payment terms:\n(a) pay on time;\n(b) pay in full.\n12.3 Late fees apply\nat 2% a month."
        ),
        [
            "Payment terms:",
            "(a) pay on time;",
            "(b) pay in full.",
            "12.3 Late fees apply\nat 2% a month."
        ]
    );
}

#[test]
fn blank_text_has_no_units() {
    assert!(split_units("").is_empty());
    assert!(split_units("\n\n").is_empty());
}

#[test]
fn packs_whole_units_up_to_max_tokens() {
    let text = "1. one two three\n2. four five\n\nsix seven eight nine\n\nten";
    assert_eq!(
        chunk_strings(text, config(7, 0)),
        [
            "1. one two three\n2. four five",
            "six seven eight nine\n\nten"
        ]
    );
}

#[test]
fn spans_are_character_offsets() {
    let text = "Caf\u{e9} na\u{ef}ve r\u{e9}sum\u{e9}.\n\n\u{a7} 2 Fees due.";
    let spans = chunk_text(text, &Words, config(4, 0)).unwrap();
    assert_eq!(
        spans,
        [Span { start: 0, end: 18 }, Span { start: 20, end: 33 }]
    );
    assert_eq!(slice(text, spans[1]), "\u{a7} 2 Fees due.");
}

#[test]
fn empty_text_has_no_chunks() {
    assert!(
        chunk_text("", &Words, ChunkConfig::default())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn long_unit_is_cut_with_overlapping_windows() {
    let text = "Intro.\n\nw0 w1 w2 w3 w4 w5 w6 w7 w8 w9\n\nOutro.";
    assert_eq!(
        chunk_strings(text, config(4, 1)),
        [
            "Intro.",
            "w0 w1 w2 w3",
            "w3 w4 w5 w6",
            "w6 w7 w8 w9",
            "Outro."
        ]
    );
    assert_eq!(
        chunk_strings("w0 w1 w2 w3 w4", config(4, 2)),
        ["w0 w1 w2 w3", "w2 w3 w4"]
    );
}

#[test]
fn overlap_must_be_smaller_than_max_tokens() {
    assert!(chunk_text("a b c", &Words, config(4, 4)).is_err());
}
