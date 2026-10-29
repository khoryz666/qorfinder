use std::ops::Range;

use qf_core::chunk::{
    ChunkConfig, ModelTokenizer, Span, Tokenize, TokenizeError, chunk_text, is_clause_start,
    split_units,
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
        "a. the Licensee",
        "B. Recitals",
        "iv. any",
        "Section 5. Term",
        "SECTION 12.3 Fees",
        "Section 5",
        "ARTICLE V",
        "Article IV: Payment",
        "ARTICLE 5",
    ] {
        assert!(is_clause_start(line), "{line}");
    }
    for line in [
        "2023 was",
        "12.",
        "(abcde) x",
        "1.2345 x",
        "a.b",
        "(a)b",
        "12,3 x",
        "Sections 5 and 6",
        "Article of association",
        "Section",
        "Mr. Smith",
        "e.g. this",
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

/// The "done when" check for the chunker: putting the chunks back together
/// recovers every character of the source. Chunks are in order, every
/// non-whitespace character falls inside one, and overlaps only repeat
/// text that is already there.
#[test]
fn rejoined_chunks_recover_the_source() {
    let mut text = String::new();
    for clause in 1..=12 {
        text.push_str(&format!("{clause}. Clause {clause} heading\n"));
        for sub in ["a", "b", "c"] {
            text.push_str(&format!("({sub}) The party shall"));
            for word in 0..(clause * 3) {
                text.push_str(&format!(" w{clause}{sub}{word}"));
            }
            text.push_str(";\n");
        }
        text.push('\n');
    }
    let text = text.trim_end();
    let chars: Vec<char> = text.chars().collect();

    for cfg in [config(20, 5), config(37, 0), config(200, 40)] {
        let spans = chunk_text(text, &Words, cfg).unwrap();
        let mut rebuilt = String::new();
        let mut covered_to = 0;
        let mut prev_start = 0;
        for span in &spans {
            assert!(span.start < span.end && span.end <= chars.len());
            assert!(span.start >= prev_start, "chunks out of order");
            prev_start = span.start;
            if span.start > covered_to {
                let gap: String = chars[covered_to..span.start].iter().collect();
                assert!(gap.trim().is_empty(), "uncovered text {gap:?}");
                rebuilt.push_str(&gap);
            }
            if span.end > covered_to {
                rebuilt.extend(&chars[covered_to.max(span.start)..span.end]);
                covered_to = span.end;
            }
            let words = chars[span.start..span.end]
                .iter()
                .collect::<String>()
                .split_whitespace()
                .count();
            assert!(words <= cfg.max_tokens, "{words} tokens in one chunk");
        }
        assert_eq!(rebuilt, text, "config {cfg:?}");
    }
}

fn model_tokenizer() -> ModelTokenizer {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/all-MiniLM-L6-v2/tokenizer.json");
    ModelTokenizer::from_file(&path).expect("run scripts/fetch_model.py first")
}

#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn model_tokenizer_offsets_point_into_the_text() {
    let text = "The Licensee shall pay royalties of 5% quarterly.";
    let offsets = model_tokenizer().token_offsets(text).unwrap();
    let tokens: Vec<&str> = offsets.iter().map(|r| &text[r.clone()]).collect();
    assert_eq!(tokens.first(), Some(&"The"));
    assert!(tokens.contains(&"royalties"));
    assert_eq!(tokens.last(), Some(&"."));
}

#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn model_tokenizer_is_not_truncated() {
    // tokenizer.json ships with truncation at 128 tokens; counting must not
    // stop there.
    let text = "word ".repeat(300);
    assert_eq!(model_tokenizer().token_offsets(&text).unwrap().len(), 300);
}
