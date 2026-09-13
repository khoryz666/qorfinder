use std::time::Duration;

use qorfinder::format::{format_hits, snippet};
use qorfinder::store::SearchHit;

fn hit(path: &str, text: &str) -> SearchHit {
    SearchHit {
        file_path: path.to_string(),
        chunk_index: 0,
        score: 0.9,
        text: text.to_string(),
    }
}

#[test]
fn snippet_keeps_short_text_unchanged() {
    assert_eq!(snippet("short text", 240), "short text");
}

#[test]
fn snippet_truncates_long_text_with_ellipsis() {
    let long = "a".repeat(300);
    let out = snippet(&long, 240);
    assert_eq!(out.chars().count(), 241);
    assert!(out.ends_with('…'));
}

#[test]
fn no_hits_message() {
    let out = format_hits("hello", &[], Duration::from_millis(10));
    assert!(out.contains("No results"));
    assert!(out.contains("hello"));
}

#[test]
fn hit_list_contains_path_and_score() {
    let hits = vec![hit("/docs/a.txt", "some text")];
    let out = format_hits("q", &hits, Duration::from_millis(10));
    assert!(out.contains("/docs/a.txt"));
    assert!(out.contains("some text"));
    assert!(out.contains("score 0.9000"));
}
