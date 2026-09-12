use std::time::Duration;

use crate::store::SearchHit;

/// Trim `text` to at most `max_chars` characters, adding an ellipsis when
/// truncated.
pub fn snippet(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(max_chars).collect();
    format!("{head}…")
}

pub fn format_hits(query: &str, hits: &[SearchHit], elapsed: Duration) -> String {
    if hits.is_empty() {
        return format!("No results for \"{query}\" ({} ms)\n", elapsed.as_millis());
    }
    let mut out = format!(
        "{} result(s) for \"{query}\" ({} ms):\n\n",
        hits.len(),
        elapsed.as_millis()
    );
    for (i, hit) in hits.iter().enumerate() {
        out.push_str(&format!(
            "{}. {} [chunk {}] (score {:.4})\n   {}\n\n",
            i + 1,
            hit.file_path,
            hit.chunk_index,
            hit.score,
            snippet(&hit.text, 240)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
