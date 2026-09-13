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
