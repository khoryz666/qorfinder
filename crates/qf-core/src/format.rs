//! Text helpers for showing results.

/// `text` on one line with runs of whitespace collapsed, cut to at most
/// `max_chars` characters plus an ellipsis when it is longer.
pub fn snippet(text: &str, max_chars: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        return flat;
    }
    let head: String = flat.chars().take(max_chars).collect();
    format!("{}…", head.trim_end())
}
