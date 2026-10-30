//! Text helpers for showing results.

/// `text` on one line with runs of whitespace collapsed, cut to at most
/// `max_chars` characters plus an ellipsis when it is longer. The cut falls
/// between words when the kept text has a space to cut at.
pub fn snippet(text: &str, max_chars: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        return flat;
    }
    // One character more than fits, so a word ending exactly at the limit
    // is kept; then back up to the last space, unless there is none.
    let head: String = flat.chars().take(max_chars + 1).collect();
    let cut = match head.rfind(' ') {
        Some(space) if space > 0 => &head[..space],
        _ => head
            .char_indices()
            .nth(max_chars)
            .map_or(head.as_str(), |(i, _)| &head[..i]),
    };
    format!("{}…", cut.trim_end())
}

/// `text` broken into lines of at most `width` characters at spaces, with
/// runs of whitespace collapsed. A word longer than `width` gets a line of
/// its own rather than being split.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let needed = line.chars().count() + 1 + word.chars().count();
        if !line.is_empty() && needed > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}
