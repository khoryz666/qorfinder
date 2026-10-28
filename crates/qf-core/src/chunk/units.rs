use std::ops::Range;

/// Split normalised text into units, the pieces a chunk boundary may fall
/// between: paragraphs (separated by `\n\n`), further split before every
/// line that starts with a clause number. Returns byte ranges into `text`,
/// in order, each holding non-blank text.
pub fn split_units(text: &str) -> Vec<Range<usize>> {
    let mut units = Vec::new();
    let mut unit_start = None;
    let mut line_start = 0;
    for line in text.split_inclusive('\n') {
        let line_end = line_start + line.len();
        let body = line.trim_end_matches('\n');
        let blank = body.trim().is_empty();
        if let Some(start) = unit_start
            && (blank || is_clause_start(body))
        {
            units.push(start..trim_end(text, line_start));
            unit_start = None;
        }
        if unit_start.is_none() && !blank {
            unit_start = Some(line_start);
        }
        line_start = line_end;
    }
    if let Some(start) = unit_start {
        units.push(start..trim_end(text, text.len()));
    }
    units
}

/// `end` moved back over trailing whitespace.
fn trim_end(text: &str, end: usize) -> usize {
    text[..end].trim_end().len()
}

/// Whether a line opens a numbered clause: `12.`, `12.3`, `12.3.1`, `4)`,
/// `(a)`, `(iv)` or `(2)`, followed by whitespace.
pub fn is_clause_start(line: &str) -> bool {
    let Some((label, _rest)) = line.split_once(char::is_whitespace) else {
        return false;
    };
    if let Some(inner) = label.strip_prefix('(').and_then(|l| l.strip_suffix(')')) {
        return (1..=4).contains(&inner.len()) && inner.chars().all(|c| c.is_ascii_alphanumeric());
    }
    let (number, closed) = match label.strip_suffix(')') {
        Some(number) => (number, true),
        None => (label.strip_suffix('.').unwrap_or(label), false),
    };
    let mut parts = number.split('.');
    let all_numeric = parts
        .by_ref()
        .all(|p| (1..=3).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()));
    // A bare number such as "2023 " is ordinary text, not a clause label.
    all_numeric && (closed || label.contains('.'))
}
