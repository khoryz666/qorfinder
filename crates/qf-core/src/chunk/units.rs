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
/// `(a)`, `(iv)`, `(2)`, `a.` or `iv.` followed by whitespace (`v.` only
/// before a word in capitals), or a `Section 5`/`ARTICLE V` heading (any
/// case, number or roman numeral) not followed by a lowercase word.
pub fn is_clause_start(line: &str) -> bool {
    let mut words = line.split_whitespace();
    let Some(first) = words.next() else {
        return false;
    };
    if first.eq_ignore_ascii_case("section") || first.eq_ignore_ascii_case("article") {
        // A heading's title, if any, starts with a capital or punctuation;
        // "Section 5 of the Act" opening a wrapped line is a reference.
        return words.next().is_some_and(is_heading_number)
            && words
                .next()
                .is_none_or(|title| !title.starts_with(|c: char| c.is_lowercase()));
    }
    let Some((label, rest)) = line.split_once(char::is_whitespace) else {
        return false;
    };
    if let Some(inner) = label.strip_prefix('(').and_then(|l| l.strip_suffix(')')) {
        return (1..=4).contains(&inner.len()) && inner.chars().all(|c| c.is_ascii_alphanumeric());
    }
    // "v." counts only before a heading in capitals ("V. TERMINATION"): at
    // the start of a wrapped line it is far more often "versus" in a case
    // citation ("v. Jones") than a clause label.
    if let Some(letters) = label.strip_suffix('.')
        && (is_single_letter(letters) || is_roman(letters))
        && (!letters.eq_ignore_ascii_case("v") || starts_with_capitals_word(rest))
    {
        return true;
    }
    let (number, closed) = match label.strip_suffix(')') {
        Some(number) => (number, true),
        None => (label.strip_suffix('.').unwrap_or(label), false),
    };
    // A bare number such as "2023 " is ordinary text, not a clause label.
    is_decimal(number) && (closed || label.contains('.'))
}

/// `5`, `12.3` or `1.2.3`: one to three digits per part.
fn is_decimal(s: &str) -> bool {
    s.split('.')
        .all(|p| (1..=3).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()))
}

/// The number after `Section` or `Article`, with an optional `.` or `:`.
fn is_heading_number(word: &str) -> bool {
    let word = word.trim_end_matches(['.', ':']);
    is_decimal(word) || is_roman(word)
}

/// Whether the first word of `s` has two or more letters, all capitals.
fn starts_with_capitals_word(s: &str) -> bool {
    let word = s.split_whitespace().next().unwrap_or("");
    let letters = word.chars().filter(|c| c.is_alphabetic());
    letters.clone().count() >= 2 && letters.into_iter().all(char::is_uppercase)
}

fn is_single_letter(s: &str) -> bool {
    s.len() == 1 && s.bytes().all(|b| b.is_ascii_alphabetic())
}

/// A short roman numeral in one case (`iv`, `XII`), as used for clauses.
fn is_roman(s: &str) -> bool {
    (1..=5).contains(&s.len())
        && (s.bytes().all(|b| b"ivxl".contains(&b)) || s.bytes().all(|b| b"IVXL".contains(&b)))
}
