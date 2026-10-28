use unicode_normalization::char::canonical_combining_class;
use unicode_normalization::{UnicodeNormalization, is_nfc};

/// Text after normalisation, with a map back to the text it came from.
///
/// Chunks are cut from the normalised text, but evaluation compares them
/// with annotation offsets into the original file, so every normalised
/// character remembers the range of source characters it stands for.
#[derive(Debug, Clone, PartialEq)]
pub struct Normalized {
    pub text: String,
    /// `src[i]` is the `[start, end)` range of source characters behind
    /// normalised character `i`.
    src: Vec<(usize, usize)>,
}

impl Normalized {
    /// Source character range covered by normalised characters
    /// `[start, end)`. An empty range maps to an empty range.
    pub fn source_range(&self, start: usize, end: usize) -> (usize, usize) {
        if start >= end {
            let at = self
                .src
                .get(start)
                .map_or_else(|| self.src.last().map_or(0, |&(_, e)| e), |&(s, _)| s);
            return (at, at);
        }
        (self.src[start].0, self.src[end - 1].1)
    }
}

/// Normalise extracted text for chunking:
///
/// * Unicode NFC, so the same word always has the same code points;
/// * line endings become `\n`;
/// * a word split across lines with `-\n` is joined (`con-\ntract` becomes
///   `contract`);
/// * runs of spaces and tabs become one space, and whitespace at the start
///   and end of every line is dropped;
/// * one line break stays `\n`, and two or more (a paragraph break) become
///   exactly `\n\n`.
pub fn normalize(text: &str) -> Normalized {
    let chars = nfc_with_sources(text);
    let mut out = Normalized {
        text: String::with_capacity(text.len()),
        src: Vec::with_capacity(chars.len()),
    };
    let mut i = 0;
    while i < chars.len() {
        let (c, span) = chars[i];
        if !c.is_whitespace() {
            if c == '-' && is_hyphen_break(&out.text, &chars, i) {
                // Skip the hyphen and the line break after it.
                i += 1;
                while chars[i].0.is_whitespace() {
                    i += 1;
                }
                continue;
            }
            out.text.push(c);
            out.src.push(span);
            i += 1;
            continue;
        }
        // A whitespace run: emit one space, one newline or one paragraph
        // break, mapped to the first source character of each kind.
        let mut space = None;
        let mut newlines = Vec::new();
        while i < chars.len() && chars[i].0.is_whitespace() {
            let (c, span) = chars[i];
            if is_line_break(c) {
                // \r\n counts as a single line break.
                if !(c == '\n' && i > 0 && chars[i - 1].0 == '\r') {
                    newlines.push(span);
                }
            } else if space.is_none() {
                space = Some(span);
            }
            i += 1;
        }
        if out.text.is_empty() || i == chars.len() {
            continue; // trim the start and end of the text
        }
        match newlines.as_slice() {
            [] => out.push(' ', space.expect("whitespace run without a char")),
            [only] => out.push('\n', *only),
            [first, second, ..] => {
                out.push('\n', *first);
                out.push('\n', *second);
            }
        }
    }
    out
}

impl Normalized {
    fn push(&mut self, c: char, span: (usize, usize)) {
        self.text.push(c);
        self.src.push(span);
    }
}

fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

/// A `-` at `chars[i]` ends a line in the middle of a word: a letter comes
/// before it, a single line break after it, then a lowercase letter.
fn is_hyphen_break(emitted: &str, chars: &[(char, (usize, usize))], i: usize) -> bool {
    if !emitted.chars().next_back().is_some_and(char::is_alphabetic) {
        return false;
    }
    let mut j = i + 1;
    let mut breaks = 0;
    while j < chars.len() && chars[j].0.is_whitespace() {
        if is_line_break(chars[j].0) && !(chars[j].0 == '\n' && chars[j - 1].0 == '\r') {
            breaks += 1;
        }
        j += 1;
    }
    breaks == 1 && j < chars.len() && chars[j].0.is_lowercase()
}

/// NFC-normalise `text`, pairing every output char with its source range.
/// Composition only happens inside a starter and the combining marks after
/// it, so each such cluster is normalised on its own and all of its output
/// chars map to the whole cluster.
fn nfc_with_sources(text: &str) -> Vec<(char, (usize, usize))> {
    if is_nfc(text) {
        return text
            .chars()
            .enumerate()
            .map(|(i, c)| (c, (i, i + 1)))
            .collect();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::with_capacity(chars.len());
    let mut start = 0;
    while start < chars.len() {
        let mut end = start + 1;
        while end < chars.len() && canonical_combining_class(chars[end]) != 0 {
            end += 1;
        }
        let cluster: String = chars[start..end].iter().collect();
        out.extend(cluster.nfc().map(|c| (c, (start, end))));
        start = end;
    }
    out
}
