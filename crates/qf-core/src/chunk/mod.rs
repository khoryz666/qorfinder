//! Clause-aware chunking of normalised document text.

mod units;

use std::ops::Range;

pub use units::{is_clause_start, split_units};

pub type TokenizeError = Box<dyn std::error::Error + Send + Sync>;

/// Splits text into model tokens. Chunk sizes are counted with the same
/// tokenizer the embedding model uses, so a chunk never silently overflows
/// the model's input.
pub trait Tokenize {
    /// Byte range in `text` of every token, in order, without special
    /// tokens such as `[CLS]`.
    fn token_offsets(&self, text: &str) -> Result<Vec<Range<usize>>, TokenizeError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    /// Most tokens in one chunk. 200 leaves room under a 256-token model
    /// for the special tokens and tokenizer differences.
    pub max_tokens: usize,
    /// Tokens shared by consecutive windows when one unit is too long.
    pub overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 200,
            overlap: 40,
        }
    }
}

/// A chunk as a `[start, end)` range of characters (not bytes) in the text
/// it was cut from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Cut `text` into chunks of at most `config.max_tokens` tokens. Whole
/// units (paragraphs and numbered clauses) are packed together in order
/// until the next one would not fit, so chunk boundaries fall between units.
/// A unit longer than `max_tokens` on its own is cut with a sliding window
/// of `max_tokens` tokens, consecutive windows sharing `overlap` tokens.
pub fn chunk_text(
    text: &str,
    tokenizer: &impl Tokenize,
    config: ChunkConfig,
) -> Result<Vec<Span>, TokenizeError> {
    if config.overlap >= config.max_tokens {
        return Err("chunk overlap must be smaller than max_tokens".into());
    }
    let mut chunks: Vec<Range<usize>> = Vec::new();
    let mut current: Option<(Range<usize>, usize)> = None;
    for unit in split_units(text) {
        let offsets = tokenizer.token_offsets(&text[unit.clone()])?;
        let tokens = offsets.len();
        if tokens > config.max_tokens {
            chunks.extend(current.take().map(|(range, _)| range));
            chunks.extend(
                windows(&offsets, config)
                    .into_iter()
                    .map(|w| unit.start + w.start..unit.start + w.end),
            );
            continue;
        }
        if let Some((range, used)) = &mut current {
            if *used + tokens <= config.max_tokens {
                range.end = unit.end;
                *used += tokens;
                continue;
            }
            chunks.push(range.clone());
        }
        current = Some((unit, tokens));
    }
    chunks.extend(current.map(|(range, _)| range));
    Ok(to_char_spans(text, &chunks))
}

/// Byte ranges of `max_tokens`-token windows over a unit's token
/// `offsets`, each starting `max_tokens - overlap` tokens after the one
/// before, until a window reaches the last token.
fn windows(offsets: &[Range<usize>], config: ChunkConfig) -> Vec<Range<usize>> {
    let step = config.max_tokens - config.overlap;
    let mut out = Vec::new();
    let mut first = 0;
    loop {
        let end = (first + config.max_tokens).min(offsets.len());
        out.push(offsets[first].start..offsets[end - 1].end);
        if end == offsets.len() {
            return out;
        }
        first += step;
    }
}

/// Convert byte ranges into character ranges.
fn to_char_spans(text: &str, ranges: &[Range<usize>]) -> Vec<Span> {
    let starts: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
    let char_at = |byte: usize| starts.partition_point(|&b| b < byte);
    ranges
        .iter()
        .map(|r| Span {
            start: char_at(r.start),
            end: char_at(r.end),
        })
        .collect()
}
