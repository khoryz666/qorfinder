//! One file in, chunk-sized passages out: extract, normalise, chunk.

use std::path::Path;

use crate::Format;
use crate::chunk::{ChunkConfig, Tokenize, TokenizeError, chunk_text};
use crate::ingest::{ParseError, extract, normalize, normalize_pdf};

#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error("tokenizer failed: {0}")]
    Tokenize(TokenizeError),
    #[error("no text")]
    Empty,
}

/// A chunk of one document before it is numbered and stored.
#[derive(Debug, Clone, PartialEq)]
pub struct Passage {
    /// Normalised chunk text: what is embedded and shown to the user.
    pub text: String,
    /// `[char_start, char_end)` in characters of the extracted text, before
    /// normalisation. For a TXT file that is the file itself, so these line
    /// up with annotation offsets.
    pub char_start: usize,
    pub char_end: usize,
    /// 1-based page of the first character, for formats with pages.
    pub page: Option<u32>,
}

/// Extract, normalise and chunk one file.
pub fn prepare(
    path: &Path,
    format: Format,
    tokenizer: &impl Tokenize,
    config: ChunkConfig,
) -> Result<Vec<Passage>, PrepareError> {
    let extracted = extract(path, format)?;
    // Only PDF layout breaks words across lines with a hyphen.
    let normalized = match format {
        Format::Pdf => normalize_pdf(&extracted.text),
        Format::Docx | Format::Txt => normalize(&extracted.text),
    };
    let spans = chunk_text(&normalized.text, tokenizer, config).map_err(PrepareError::Tokenize)?;
    if spans.is_empty() {
        return Err(PrepareError::Empty);
    }
    let chars: Vec<char> = normalized.text.chars().collect();
    Ok(spans
        .into_iter()
        .map(|span| {
            let (char_start, char_end) = normalized.source_range(span.start, span.end);
            Passage {
                text: chars[span.start..span.end].iter().collect(),
                char_start,
                char_end,
                page: extracted.page_at(char_start),
            }
        })
        .collect())
}
