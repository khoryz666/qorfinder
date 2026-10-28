use std::path::Path;

use crate::Format;

/// Files larger than this are skipped rather than loaded: every extractor
/// buffers the whole file, and no single legal document needs 50 MB.
pub const MAX_FILE_SIZE_BYTES: u64 = 50 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("cannot read file: {0}")]
    Io(#[from] std::io::Error),
    #[error("file is {size} bytes, over the {limit} byte limit")]
    TooLarge { size: u64, limit: u64 },
    #[error("cannot parse file: {0}")]
    Format(String),
}

/// Text pulled out of one file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Extracted {
    pub text: String,
    /// `(char_offset, page_no)` for the first character of every page, in
    /// order. Empty for formats without pages.
    pub pages: Vec<(usize, u32)>,
}

impl Extracted {
    /// 1-based page holding the character at `char_offset`, if the format
    /// has pages.
    pub fn page_at(&self, char_offset: usize) -> Option<u32> {
        let i = self
            .pages
            .partition_point(|&(start, _)| start <= char_offset);
        i.checked_sub(1).map(|i| self.pages[i].1)
    }
}

/// Extract the text of `path` with the extractor for `format`.
pub fn extract(path: &Path, format: Format) -> Result<Extracted, ParseError> {
    let size = std::fs::metadata(path)?.len();
    if size > MAX_FILE_SIZE_BYTES {
        return Err(ParseError::TooLarge {
            size,
            limit: MAX_FILE_SIZE_BYTES,
        });
    }
    match format {
        Format::Txt => extract_txt(path),
        Format::Pdf | Format::Docx => Err(ParseError::Format(format!(
            "{} extraction is not supported yet",
            format.as_str()
        ))),
    }
}

/// Plain text is decoded as UTF-8; invalid bytes become U+FFFD instead of
/// failing the file. A leading byte-order mark is dropped.
fn extract_txt(path: &Path) -> Result<Extracted, ParseError> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text).to_string();
    Ok(Extracted {
        text,
        pages: Vec::new(),
    })
}
