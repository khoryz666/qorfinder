use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::Once;

use crate::Format;

/// Files larger than this are skipped rather than loaded: every extractor
/// buffers the whole file, and no single legal document needs 50 MB.
pub const MAX_FILE_SIZE_BYTES: u64 = 50 * 1024 * 1024;

/// A PDF averaging fewer visible characters per page than this is treated
/// as a scan without a text layer. Even a sparse signature page of a real
/// contract has more text than this.
pub const MIN_PDF_CHARS_PER_PAGE: usize = 100;

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("cannot read file: {0}")]
    Io(#[from] std::io::Error),
    #[error("file is {size} bytes, over the {limit} byte limit")]
    TooLarge { size: u64, limit: u64 },
    #[error("cannot parse file: {0}")]
    Format(String),
    #[error("encrypted")]
    Encrypted,
    #[error("no text layer")]
    NoTextLayer,
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

thread_local! {
    /// Set while this thread runs a document parser.
    static IN_PARSER: Cell<bool> = const { Cell::new(false) };
}

/// Run a document parser, turning a panic into `None`. Parsers panic on
/// some malformed files (pdf-extract and its CMap parser did on 214 of 240
/// corrupted CUAD PDFs); the file is then skipped with a reason, so the
/// panic message would only print over the progress bars. The panic hook
/// stays quiet for such panics and reports every other panic as before.
fn catch_parser_panic<T>(parse: impl FnOnce() -> T) -> Option<T> {
    static QUIET_HOOK: Once = Once::new();
    QUIET_HOOK.call_once(|| {
        let report = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if !IN_PARSER.with(Cell::get) {
                report(info);
            }
        }));
    });
    IN_PARSER.with(|flag| flag.set(true));
    let result = panic::catch_unwind(AssertUnwindSafe(parse));
    IN_PARSER.with(|flag| flag.set(false));
    result.ok()
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
        Format::Pdf => extract_pdf(path),
        Format::Docx => extract_docx(path),
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

/// PDF text, one page at a time so every page's start offset is known.
/// Pages are separated by a paragraph break. The text comes from
/// pdf-extract, which decodes font encodings that lopdf's own
/// `extract_text` cannot (lopdf alone got no text from any CUAD PDF).
/// Both can panic on malformed files, so a panic becomes an ordinary
/// parse error.
fn extract_pdf(path: &Path) -> Result<Extracted, ParseError> {
    let bytes = std::fs::read(path)?;
    catch_parser_panic(|| extract_pdf_bytes(&bytes))
        .unwrap_or_else(|| Err(ParseError::Format("pdf parser panicked".into())))
}

fn extract_pdf_bytes(bytes: &[u8]) -> Result<Extracted, ParseError> {
    let mut doc = pdf_extract::Document::load_mem(bytes).map_err(|e| match e {
        pdf_extract::Error::Decryption(_) => ParseError::Encrypted,
        e => ParseError::Format(e.to_string()),
    })?;
    // Many PDFs are encrypted with an empty user password only to set
    // permissions; those open. Anything else needs a password we lack.
    if doc.is_encrypted() && doc.decrypt("").is_err() {
        return Err(ParseError::Encrypted);
    }
    let mut out = Extracted::default();
    let mut chars = 0;
    for page_no in doc.get_pages().into_keys() {
        if !out.text.is_empty() {
            out.text.push_str("\n\n");
            chars += 2;
        }
        out.pages.push((chars, page_no));
        // A page whose content stream cannot be decoded contributes no text
        // rather than failing the whole document.
        let mut text = String::new();
        let mut output = pdf_extract::PlainTextOutput::new(&mut text);
        if pdf_extract::output_doc_page(&doc, &mut output, page_no).is_err() {
            text.clear();
        }
        chars += text.chars().count();
        out.text.push_str(&text);
    }
    let visible = out.text.chars().filter(|c| !c.is_whitespace()).count();
    if visible < MIN_PDF_CHARS_PER_PAGE * out.pages.len().max(1) {
        return Err(ParseError::NoTextLayer);
    }
    Ok(out)
}

/// DOCX text: every non-empty paragraph, including those inside tables,
/// separated by a paragraph break. DOCX has no fixed pages, so none are
/// recorded.
fn extract_docx(path: &Path) -> Result<Extracted, ParseError> {
    let bytes = std::fs::read(path)?;
    let docx = catch_parser_panic(|| docx_rs::read_docx(&bytes))
        .ok_or_else(|| ParseError::Format("docx parser panicked".into()))?
        .map_err(|e| ParseError::Format(e.to_string()))?;
    let mut paragraphs = Vec::new();
    for child in &docx.document.children {
        match child {
            docx_rs::DocumentChild::Paragraph(p) => paragraphs.push(p.raw_text()),
            docx_rs::DocumentChild::Table(t) => table_paragraphs(t, &mut paragraphs),
            _ => {}
        }
    }
    paragraphs.retain(|p| !p.trim().is_empty());
    Ok(Extracted {
        text: paragraphs.join("\n\n"),
        pages: Vec::new(),
    })
}

fn table_paragraphs(table: &docx_rs::Table, out: &mut Vec<String>) {
    use docx_rs::{TableCellContent, TableChild, TableRowChild};

    for TableChild::TableRow(row) in &table.rows {
        for TableRowChild::TableCell(cell) in &row.cells {
            for content in &cell.children {
                match content {
                    TableCellContent::Paragraph(p) => out.push(p.raw_text()),
                    TableCellContent::Table(t) => table_paragraphs(t, out),
                    _ => {}
                }
            }
        }
    }
}
