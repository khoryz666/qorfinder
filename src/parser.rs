use std::io::Read;
use std::path::Path;

/// Files larger than this are skipped rather than fully loaded into memory —
/// a personal document corpus has no legitimate use for a single file this
/// large, and every parser here (`read_to_string`, `lopdf::Document::load`,
/// `read_to_end`) buffers the whole file up front.
pub const MAX_FILE_SIZE_BYTES: u64 = 50 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("unsupported file type: {0}")]
    Unsupported(String),
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {message}")]
    Format { path: String, message: String },
    #[error("{path} is {size} bytes, over the {limit} byte limit")]
    TooLarge { path: String, size: u64, limit: u64 },
}

pub fn is_supported(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("txt" | "md" | "markdown" | "pdf" | "docx")
    )
}

pub fn parse_file(path: &Path) -> Result<String, ParseError> {
    let meta = std::fs::metadata(path).map_err(|source| ParseError::Io {
        path: path.display().to_string(),
        source,
    })?;
    if meta.len() > MAX_FILE_SIZE_BYTES {
        return Err(ParseError::TooLarge {
            path: path.display().to_string(),
            size: meta.len(),
            limit: MAX_FILE_SIZE_BYTES,
        });
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "txt" | "md" | "markdown" => parse_plain_text(path),
        "pdf" => parse_pdf(path),
        "docx" => parse_docx(path),
        other => Err(ParseError::Unsupported(other.to_string())),
    }
}

fn parse_plain_text(path: &Path) -> Result<String, ParseError> {
    let mut contents = String::new();
    let mut file = std::fs::File::open(path).map_err(|source| ParseError::Io {
        path: path.display().to_string(),
        source,
    })?;
    file.read_to_string(&mut contents)
        .map_err(|source| ParseError::Io {
            path: path.display().to_string(),
            source,
        })?;
    Ok(contents)
}

fn parse_pdf(path: &Path) -> Result<String, ParseError> {
    let doc = lopdf::Document::load(path).map_err(|e| ParseError::Format {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;
    let mut out = String::new();
    for page_num in doc.get_pages().keys() {
        if let Ok(text) = doc.extract_text(&[*page_num]) {
            out.push_str(&text);
            out.push('\n');
        }
    }
    Ok(out)
}

fn parse_docx(path: &Path) -> Result<String, ParseError> {
    use docx_rs::{DocumentChild, ParagraphChild, RunChild, read_docx};

    let mut buf = Vec::new();
    let mut file = std::fs::File::open(path).map_err(|source| ParseError::Io {
        path: path.display().to_string(),
        source,
    })?;
    file.read_to_end(&mut buf)
        .map_err(|source| ParseError::Io {
            path: path.display().to_string(),
            source,
        })?;

    let docx = read_docx(&buf).map_err(|e| ParseError::Format {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;

    let mut out = String::new();
    for child in docx.document.children {
        if let DocumentChild::Paragraph(p) = child {
            for paragraph_child in p.children {
                if let ParagraphChild::Run(r) = paragraph_child {
                    for run_child in r.children {
                        if let RunChild::Text(t) = run_child {
                            out.push_str(&t.text);
                        }
                    }
                }
            }
            out.push('\n');
        }
    }
    Ok(out)
}
