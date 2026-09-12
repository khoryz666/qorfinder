use std::io::Read;
use std::path::Path;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_expected_extensions() {
        for ext in ["txt", "md", "markdown", "pdf", "docx", "TXT", "PDF"] {
            assert!(is_supported(Path::new(&format!("file.{ext}"))));
        }
        for ext in ["png", "xlsx", "exe", ""] {
            assert!(!is_supported(Path::new(&format!("file.{ext}"))));
        }
    }

    #[test]
    fn parses_plain_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("note.txt");
        std::fs::write(&path, "hello qorfinder\nsecond line").unwrap();
        assert_eq!(parse_file(&path).unwrap(), "hello qorfinder\nsecond line");
    }

    #[test]
    fn rejects_unsupported_extensions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.png");
        std::fs::write(&path, "not an image").unwrap();
        assert!(matches!(parse_file(&path), Err(ParseError::Unsupported(_))));
    }

    #[test]
    fn missing_file_is_io_error() {
        let path = Path::new("/nonexistent/qorfinder/nope.txt");
        assert!(matches!(parse_file(path), Err(ParseError::Io { .. })));
    }
}
