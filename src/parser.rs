use std::io::Read;
use std::path::Path;

/// Files larger than this are skipped rather than fully loaded into memory —
/// a personal document corpus has no legitimate use for a single file this
/// large, and every parser here (`read_to_string`, `lopdf::Document::load`,
/// `read_to_end`) buffers the whole file up front.
const MAX_FILE_SIZE_BYTES: u64 = 50 * 1024 * 1024;

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

    #[test]
    fn rejects_file_over_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.txt");
        // A sparse file reports the target length without writing that much
        // data, so this stays fast regardless of the limit's size.
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_FILE_SIZE_BYTES + 1).unwrap();
        assert!(matches!(
            parse_file(&path),
            Err(ParseError::TooLarge { .. })
        ));
    }

    #[test]
    fn parses_pdf_text() {
        use lopdf::content::{Content, Operation};
        use lopdf::{Document, Object, Stream, dictionary};

        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Courier",
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
        });
        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 24.into()]),
                Operation::new("Td", vec![50.into(), 700.into()]),
                Operation::new("Tj", vec![Object::string_literal("HelloPdfFixture")]),
                Operation::new("ET", vec![]),
            ],
        };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
        });
        let pages = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        };
        doc.objects.insert(pages_id, Object::Dictionary(pages));
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.pdf");
        doc.save(&path).unwrap();

        let text = parse_file(&path).unwrap();
        assert!(
            text.contains("HelloPdfFixture"),
            "extracted text was {text:?}"
        );
    }

    #[test]
    fn parses_docx_text() {
        use docx_rs::{Docx, Paragraph, Run};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.docx");
        let file = std::fs::File::create(&path).unwrap();
        Docx::new()
            .add_paragraph(Paragraph::new().add_run(Run::new().add_text("HelloDocxFixture")))
            .build()
            .pack(file)
            .unwrap();

        let text = parse_file(&path).unwrap();
        assert!(
            text.contains("HelloDocxFixture"),
            "extracted text was {text:?}"
        );
    }
}
