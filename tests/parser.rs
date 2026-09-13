use std::path::Path;

use qorfinder::parser::{MAX_FILE_SIZE_BYTES, ParseError, is_supported, parse_file};

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
