use std::fs;

use qf_core::Format;
use qf_core::ingest::{Extracted, MAX_FILE_SIZE_BYTES, ParseError, extract, hash_file, walk};

#[test]
fn walk_finds_supported_files_recursively_in_path_order() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("sub")).unwrap();
    fs::write(dir.path().join("b.txt"), "hello").unwrap();
    fs::write(dir.path().join("a.PDF"), "%PDF").unwrap();
    fs::write(dir.path().join("sub/c.docx"), "PK").unwrap();
    fs::write(dir.path().join("notes.md"), "not indexed").unwrap();

    let (found, skipped) = walk(dir.path());
    assert_eq!(skipped.len(), 1);
    assert!(skipped[0].path.ends_with("notes.md"));
    assert_eq!(skipped[0].reason, "unsupported file type");
    let names: Vec<_> = found
        .iter()
        .map(|f| f.path.strip_prefix(dir.path()).unwrap().to_path_buf())
        .collect();
    assert_eq!(
        names,
        ["a.PDF", "b.txt", "sub/c.docx"].map(std::path::PathBuf::from)
    );
    assert_eq!(
        found.iter().map(|f| f.format).collect::<Vec<_>>(),
        [Format::Pdf, Format::Txt, Format::Docx]
    );
    assert_eq!(found[1].size, 5);
    assert!(found[1].mtime > 0);
}

#[test]
fn walk_reports_missing_root_as_skipped() {
    let (found, skipped) = walk(std::path::Path::new("/nonexistent/qorfinder"));
    assert!(found.is_empty());
    assert_eq!(skipped.len(), 1);
}

#[test]
fn hash_file_is_sha256_of_contents() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.txt");
    fs::write(&path, "abc").unwrap();
    let hex: String = hash_file(&path)
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        hex,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn extracts_txt_as_is() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.txt");
    fs::write(&path, "\u{feff}1. Term\n\nThe  term is  one year.").unwrap();
    let out = extract(&path, Format::Txt).unwrap();
    assert_eq!(out.text, "1. Term\n\nThe  term is  one year.");
    assert!(out.pages.is_empty());
}

#[test]
fn txt_with_invalid_utf8_is_still_extracted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.txt");
    fs::write(&path, b"caf\xe9 owner").unwrap();
    assert_eq!(
        extract(&path, Format::Txt).unwrap().text,
        "caf\u{fffd} owner"
    );
}

#[test]
fn missing_file_is_io_error() {
    let path = std::path::Path::new("/nonexistent/qorfinder/a.txt");
    assert!(matches!(extract(path, Format::Txt), Err(ParseError::Io(_))));
}

#[test]
fn file_over_size_limit_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge.txt");
    // A sparse file reports the length without writing the data.
    fs::File::create(&path)
        .unwrap()
        .set_len(MAX_FILE_SIZE_BYTES + 1)
        .unwrap();
    assert!(matches!(
        extract(&path, Format::Txt),
        Err(ParseError::TooLarge { .. })
    ));
}

#[test]
fn page_at_maps_offsets_to_pages() {
    let extracted = Extracted {
        text: "aaaa\n\nbbbb".into(),
        pages: vec![(0, 1), (6, 2)],
    };
    assert_eq!(extracted.page_at(0), Some(1));
    assert_eq!(extracted.page_at(5), Some(1));
    assert_eq!(extracted.page_at(6), Some(2));
    assert_eq!(extracted.page_at(100), Some(2));
    assert_eq!(Extracted::default().page_at(0), None);
}

/// A one-page-per-string PDF built in memory, so tests need no fixture files.
fn write_pdf(path: &std::path::Path, pages: &[&str]) {
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
    let mut kids = Vec::new();
    for text in pages {
        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 10.into()]),
                Operation::new("Td", vec![50.into(), 700.into()]),
                Operation::new("Tj", vec![Object::string_literal(*text)]),
                Operation::new("ET", vec![]),
            ],
        };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
        });
        kids.push(page_id.into());
    }
    let count = kids.len() as i64;
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => count,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        }),
    );
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    doc.save(path).unwrap();
}

// Each page needs at least MIN_PDF_CHARS_PER_PAGE visible characters, or it
// looks like a scan.
const PAGE_ONE: &str = "1. Confidentiality. The Receiving Party shall keep all Confidential \
    Information secret and shall not disclose it to any third party without prior consent.";
const PAGE_TWO: &str = "2. Term. This Agreement remains in force for two years from the \
    Effective Date unless terminated earlier by either party on thirty days written notice.";

#[test]
fn extracts_pdf_text_with_page_offsets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.pdf");
    write_pdf(&path, &[PAGE_ONE, PAGE_TWO]);

    let out = extract(&path, Format::Pdf).unwrap();
    assert_eq!(out.pages.len(), 2);
    assert_eq!(out.pages[0], (0, 1));
    let (second_start, second_no) = out.pages[1];
    assert_eq!(second_no, 2);
    let second: String = out.text.chars().skip(second_start).collect();
    assert!(second.contains("2. Term."), "page two was {second:?}");
    let first: String = out.text.chars().take(second_start).collect();
    assert!(
        first.contains("1. Confidentiality."),
        "page one was {first:?}"
    );
    assert!(!first.contains("2. Term."));
}

#[test]
fn corrupt_pdf_is_a_parse_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.pdf");
    fs::write(&path, b"%PDF-1.5\nthis is not really a pdf").unwrap();
    assert!(matches!(
        extract(&path, Format::Pdf),
        Err(ParseError::Format(_))
    ));
}

/// A CUAD PDF (CC BY 4.0) with 40 bytes overwritten, on which pdf-extract
/// panics. The panic must end as a parse error for that file alone.
#[test]
fn pdf_that_panics_the_parser_is_a_parse_error() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parser_panic.pdf");
    let err = extract(&path, Format::Pdf).unwrap_err();
    assert!(
        matches!(&err, ParseError::Format(m) if m == "pdf parser panicked"),
        "got {err:?}"
    );
}

#[test]
fn pdf_needing_a_password_is_reported_as_encrypted() {
    use lopdf::{Document, Object, dictionary};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("locked.pdf");
    write_pdf(&path, &[PAGE_ONE]);
    // Add a standard security handler whose password hashes match no
    // password, so the empty user password lopdf tries on load fails.
    let mut doc = Document::load(&path).unwrap();
    let encrypt_id = doc.add_object(dictionary! {
        "Filter" => "Standard",
        "V" => 1,
        "R" => 2,
        "O" => Object::string_literal(vec![1u8; 32]),
        "U" => Object::string_literal(vec![2u8; 32]),
        "P" => -4,
    });
    doc.trailer.set("Encrypt", encrypt_id);
    doc.trailer.set(
        "ID",
        vec![
            Object::string_literal(vec![3u8; 16]),
            Object::string_literal(vec![3u8; 16]),
        ],
    );
    doc.save(&path).unwrap();

    let err = extract(&path, Format::Pdf).unwrap_err();
    assert!(matches!(err, ParseError::Encrypted), "got {err:?}");
    assert_eq!(err.to_string(), "encrypted");
}

#[test]
fn pdf_with_almost_no_text_is_reported_as_scanned() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scan.pdf");
    // Two pages but only one page's worth of text in total.
    write_pdf(&path, &[PAGE_ONE, "Page 2"]);
    let err = extract(&path, Format::Pdf).unwrap_err();
    assert!(matches!(err, ParseError::NoTextLayer), "got {err:?}");
    assert_eq!(err.to_string(), "no text layer");
}

#[test]
fn extracts_docx_paragraphs_and_tables() {
    use docx_rs::{Docx, Paragraph, Run, Table, TableCell, TableRow};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.docx");
    let para = |text: &str| Paragraph::new().add_run(Run::new().add_text(text));
    Docx::new()
        .add_paragraph(para("1. Definitions"))
        .add_paragraph(Paragraph::new())
        .add_paragraph(para("\"Agreement\" means this ").add_run(Run::new().add_text("contract.")))
        .add_table(Table::new(vec![TableRow::new(vec![
            TableCell::new().add_paragraph(para("Fee")),
            TableCell::new().add_paragraph(para("USD 500")),
        ])]))
        .build()
        .pack(fs::File::create(&path).unwrap())
        .unwrap();

    let out = extract(&path, Format::Docx).unwrap();
    assert_eq!(
        out.text,
        "1. Definitions\n\n\"Agreement\" means this contract.\n\nFee\n\nUSD 500"
    );
    assert!(out.pages.is_empty());
}

#[test]
fn corrupt_docx_is_a_parse_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.docx");
    fs::write(&path, b"PK not a zip").unwrap();
    assert!(matches!(
        extract(&path, Format::Docx),
        Err(ParseError::Format(_))
    ));
}
