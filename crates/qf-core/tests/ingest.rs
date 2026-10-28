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
    fs::write(dir.path().join("notes.md"), "ignored").unwrap();

    let (found, skipped) = walk(dir.path());
    assert!(skipped.is_empty());
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
