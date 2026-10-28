use std::fs;

use qf_core::Format;
use qf_core::ingest::{hash_file, walk};

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
