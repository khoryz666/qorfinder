use std::path::Path;

use qf_core::Format;

#[test]
fn format_from_supported_extensions() {
    assert_eq!(Format::from_path(Path::new("a.pdf")), Some(Format::Pdf));
    assert_eq!(Format::from_path(Path::new("a.DOCX")), Some(Format::Docx));
    assert_eq!(Format::from_path(Path::new("dir/a.Txt")), Some(Format::Txt));
}

#[test]
fn format_rejects_everything_else() {
    for name in ["a.md", "a.doc", "a.png", "README", ".pdf"] {
        assert_eq!(Format::from_path(Path::new(name)), None, "{name}");
    }
}
