use qorfinder::corpus::{beir_body, count_txt_files, quran_file_name, sanitize_doc_id};

#[test]
fn sanitizes_forbidden_filename_chars() {
    assert_eq!(sanitize_doc_id("PLAIN-2"), "PLAIN-2");
    assert_eq!(
        sanitize_doc_id("a/b\\c:d*e?f\"g<h>i|j"),
        "a_b_c_d_e_f_g_h_i_j"
    );
}

#[test]
fn beir_body_prefers_title() {
    assert_eq!(beir_body(Some("Title"), "text"), "Title\ntext");
    assert_eq!(beir_body(None, "text"), "text");
    assert_eq!(beir_body(Some(""), "text"), "text");
}

#[test]
fn quran_file_names_are_stable() {
    assert_eq!(quran_file_name(2, 255), "surah-2-ayah-255.txt");
}

#[test]
fn counts_txt_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "x").unwrap();
    std::fs::write(dir.path().join("b.md"), "y").unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("sub").join("c.txt"), "z").unwrap();
    assert_eq!(count_txt_files(dir.path()), 2);
}
