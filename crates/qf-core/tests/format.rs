use qf_core::format::snippet;

#[test]
fn short_text_is_kept_on_one_line() {
    assert_eq!(
        snippet("  1. Term\n\nOne  year. ", 240),
        "1. Term One year."
    );
}

#[test]
fn long_text_is_cut_with_an_ellipsis() {
    let out = snippet(&"a".repeat(300), 240);
    assert_eq!(out.chars().count(), 241);
    assert!(out.ends_with('…'));
    assert_eq!(snippet("one two three", 4), "one…");
}
