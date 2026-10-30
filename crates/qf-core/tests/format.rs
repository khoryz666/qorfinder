use qf_core::format::{snippet, wrap};

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

#[test]
fn long_text_is_cut_between_words() {
    assert_eq!(snippet("its terms apply", 7), "its…");
    assert_eq!(snippet("its terms apply", 9), "its terms…");
    assert_eq!(
        snippet("Licensee shall not assign", 20),
        "Licensee shall not…"
    );
}

#[test]
fn wraps_at_spaces_within_the_width() {
    assert_eq!(
        wrap("The Licensee  shall not\nassign this Agreement.", 16),
        ["The Licensee", "shall not assign", "this Agreement."]
    );
    assert_eq!(wrap("a verylongword b", 4), ["a", "verylongword", "b"]);
    assert!(wrap("  \n ", 10).is_empty());
}
