use qorfinder::embedder::{passage_text, query_text};

#[test]
fn passage_text_gets_passage_prefix() {
    assert_eq!(passage_text("hello"), "passage: hello");
}

#[test]
fn query_text_gets_query_prefix() {
    assert_eq!(query_text("hello"), "query: hello");
}

#[test]
fn prefixes_differ() {
    assert_ne!(passage_text("x"), query_text("x"));
}
