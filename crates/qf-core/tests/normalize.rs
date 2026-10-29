use qf_core::ingest::{normalize, normalize_pdf};

#[test]
fn collapses_spaces_and_trims_lines() {
    let n = normalize("  1.  Term \t of\u{a0}the  Agreement  \n   The term is one year.  ");
    assert_eq!(n.text, "1. Term of the Agreement\nThe term is one year.");
}

#[test]
fn keeps_paragraph_breaks_as_two_newlines() {
    let n = normalize("First para.\r\n\r\n\r\n  \nSecond para.\rSame para.");
    assert_eq!(n.text, "First para.\n\nSecond para.\nSame para.");
}

#[test]
fn pdf_text_joins_words_hyphenated_across_lines() {
    let n = normalize_pdf("the con-\n  tract is bind-\ning; a well-\nKnown term; non-\n\ncompete");
    assert_eq!(
        n.text,
        "the contract is binding; a well-\nKnown term; non-\n\ncompete"
    );
}

#[test]
fn typed_text_keeps_hyphens_at_line_ends() {
    let n = normalize("a third-\nparty licence");
    assert_eq!(n.text, "a third-\nparty licence");
}

#[test]
fn applies_nfc() {
    // "e" + combining acute accent composes to a single "é".
    let n = normalize("cafe\u{301} owner");
    assert_eq!(n.text, "caf\u{e9} owner");
    // The composed char maps back to both source chars.
    assert_eq!(n.source_range(3, 4), (3, 5));
    assert_eq!(n.source_range(5, 10), (6, 11));
}

#[test]
fn source_range_maps_back_to_the_original_text() {
    let source = "  Section   12.3\n\n\nThe Licensee shall pay   royal-\nties.";
    let n = normalize_pdf(source);
    assert_eq!(n.text, "Section 12.3\n\nThe Licensee shall pay royalties.");
    let src: Vec<char> = source.chars().collect();
    let slice = |(s, e): (usize, usize)| src[s..e].iter().collect::<String>();

    let find = |needle: &str| {
        let byte = n.text.find(needle).unwrap();
        let start = n.text[..byte].chars().count();
        (start, start + needle.chars().count())
    };
    let (s, e) = find("12.3");
    assert_eq!(slice(n.source_range(s, e)), "12.3");
    let (s, e) = find("The Licensee shall pay");
    assert_eq!(slice(n.source_range(s, e)), "The Licensee shall pay");
    let (s, e) = find("royalties.");
    assert_eq!(slice(n.source_range(s, e)), "royal-\nties.");
    assert_eq!(n.source_range(0, 0), (2, 2));
}

#[test]
fn empty_and_blank_text_normalise_to_empty() {
    assert_eq!(normalize("").text, "");
    assert_eq!(normalize(" \n\n\t ").text, "");
    assert_eq!(normalize("").source_range(0, 0), (0, 0));
}
