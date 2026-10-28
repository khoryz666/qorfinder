use qf_core::chunk::{is_clause_start, split_units};

fn unit_texts(text: &str) -> Vec<&str> {
    split_units(text).into_iter().map(|r| &text[r]).collect()
}

#[test]
fn recognises_clause_numbers() {
    for line in [
        "12. Term",
        "12.3 Fees",
        "1.2.3. Notice",
        "4) Remedies",
        "(a) the",
        "(iv) any",
        "(2) each",
    ] {
        assert!(is_clause_start(line), "{line}");
    }
    for line in [
        "2023 was",
        "12.",
        "Section 5",
        "(abcde) x",
        "1.2345 x",
        "a. b",
        "(a)b",
        "12,3 x",
    ] {
        assert!(!is_clause_start(line), "{line}");
    }
}

#[test]
fn splits_on_paragraphs() {
    assert_eq!(
        unit_texts("First paragraph\nstill first.\n\nSecond paragraph."),
        ["First paragraph\nstill first.", "Second paragraph."]
    );
}

#[test]
fn splits_before_clause_lines_inside_a_paragraph() {
    assert_eq!(
        unit_texts(
            "Payment terms:\n(a) pay on time;\n(b) pay in full.\n12.3 Late fees apply\nat 2% a month."
        ),
        [
            "Payment terms:",
            "(a) pay on time;",
            "(b) pay in full.",
            "12.3 Late fees apply\nat 2% a month."
        ]
    );
}

#[test]
fn blank_text_has_no_units() {
    assert!(split_units("").is_empty());
    assert!(split_units("\n\n").is_empty());
}
