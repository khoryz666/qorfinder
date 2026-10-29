use std::fs;
use std::ops::Range;

use qf_core::Format;
use qf_core::chunk::{ChunkConfig, Tokenize, TokenizeError};
use qf_core::prepare::{PrepareError, prepare};

struct Words;

impl Tokenize for Words {
    fn token_offsets(&self, text: &str) -> Result<Vec<Range<usize>>, TokenizeError> {
        let base = text.as_ptr() as usize;
        Ok(text
            .split_whitespace()
            .map(|w| {
                let start = w.as_ptr() as usize - base;
                start..start + w.len()
            })
            .collect())
    }
}

#[test]
fn passages_point_back_into_the_original_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("contract.txt");
    let source = "  1.  Licence.   The Licensor  grants a\nnon-exclusive licence.\n\n\n\
                  2. Royal-\nties.  The Licensee pays   5%.";
    fs::write(&path, source).unwrap();
    let config = ChunkConfig {
        max_tokens: 8,
        overlap: 0,
    };

    let passages = prepare(&path, Format::Txt, &Words, config).unwrap();
    let texts: Vec<&str> = passages.iter().map(|p| p.text.as_str()).collect();
    assert_eq!(
        texts,
        [
            "1. Licence. The Licensor grants a\nnon-exclusive licence.",
            "2. Royal-\nties. The Licensee pays 5%."
        ]
    );
    let chars: Vec<char> = source.chars().collect();
    let original = |i: usize| -> String {
        chars[passages[i].char_start..passages[i].char_end]
            .iter()
            .collect()
    };
    assert_eq!(
        original(0),
        "1.  Licence.   The Licensor  grants a\nnon-exclusive licence."
    );
    assert_eq!(original(1), "2. Royal-\nties.  The Licensee pays   5%.");
    assert!(passages.iter().all(|p| p.page.is_none()));
}

/// CUAD counts annotation offsets in characters, as Python does. This
/// excerpt (the first 4,500 characters of a CUAD contract, CC BY 4.0) has
/// 2- and 3-byte characters (`é`, `®`, `•`) before a "Rofr/Rofo/Rofn"
/// clause at characters 4100..4479, 22 bytes further in than its character
/// offset, so any byte offset leaking through would miss it.
#[test]
fn passages_keep_cuad_character_offsets_after_non_ascii_text() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cuad_playa_excerpt.txt");
    let source = fs::read_to_string(&path).unwrap();
    let chars: Vec<char> = source.chars().collect();
    let (start, end) = (4100, 4479);
    let clause: String = chars[start..end].iter().collect();
    assert!(clause.starts_with("If the Offering Party is required to off"));
    assert!(clause.ends_with("(collectively, the \"Offer Notice\")."));
    assert!(source.char_indices().nth(start).unwrap().0 > start);

    let config = ChunkConfig {
        max_tokens: 40,
        overlap: 8,
    };
    let passages = prepare(&path, Format::Txt, &Words, config).unwrap();
    for p in &passages {
        assert_eq!(p.text.chars().next(), Some(chars[p.char_start]));
        assert_eq!(p.text.chars().next_back(), Some(chars[p.char_end - 1]));
    }
    let covering = passages
        .iter()
        .find(|p| p.char_start <= start && start < p.char_end)
        .unwrap();
    let original: String = chars[covering.char_start..covering.char_end]
        .iter()
        .collect();
    assert!(original.contains("If the Offering Party"));
}

#[test]
fn blank_file_has_no_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blank.txt");
    fs::write(&path, " \n\n ").unwrap();
    let err = prepare(&path, Format::Txt, &Words, ChunkConfig::default()).unwrap_err();
    assert!(matches!(err, PrepareError::Empty));
    assert_eq!(err.to_string(), "no text");
}
