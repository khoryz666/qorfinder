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
            "2. Royalties. The Licensee pays 5%."
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

#[test]
fn blank_file_has_no_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blank.txt");
    fs::write(&path, " \n\n ").unwrap();
    let err = prepare(&path, Format::Txt, &Words, ChunkConfig::default()).unwrap_err();
    assert!(matches!(err, PrepareError::Empty));
    assert_eq!(err.to_string(), "no text");
}
