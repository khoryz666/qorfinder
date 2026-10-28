use std::ops::Range;
use std::path::Path;

use super::{Tokenize, TokenizeError};

/// The embedding model's own tokenizer, used to count chunk tokens.
pub struct ModelTokenizer(tokenizers::Tokenizer);

impl ModelTokenizer {
    /// Load `tokenizer.json`. Its truncation and padding settings are turned
    /// off: they are for model input, and would make counts wrong here.
    pub fn from_file(path: &Path) -> Result<Self, TokenizeError> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(path)?;
        tokenizer.with_truncation(None)?.with_padding(None);
        Ok(Self(tokenizer))
    }
}

impl Tokenize for ModelTokenizer {
    fn token_offsets(&self, text: &str) -> Result<Vec<Range<usize>>, TokenizeError> {
        let encoding = self.0.encode(text, false)?;
        Ok(encoding.get_offsets().iter().map(|&(s, e)| s..e).collect())
    }
}
