//! Ingestion: find the files to index and extract their text.

mod extract;
mod hash;
mod normalize;
mod walk;

pub use extract::{Extracted, MAX_FILE_SIZE_BYTES, MIN_PDF_CHARS_PER_PAGE, ParseError, extract};
pub use hash::hash_file;
pub use normalize::{Normalized, normalize, normalize_pdf};
pub use walk::{FoundFile, walk, walk_except};

use std::path::PathBuf;

/// A file that was left out of the index, and why. Bad files never stop a
/// run; they are reported back to the user instead (NFR6).
#[derive(Debug, Clone, PartialEq)]
pub struct Skipped {
    pub path: PathBuf,
    pub reason: String,
}
