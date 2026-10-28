//! Ingestion: find the files to index and extract their text.

mod walk;

pub use walk::{FoundFile, walk};

use std::path::PathBuf;

/// A file that was left out of the index, and why. Bad files never stop a
/// run; they are reported back to the user instead (NFR6).
#[derive(Debug, Clone, PartialEq)]
pub struct Skipped {
    pub path: PathBuf,
    pub reason: String,
}
