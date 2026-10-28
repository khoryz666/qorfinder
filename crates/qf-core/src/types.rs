use std::path::{Path, PathBuf};

/// File formats QorFinder can extract text from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    Pdf,
    Docx,
    Txt,
}

impl Format {
    /// Format from the file extension (case-insensitive), `None` for
    /// anything QorFinder does not index.
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "pdf" => Some(Self::Pdf),
            "docx" => Some(Self::Docx),
            "txt" => Some(Self::Txt),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Docx => "docx",
            Self::Txt => "txt",
        }
    }
}

/// One indexed source file.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub id: u32,
    pub path: PathBuf,
    pub format: Format,
    /// SHA-256 of the file bytes, kept so FYP2 can re-index only what changed.
    pub hash: [u8; 32],
    /// Modification time in seconds since the Unix epoch.
    pub mtime: i64,
}
