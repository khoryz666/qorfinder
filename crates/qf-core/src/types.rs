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

    /// Inverse of [`Format::as_str`].
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pdf" => Some(Self::Pdf),
            "docx" => Some(Self::Docx),
            "txt" => Some(Self::Txt),
            _ => None,
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

/// One stored passage of a document. Its embedding is row `id` of the
/// vector file.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub id: u32,
    pub doc_id: u32,
    /// Position of the chunk within its document, from 0.
    pub ord: u32,
    pub text: String,
    /// `[char_start, char_end)` in characters of the extracted text.
    pub char_start: usize,
    pub char_end: usize,
    pub page: Option<u32>,
}

/// Settings an index was built with. Search refuses an index whose
/// settings differ from the running program's, because its vectors or
/// chunks would not be comparable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMeta {
    pub model_id: String,
    /// SHA-256 of the model weights, so a changed weights file is caught
    /// even under the same model id.
    pub model_rev: String,
    pub dim: usize,
    pub max_tokens: usize,
    pub overlap: usize,
}

/// One search result: a chunk and its similarity to the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub chunk_id: u32,
    pub score: f32,
}
