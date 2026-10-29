use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use walkdir::WalkDir;

use super::Skipped;
use crate::Format;

/// A supported file found under the indexed folder.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundFile {
    pub path: PathBuf,
    pub format: Format,
    pub size: u64,
    /// Modification time in seconds since the Unix epoch.
    pub mtime: i64,
}

/// Recursively list every `.pdf`, `.docx` and `.txt` file under `root`,
/// sorted by path so document ids are the same run to run. Entries that
/// cannot be read, and files of any other type, are returned as skipped.
pub fn walk(root: &Path) -> (Vec<FoundFile>, Vec<Skipped>) {
    let mut found = Vec::new();
    let mut skipped = Vec::new();
    for entry in WalkDir::new(root).follow_links(false).sort_by_file_name() {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                skipped.push(Skipped {
                    path: err.path().unwrap_or(root).to_path_buf(),
                    reason: err.to_string(),
                });
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let Some(format) = Format::from_path(entry.path()) else {
            skipped.push(Skipped {
                path: entry.into_path(),
                reason: "unsupported file type".into(),
            });
            continue;
        };
        match entry.metadata() {
            Ok(meta) => found.push(FoundFile {
                path: entry.into_path(),
                format,
                size: meta.len(),
                mtime: meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs() as i64),
            }),
            Err(err) => skipped.push(Skipped {
                path: entry.into_path(),
                reason: err.to_string(),
            }),
        }
    }
    (found, skipped)
}
