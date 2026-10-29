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
/// cannot be read, files of any other type and symbolic links are returned
/// as skipped. Links are not followed: they can point outside the folder
/// or at a file that is indexed already.
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
        // The folder the user chose is followed even when it is a link.
        if entry.depth() > 0 && entry.path_is_symlink() {
            skipped.push(Skipped {
                path: entry.into_path(),
                reason: "symbolic link, not followed".into(),
            });
            continue;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        // Word and other Office apps keep a `~$name.docx` lock file next to
        // an open document; it holds no document text.
        if entry.file_name().to_string_lossy().starts_with("~$") {
            skipped.push(Skipped {
                path: entry.into_path(),
                reason: "Office lock file".into(),
            });
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
