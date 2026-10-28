use std::fs::File;
use std::io;
use std::path::Path;

use sha2::{Digest, Sha256};

/// SHA-256 of a file's bytes, streamed so large files are never held in
/// memory twice.
pub fn hash_file(path: &Path) -> io::Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    io::copy(&mut File::open(path)?, &mut hasher)?;
    Ok(hasher.finalize().into())
}
