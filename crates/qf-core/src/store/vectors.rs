use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::StoreError;

/// Appends embeddings to `vectors.f32`: every row is `dim` little-endian
/// `f32`s, row `i` belonging to chunk `i`, with no header.
pub struct VectorWriter {
    out: BufWriter<File>,
    dim: usize,
    rows: usize,
}

impl VectorWriter {
    pub fn create(path: &Path, dim: usize) -> Result<Self, StoreError> {
        Ok(Self {
            out: BufWriter::new(File::create(path)?),
            dim,
            rows: 0,
        })
    }

    /// Append one row.
    pub fn push(&mut self, vector: &[f32]) -> Result<(), StoreError> {
        if vector.len() != self.dim {
            return Err(StoreError::Corrupt(format!(
                "vector of length {} in a {}-dimension index",
                vector.len(),
                self.dim
            )));
        }
        for x in vector {
            self.out.write_all(&x.to_le_bytes())?;
        }
        self.rows += 1;
        Ok(())
    }

    /// Flush to disk and return the number of rows written.
    pub fn finish(mut self) -> Result<usize, StoreError> {
        self.out.flush()?;
        self.out.get_ref().sync_all()?;
        Ok(self.rows)
    }
}

/// Load the whole vector file into one row-major `Vec<f32>`.
pub fn read_vectors(path: &Path, dim: usize) -> Result<Vec<f32>, StoreError> {
    let bytes = std::fs::read(path)?;
    if dim == 0 || bytes.len() % (dim * 4) != 0 {
        return Err(StoreError::Corrupt(format!(
            "{} is {} bytes, not a whole number of {dim}-dimension rows",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}
