use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

const INITIAL_CAPACITY: usize = 1024;

/// Embedded HNSW ANN vector index (usearch), keyed by u64. Stores only
/// vector geometry — no payload; callers resolve a hit's key back to its
/// text/file_path via the lexical store.
pub struct VectorStore {
    index: Index,
    path: PathBuf,
}

impl VectorStore {
    /// Open the index at `path` if it exists, otherwise create a fresh one
    /// for `dims`-dimensional vectors under cosine distance.
    pub fn open(path: &Path, dims: usize) -> Result<Self> {
        let options = IndexOptions {
            dimensions: dims,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options).context("failed to create usearch index")?;
        if path.exists() {
            let path_str = path.to_str().context("vector index path is not valid UTF-8")?;
            index
                .load(path_str)
                .with_context(|| format!("failed to load vector index from {}", path.display()))?;
        } else {
            index
                .reserve(INITIAL_CAPACITY)
                .context("failed to reserve initial vector index capacity")?;
        }
        Ok(Self {
            index,
            path: path.to_path_buf(),
        })
    }

    fn ensure_capacity(&self, extra: usize) -> Result<()> {
        let needed = self.index.size() + extra;
        if needed > self.index.capacity() {
            let grown = (needed * 2).max(INITIAL_CAPACITY);
            self.index
                .reserve(grown)
                .context("failed to grow vector index capacity")?;
        }
        Ok(())
    }

    /// Insert or replace the vector at `key`. Idempotent: removes any
    /// existing vector under `key` first, so re-indexing an unchanged-looking
    /// file never duplicates a vector.
    pub fn add(&self, key: u64, vector: &[f32]) -> Result<()> {
        if self.index.contains(key) {
            self.index
                .remove(key)
                .context("failed to remove existing vector before re-adding")?;
        }
        self.ensure_capacity(1)?;
        self.index.add(key, vector).context("failed to add vector")?;
        Ok(())
    }

    /// Remove the vector at `key`, if present.
    pub fn remove(&self, key: u64) -> Result<()> {
        if self.index.contains(key) {
            self.index.remove(key).context("failed to remove vector")?;
        }
        Ok(())
    }

    /// Top-k nearest neighbors by cosine distance, as (key, distance) pairs.
    pub fn search(&self, vector: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        if self.index.size() == 0 {
            return Ok(Vec::new());
        }
        let matches = self
            .index
            .search(vector, k)
            .context("vector search failed")?;
        Ok(matches.keys.into_iter().zip(matches.distances).collect())
    }

    pub fn save(&self) -> Result<()> {
        let path_str = self
            .path
            .to_str()
            .context("vector index path is not valid UTF-8")?;
        self.index
            .save(path_str)
            .with_context(|| format!("failed to save vector index to {}", self.path.display()))?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.index.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(vals: &[f32]) -> Vec<f32> {
        vals.to_vec()
    }

    #[test]
    fn add_then_search_finds_nearest() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&dir.path().join("v.usearch"), 3).unwrap();
        store.add(1, &v(&[1.0, 0.0, 0.0])).unwrap();
        store.add(2, &v(&[0.0, 1.0, 0.0])).unwrap();
        let results = store.search(&v(&[1.0, 0.0, 0.0]), 1).unwrap();
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn add_is_idempotent_on_same_key() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&dir.path().join("v.usearch"), 3).unwrap();
        store.add(1, &v(&[1.0, 0.0, 0.0])).unwrap();
        store.add(1, &v(&[0.0, 1.0, 0.0])).unwrap();
        assert_eq!(store.len(), 1);
        let results = store.search(&v(&[0.0, 1.0, 0.0]), 1).unwrap();
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn remove_deletes_vector() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&dir.path().join("v.usearch"), 3).unwrap();
        store.add(1, &v(&[1.0, 0.0, 0.0])).unwrap();
        store.remove(1).unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn remove_missing_key_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&dir.path().join("v.usearch"), 3).unwrap();
        store.remove(42).unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn search_on_empty_index_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&dir.path().join("v.usearch"), 3).unwrap();
        assert!(store.search(&v(&[1.0, 0.0, 0.0]), 5).unwrap().is_empty());
    }

    #[test]
    fn save_then_reopen_preserves_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.usearch");
        {
            let store = VectorStore::open(&path, 3).unwrap();
            store.add(7, &v(&[0.5, 0.5, 0.0])).unwrap();
            store.save().unwrap();
        }
        let reopened = VectorStore::open(&path, 3).unwrap();
        assert_eq!(reopened.len(), 1);
        let results = reopened.search(&v(&[0.5, 0.5, 0.0]), 1).unwrap();
        assert_eq!(results[0].0, 7);
    }

    #[test]
    fn grows_capacity_beyond_initial_reservation() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&dir.path().join("v.usearch"), 2).unwrap();
        for i in 0..(INITIAL_CAPACITY as u64 + 10) {
            store.add(i, &v(&[i as f32, 1.0])).unwrap();
        }
        assert_eq!(store.len(), INITIAL_CAPACITY + 10);
    }
}
