mod fingerprint;
mod lexical;
mod vector;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use fingerprint::FingerprintRecord;
pub use lexical::{LexicalHit, ResolvedChunk};

use fingerprint::FingerprintStore;
use lexical::LexicalStore;
use vector::VectorStore;

pub const DEFAULT_INDEX_DIR_NAME: &str = "default";

pub struct SearchHit {
    pub file_path: String,
    pub chunk_index: u64,
    pub score: f32,
    pub text: String,
}

/// (mtime seconds, size bytes) fingerprint of a file on disk.
pub fn file_fingerprint(meta: &std::fs::Metadata) -> (i64, i64) {
    let secs = meta
        .modified()
        .ok()
        .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    (secs, meta.len() as i64)
}

fn chunk_id(path_str: &str, chunk_index: u64) -> String {
    format!("{path_str}:{chunk_index}")
}

/// Derive the usearch vector key for a (file, chunk index) pair. Collisions
/// are astronomically unlikely at personal-corpus scale (u64 keyspace).
fn vector_key(path_str: &str, chunk_index: u64) -> u64 {
    let digest = Sha256::digest(chunk_id(path_str, chunk_index).as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("sha256 digest is >= 8 bytes"))
}

#[derive(Serialize, Deserialize)]
struct VersionInfo {
    dims: usize,
}

fn check_or_write_version(path: &Path, dims: usize) -> Result<()> {
    if path.exists() {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let info: VersionInfo = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        anyhow::ensure!(
            info.dims == dims,
            "index at {} was built with {} dims but the embedding model produces {}; \
             delete the index directory or use a matching model",
            path.display(),
            info.dims,
            dims
        );
    } else {
        let info = VersionInfo { dims };
        std::fs::write(path, serde_json::to_string_pretty(&info)?)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

/// Embedded hybrid store: a tantivy lexical index + a usearch ANN vector
/// index + a redb fingerprint side-table, presented as one facade. No
/// external service, no Docker — everything lives under one `index_dir`.
///
/// All engines are internally `&self`-safe (see `lexical::LexicalStore`'s
/// doc comment for how tantivy's one `&mut self` method is handled), so the
/// whole facade stays `&self` throughout, matching `Indexer`'s methods.
pub struct Store {
    lexical: LexicalStore,
    vector: VectorStore,
    fingerprint: FingerprintStore,
    /// Fingerprint updates staged by `stage_chunk`/`delete_file`, flushed to
    /// disk by `commit()` only after the lexical/vector writes they describe
    /// are themselves durable. `None` means "delete this file's record".
    pending: Mutex<HashMap<String, Option<FingerprintRecord>>>,
}

impl Store {
    pub fn open(index_dir: &Path, dims: usize) -> Result<Self> {
        std::fs::create_dir_all(index_dir)
            .with_context(|| format!("failed to create index dir {}", index_dir.display()))?;
        check_or_write_version(&index_dir.join("version.json"), dims)?;
        let lexical = LexicalStore::open(&index_dir.join("tantivy"))?;
        let vector = VectorStore::open(&index_dir.join("vectors.usearch"), dims)?;
        let fingerprint = FingerprintStore::open(&index_dir.join("meta.redb"))?;
        Ok(Self {
            lexical,
            vector,
            fingerprint,
            pending: Mutex::new(HashMap::new()),
        })
    }

    /// Look up a file's current record: whatever's staged this scan, else
    /// what's already durable.
    fn record_for(&self, path_str: &str) -> Result<Option<FingerprintRecord>> {
        let pending = self.pending.lock().unwrap();
        match pending.get(path_str) {
            Some(record) => Ok(record.clone()),
            None => self.fingerprint.get(path_str),
        }
    }

    /// Remove every chunk currently known for `path` (staged or durable) and
    /// mark it deleted. Idempotent — safe to call for a file with no chunks.
    pub fn delete_file(&self, path: &Path) -> Result<()> {
        let path_str = path.display().to_string();
        if let Some(record) = self.record_for(&path_str)? {
            self.lexical.delete_file(&path_str);
            for key in &record.vector_keys {
                self.vector.remove(*key)?;
            }
        }
        self.pending.lock().unwrap().insert(path_str, None);
        Ok(())
    }

    /// Stage one chunk of `path`. Chunks for a given file are expected to
    /// arrive with increasing `chunk_index` (indexer.rs processes a file's
    /// chunks in order even when embedding batches span multiple files) —
    /// each call appends to that file's pending record rather than
    /// replacing it, so cross-file embedding batches never corrupt a
    /// partially-staged file.
    pub fn stage_chunk(
        &self,
        path: &Path,
        chunk_index: u64,
        text: &str,
        vector: &[f32],
        fingerprint: (i64, i64),
    ) -> Result<()> {
        let path_str = path.display().to_string();
        let cid = chunk_id(&path_str, chunk_index);
        let vkey = vector_key(&path_str, chunk_index);
        self.lexical
            .upsert_chunk(&cid, &path_str, chunk_index, text, fingerprint, vkey)?;
        self.vector.add(vkey, vector)?;

        let mut pending = self.pending.lock().unwrap();
        let record = pending
            .entry(path_str)
            .or_insert_with(|| {
                Some(FingerprintRecord {
                    mtime_secs: fingerprint.0,
                    size_bytes: fingerprint.1,
                    chunk_count: 0,
                    vector_keys: Vec::new(),
                })
            })
            .get_or_insert_with(|| FingerprintRecord {
                mtime_secs: fingerprint.0,
                size_bytes: fingerprint.1,
                chunk_count: 0,
                vector_keys: Vec::new(),
            });
        record.chunk_count += 1;
        record.vector_keys.push(vkey);
        Ok(())
    }

    /// Commit staged lexical/vector writes to disk, then flush pending
    /// fingerprint updates. Order matters for crash-safety: a fingerprint is
    /// only marked done after the data it describes is durable.
    pub fn commit(&self) -> Result<()> {
        self.lexical.commit()?;
        self.vector.save()?;
        let pending = std::mem::take(&mut *self.pending.lock().unwrap());
        self.fingerprint.apply_many(&pending)?;
        Ok(())
    }

    /// Fingerprint of every indexed file (durable only — call after
    /// `commit()` for a consistent view of a completed scan).
    pub fn file_fingerprints(&self) -> Result<HashMap<String, FingerprintRecord>> {
        self.fingerprint.all()
    }

    /// Fingerprint of a single file, `None` if it has no chunks.
    pub fn file_info(&self, path: &Path) -> Result<Option<FingerprintRecord>> {
        self.fingerprint.get(&path.display().to_string())
    }

    /// Dense-only nearest neighbors, as (usearch key, cosine distance).
    pub fn search_vector(&self, vector: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        self.vector.search(vector, k)
    }

    /// BM25 lexical search over chunk text.
    pub fn search_lexical(&self, query_text: &str, k: usize) -> Result<Vec<LexicalHit>> {
        self.lexical.search(query_text, k)
    }

    /// Resolve a usearch vector key back to its chunk (file, index, text) —
    /// the dense side of a hybrid query only has keys and distances.
    pub fn resolve_vector_key(&self, key: u64) -> Result<Option<ResolvedChunk>> {
        self.lexical.resolve_vector_key(key)
    }

    /// Dense-only search, resolved to full hits. A thin convenience wrapper
    /// kept for callers (eval.rs, and cli.rs/query.rs until the hybrid RRF
    /// query path lands) that don't yet need the lexical side.
    pub fn search(&self, vector: Vec<f32>, top_k: u64) -> Result<Vec<SearchHit>> {
        let hits = self.vector.search(&vector, top_k as usize)?;
        let mut out = Vec::with_capacity(hits.len());
        for (key, score) in hits {
            if let Some(resolved) = self.resolve_vector_key(key)? {
                out.push(SearchHit {
                    file_path: resolved.file_path,
                    chunk_index: resolved.chunk_index,
                    score,
                    text: resolved.text,
                });
            }
        }
        Ok(out)
    }

    pub fn count(&self) -> Result<u64> {
        Ok(self.vector.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(dir: &Path) -> Store {
        Store::open(dir, 3).unwrap()
    }

    #[test]
    fn stage_commit_search_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let store = open(dir.path());
        let path = Path::new("/a.txt");
        store
            .stage_chunk(path, 0, "hello world", &[1.0, 0.0, 0.0], (1, 2))
            .unwrap();
        store.commit().unwrap();

        let hits = store.search(vec![1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file_path, "/a.txt");
        assert_eq!(hits[0].text, "hello world");

        let lexical_hits = store.search_lexical("hello", 5).unwrap();
        assert_eq!(lexical_hits.len(), 1);
    }

    #[test]
    fn chunks_across_embedding_batches_accumulate_not_replace() {
        // Simulates indexer.rs staging one file's chunks interleaved with
        // another file's, as happens when an embed batch spans file
        // boundaries. A naive "replace all chunks for path" implementation
        // would drop earlier chunks when later ones for the same path
        // arrive in a later call.
        let dir = tempfile::tempdir().unwrap();
        let store = open(dir.path());
        let a = Path::new("/a.txt");
        let b = Path::new("/b.txt");
        store
            .stage_chunk(a, 0, "a chunk zero", &[1.0, 0.0, 0.0], (1, 1))
            .unwrap();
        store
            .stage_chunk(b, 0, "b chunk zero", &[0.0, 1.0, 0.0], (1, 1))
            .unwrap();
        store
            .stage_chunk(a, 1, "a chunk one", &[0.0, 0.0, 1.0], (1, 1))
            .unwrap();
        store.commit().unwrap();

        let fps = store.file_fingerprints().unwrap();
        assert_eq!(fps["/a.txt"].chunk_count, 2);
        assert_eq!(fps["/b.txt"].chunk_count, 1);
        let hits = store.search_lexical("chunk", 10).unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn delete_file_removes_staged_and_durable_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let store = open(dir.path());
        let path = Path::new("/a.txt");
        store
            .stage_chunk(path, 0, "first version", &[1.0, 0.0, 0.0], (1, 1))
            .unwrap();
        store.commit().unwrap();

        store.delete_file(path).unwrap();
        store
            .stage_chunk(path, 0, "second version", &[0.0, 1.0, 0.0], (2, 2))
            .unwrap();
        store.commit().unwrap();

        let hits = store.search_lexical("version", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "second version");
        assert_eq!(store.file_fingerprints().unwrap()["/a.txt"].chunk_count, 1);
    }

    #[test]
    fn delete_of_unknown_file_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let store = open(dir.path());
        store.delete_file(Path::new("/missing.txt")).unwrap();
    }

    #[test]
    fn file_info_reflects_committed_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let store = open(dir.path());
        let path = Path::new("/a.txt");
        assert!(store.file_info(path).unwrap().is_none());
        store
            .stage_chunk(path, 0, "text", &[1.0, 0.0, 0.0], (10, 20))
            .unwrap();
        store.commit().unwrap();
        let info = store.file_info(path).unwrap().unwrap();
        assert_eq!(info.mtime_secs, 10);
        assert_eq!(info.size_bytes, 20);
        assert_eq!(info.chunk_count, 1);
    }

    #[test]
    fn count_reflects_vector_count() {
        let dir = tempfile::tempdir().unwrap();
        let store = open(dir.path());
        store
            .stage_chunk(Path::new("/a.txt"), 0, "x", &[1.0, 0.0, 0.0], (1, 1))
            .unwrap();
        store
            .stage_chunk(Path::new("/a.txt"), 1, "y", &[0.0, 1.0, 0.0], (1, 1))
            .unwrap();
        store.commit().unwrap();
        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn reopening_existing_index_rejects_mismatched_dims() {
        let dir = tempfile::tempdir().unwrap();
        Store::open(dir.path(), 3).unwrap();
        match Store::open(dir.path(), 4) {
            Err(err) => assert!(err.to_string().contains("dims")),
            Ok(_) => panic!("expected a dims mismatch error"),
        }
    }

    #[test]
    fn reopening_existing_index_with_same_dims_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = Store::open(dir.path(), 3).unwrap();
            store
                .stage_chunk(
                    Path::new("/a.txt"),
                    0,
                    "persisted",
                    &[1.0, 0.0, 0.0],
                    (1, 1),
                )
                .unwrap();
            store.commit().unwrap();
        }
        let store = Store::open(dir.path(), 3).unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }
}
