use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

const TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("fingerprints");

/// What the store knows about a previously indexed file: its (mtime, size)
/// fingerprint, how many chunks it produced, and the usearch keys those
/// chunks live under (needed to remove their vectors on change/delete).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FingerprintRecord {
    pub mtime_secs: i64,
    pub size_bytes: i64,
    pub chunk_count: u64,
    pub vector_keys: Vec<u64>,
}

impl FingerprintRecord {
    pub fn matches(&self, meta: &std::fs::Metadata) -> bool {
        let (mtime, size) = file_fingerprint(meta);
        self.mtime_secs == mtime && self.size_bytes == size
    }
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

/// Embedded key-value side-table (redb) mapping canonical file path -> its
/// fingerprint record. This is the single place that knows which chunk ids
/// and vector keys belong to a given file, so incremental re-indexing never
/// needs to scan the lexical or vector store.
pub struct FingerprintStore {
    db: Database,
}

impl FingerprintStore {
    pub fn open(path: &Path) -> Result<Self> {
        let db = Database::create(path)
            .with_context(|| format!("failed to open fingerprint db at {}", path.display()))?;
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(TABLE)?;
        }
        write_txn.commit()?;
        Ok(Self { db })
    }

    pub fn get(&self, file_path: &str) -> Result<Option<FingerprintRecord>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(TABLE)?;
        match table.get(file_path)? {
            Some(guard) => {
                let record = serde_json::from_slice(guard.value())
                    .context("failed to decode fingerprint record")?;
                Ok(Some(record))
            }
            None => Ok(None),
        }
    }

    pub fn set(&self, file_path: &str, record: &FingerprintRecord) -> Result<()> {
        let bytes = serde_json::to_vec(record).context("failed to encode fingerprint record")?;
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TABLE)?;
            table.insert(file_path, bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn delete(&self, file_path: &str) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TABLE)?;
            table.remove(file_path)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Every known file's fingerprint. Replaces the old Qdrant-backed
    /// `Store::file_fingerprints()`'s full-collection scroll with an O(files)
    /// table iteration.
    pub fn all(&self) -> Result<HashMap<String, FingerprintRecord>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(TABLE)?;
        let mut out = HashMap::new();
        for entry in table.iter()? {
            let (key, value) = entry?;
            let record: FingerprintRecord = serde_json::from_slice(value.value())
                .context("failed to decode fingerprint record")?;
            out.insert(key.value().to_string(), record);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(mtime: i64, size: i64, count: u64, keys: &[u64]) -> FingerprintRecord {
        FingerprintRecord {
            mtime_secs: mtime,
            size_bytes: size,
            chunk_count: count,
            vector_keys: keys.to_vec(),
        }
    }

    #[test]
    fn set_then_get_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let store = FingerprintStore::open(&dir.path().join("fp.redb")).unwrap();
        let record = rec(100, 50, 3, &[1, 2, 3]);
        store.set("/a.txt", &record).unwrap();
        assert_eq!(store.get("/a.txt").unwrap(), Some(record));
    }

    #[test]
    fn missing_key_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = FingerprintStore::open(&dir.path().join("fp.redb")).unwrap();
        assert_eq!(store.get("/missing.txt").unwrap(), None);
    }

    #[test]
    fn delete_removes_entry() {
        let dir = tempfile::tempdir().unwrap();
        let store = FingerprintStore::open(&dir.path().join("fp.redb")).unwrap();
        store.set("/a.txt", &rec(1, 1, 1, &[9])).unwrap();
        store.delete("/a.txt").unwrap();
        assert_eq!(store.get("/a.txt").unwrap(), None);
    }

    #[test]
    fn all_returns_every_entry() {
        let dir = tempfile::tempdir().unwrap();
        let store = FingerprintStore::open(&dir.path().join("fp.redb")).unwrap();
        store.set("/a.txt", &rec(1, 1, 1, &[1])).unwrap();
        store.set("/b.txt", &rec(2, 2, 2, &[2, 3])).unwrap();
        let all = store.all().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all["/b.txt"].vector_keys, vec![2, 3]);
    }

    #[test]
    fn reopening_existing_db_preserves_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fp.redb");
        {
            let store = FingerprintStore::open(&path).unwrap();
            store.set("/a.txt", &rec(1, 1, 1, &[1])).unwrap();
        }
        let store = FingerprintStore::open(&path).unwrap();
        assert!(store.get("/a.txt").unwrap().is_some());
    }
}
