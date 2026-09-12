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
        let (mtime, size) = super::file_fingerprint(meta);
        self.mtime_secs == mtime && self.size_bytes == size
    }
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

    /// Apply many sets/deletes (`None` = delete) in a single transaction.
    /// Used by the Store facade to flush pending fingerprint updates only
    /// after the corresponding lexical/vector writes are already durable —
    /// so a crash never leaves a fingerprint marked "done" for data that
    /// wasn't actually persisted.
    pub fn apply_many(&self, updates: &HashMap<String, Option<FingerprintRecord>>) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TABLE)?;
            for (file_path, update) in updates {
                match update {
                    Some(record) => {
                        let bytes = serde_json::to_vec(record)
                            .context("failed to encode fingerprint record")?;
                        table.insert(file_path.as_str(), bytes.as_slice())?;
                    }
                    None => {
                        table.remove(file_path.as_str())?;
                    }
                }
            }
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

    fn put(store: &FingerprintStore, file_path: &str, record: FingerprintRecord) {
        let mut updates = HashMap::new();
        updates.insert(file_path.to_string(), Some(record));
        store.apply_many(&updates).unwrap();
    }

    #[test]
    fn set_then_get_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let store = FingerprintStore::open(&dir.path().join("fp.redb")).unwrap();
        let record = rec(100, 50, 3, &[1, 2, 3]);
        put(&store, "/a.txt", record.clone());
        assert_eq!(store.get("/a.txt").unwrap(), Some(record));
    }

    #[test]
    fn missing_key_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = FingerprintStore::open(&dir.path().join("fp.redb")).unwrap();
        assert_eq!(store.get("/missing.txt").unwrap(), None);
    }

    #[test]
    fn apply_many_sets_and_deletes_in_one_transaction() {
        let dir = tempfile::tempdir().unwrap();
        let store = FingerprintStore::open(&dir.path().join("fp.redb")).unwrap();
        put(&store, "/a.txt", rec(1, 1, 1, &[1]));
        let mut updates = HashMap::new();
        updates.insert("/a.txt".to_string(), None);
        updates.insert("/b.txt".to_string(), Some(rec(2, 2, 2, &[2])));
        store.apply_many(&updates).unwrap();
        assert_eq!(store.get("/a.txt").unwrap(), None);
        assert_eq!(store.get("/b.txt").unwrap(), Some(rec(2, 2, 2, &[2])));
    }

    #[test]
    fn all_returns_every_entry() {
        let dir = tempfile::tempdir().unwrap();
        let store = FingerprintStore::open(&dir.path().join("fp.redb")).unwrap();
        put(&store, "/a.txt", rec(1, 1, 1, &[1]));
        put(&store, "/b.txt", rec(2, 2, 2, &[2, 3]));
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
            put(&store, "/a.txt", rec(1, 1, 1, &[1]));
        }
        let store = FingerprintStore::open(&path).unwrap();
        assert!(store.get("/a.txt").unwrap().is_some());
    }
}
