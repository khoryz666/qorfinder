//! The on-disk index: SQLite for documents and chunks, one flat file for
//! the vectors.

mod vectors;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub use vectors::{VectorWriter, read_vectors};

use rusqlite::{Connection, OptionalExtension, params};

use crate::{Chunk, Document, Format, IndexMeta};

/// SQLite database inside the index folder.
pub const DB_FILE: &str = "index.sqlite";
/// Embedding vectors inside the index folder.
pub const VECTORS_FILE: &str = "vectors.f32";

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("index database: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("index file: {0}")]
    Io(#[from] std::io::Error),
    #[error("index is damaged: {0}")]
    Corrupt(String),
    #[error(
        "index was built with {field} {built}, but this run uses {current}; \
         re-run `qorfinder index` to rebuild it"
    )]
    Mismatch {
        field: &'static str,
        built: String,
        current: String,
    },
}

const SCHEMA: &str = "
CREATE TABLE documents (
    id     INTEGER PRIMARY KEY,
    path   TEXT NOT NULL,
    format TEXT NOT NULL,
    hash   BLOB NOT NULL,
    mtime  INTEGER NOT NULL
);
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE chunks (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id),
    ord        INTEGER NOT NULL,
    text       TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    page       INTEGER
);
";

/// The folder an index is built in before it replaces `dir`: a sibling
/// named like `dir` plus `.partial`. Building there means a run that fails
/// or is interrupted leaves the previous index usable.
pub fn staging_dir(dir: &Path) -> PathBuf {
    sibling(dir, ".partial")
}

/// Move a finished index from `staging` into `dir`, replacing the index
/// that was there. The old index is moved aside first and deleted only
/// once the new one is in place.
pub fn install(staging: &Path, dir: &Path) -> Result<(), StoreError> {
    let old = sibling(dir, ".old");
    if old.exists() {
        std::fs::remove_dir_all(&old)?;
    }
    if dir.exists() {
        std::fs::rename(dir, &old)?;
    }
    if let Err(err) = std::fs::rename(staging, dir) {
        if old.exists() {
            std::fs::rename(&old, dir)?;
        }
        return Err(err.into());
    }
    if old.exists() {
        std::fs::remove_dir_all(&old)?;
    }
    Ok(())
}

/// `dir` with `suffix` appended to its last component.
fn sibling(dir: &Path, suffix: &str) -> PathBuf {
    let mut name = dir
        .file_name()
        .map_or_else(|| OsString::from("index"), OsString::from);
    name.push(suffix);
    dir.with_file_name(name)
}

/// Documents and chunks of one index.
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Start a new, empty index in `dir`, replacing any index already there.
    /// FYP1 always rebuilds from scratch (FR8).
    pub fn create(dir: &Path) -> Result<Self, StoreError> {
        std::fs::create_dir_all(dir)?;
        let path = db_path(dir);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        let conn = Connection::open(&path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    /// Open the index in `dir`.
    pub fn open(dir: &Path) -> Result<Self, StoreError> {
        let path = db_path(dir);
        if !path.is_file() {
            return Err(StoreError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "no index at {} (run `qorfinder index` first)",
                    dir.display()
                ),
            )));
        }
        Ok(Self {
            conn: Connection::open(path)?,
        })
    }

    /// Record the settings the index is being built with.
    pub fn write_meta(&self, meta: &IndexMeta) -> Result<(), StoreError> {
        let rows = [
            ("model_id", meta.model_id.clone()),
            ("weights_sha256", meta.weights_sha256.clone()),
            ("dim", meta.dim.to_string()),
            ("max_tokens", meta.max_tokens.to_string()),
            ("overlap", meta.overlap.to_string()),
        ];
        for (key, value) in rows {
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
                params![key, value],
            )?;
        }
        Ok(())
    }

    /// Record the folder the index was built from, so it can be rebuilt
    /// from the same place later (FR8). Not part of the settings check.
    pub fn write_source_folder(&self, folder: &Path) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('source_folder', ?1)",
            [folder.to_string_lossy()],
        )?;
        Ok(())
    }

    /// The folder the index was built from, if recorded.
    pub fn source_folder(&self) -> Result<Option<PathBuf>, StoreError> {
        Ok(self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'source_folder'",
                [],
                |r| r.get::<_, String>(0),
            )
            .optional()?
            .map(PathBuf::from))
    }

    pub fn read_meta(&self) -> Result<IndexMeta, StoreError> {
        let get = |key: &str| -> Result<String, StoreError> {
            self.conn
                .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
                .optional()?
                .ok_or_else(|| StoreError::Corrupt(format!("missing meta {key}")))
        };
        let number = |key: &str| -> Result<usize, StoreError> {
            get(key)?
                .parse()
                .map_err(|_| StoreError::Corrupt(format!("meta {key} is not a number")))
        };
        Ok(IndexMeta {
            model_id: get("model_id")?,
            weights_sha256: get("weights_sha256")?,
            dim: number("dim")?,
            max_tokens: number("max_tokens")?,
            overlap: number("overlap")?,
        })
    }

    /// Fail unless the index was built with exactly `current`'s settings.
    pub fn check_meta(&self, current: &IndexMeta) -> Result<IndexMeta, StoreError> {
        let built = self.read_meta()?;
        let fields = [
            ("model", &built.model_id, &current.model_id),
            (
                "model weights (SHA-256)",
                &built.weights_sha256,
                &current.weights_sha256,
            ),
        ];
        for (field, b, c) in fields {
            if b != c {
                return Err(mismatch(field, b, c));
            }
        }
        let numbers = [
            ("dimension", built.dim, current.dim),
            ("max_tokens", built.max_tokens, current.max_tokens),
            ("overlap", built.overlap, current.overlap),
        ];
        for (field, b, c) in numbers {
            if b != c {
                return Err(mismatch(field, b, c));
            }
        }
        Ok(built)
    }

    /// Insert many documents in one transaction.
    pub fn insert_documents(&mut self, docs: &[Document]) -> Result<(), StoreError> {
        let tx = self.conn.transaction()?;
        {
            let mut insert = tx.prepare(
                "INSERT INTO documents (id, path, format, hash, mtime) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for doc in docs {
                insert.execute(params![
                    doc.id,
                    doc.path.to_string_lossy(),
                    doc.format.as_str(),
                    doc.hash.as_slice(),
                    doc.mtime
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Insert many chunks in one transaction.
    pub fn insert_chunks(&mut self, chunks: &[Chunk]) -> Result<(), StoreError> {
        let tx = self.conn.transaction()?;
        {
            let mut insert = tx.prepare(
                "INSERT INTO chunks (id, doc_id, ord, text, char_start, char_end, page)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for c in chunks {
                insert.execute(params![
                    c.id,
                    c.doc_id,
                    c.ord,
                    c.text,
                    c.char_start as i64,
                    c.char_end as i64,
                    c.page
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn document(&self, id: u32) -> Result<Option<Document>, StoreError> {
        let row = self
            .conn
            .query_row(
                "SELECT path, format, hash, mtime FROM documents WHERE id = ?1",
                [id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Vec<u8>>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;
        let Some((path, format, hash, mtime)) = row else {
            return Ok(None);
        };
        let format = Format::parse(&format)
            .ok_or_else(|| StoreError::Corrupt(format!("unknown format {format:?}")))?;
        let hash = hash
            .try_into()
            .map_err(|_| StoreError::Corrupt(format!("bad hash for document {id}")))?;
        Ok(Some(Document {
            id,
            path: PathBuf::from(path),
            format,
            hash,
            mtime,
        }))
    }

    pub fn chunk(&self, id: u32) -> Result<Option<Chunk>, StoreError> {
        Ok(self
            .conn
            .query_row(
                "SELECT doc_id, ord, text, char_start, char_end, page FROM chunks WHERE id = ?1",
                [id],
                |r| {
                    Ok(Chunk {
                        id,
                        doc_id: r.get(0)?,
                        ord: r.get(1)?,
                        text: r.get(2)?,
                        char_start: r.get::<_, i64>(3)? as usize,
                        char_end: r.get::<_, i64>(4)? as usize,
                        page: r.get(5)?,
                    })
                },
            )
            .optional()?)
    }

    pub fn chunk_count(&self) -> Result<usize, StoreError> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok(n as usize)
    }
}

fn mismatch(field: &'static str, built: impl ToString, current: impl ToString) -> StoreError {
    StoreError::Mismatch {
        field,
        built: built.to_string(),
        current: current.to_string(),
    }
}

fn db_path(dir: &Path) -> PathBuf {
    dir.join(DB_FILE)
}
