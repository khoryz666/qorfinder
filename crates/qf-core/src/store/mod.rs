//! The on-disk index: SQLite for documents and chunks, one flat file for
//! the vectors.

use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, params};

use crate::{Chunk, Document, Format};

/// SQLite database inside the index folder.
pub const DB_FILE: &str = "index.sqlite";

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("index database: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("index file: {0}")]
    Io(#[from] std::io::Error),
    #[error("index is damaged: {0}")]
    Corrupt(String),
}

const SCHEMA: &str = "
CREATE TABLE documents (
    id     INTEGER PRIMARY KEY,
    path   TEXT NOT NULL,
    format TEXT NOT NULL,
    hash   BLOB NOT NULL,
    mtime  INTEGER NOT NULL
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

    pub fn insert_document(&self, doc: &Document) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT INTO documents (id, path, format, hash, mtime) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                doc.id,
                doc.path.to_string_lossy(),
                doc.format.as_str(),
                doc.hash.as_slice(),
                doc.mtime
            ],
        )?;
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

fn db_path(dir: &Path) -> PathBuf {
    dir.join(DB_FILE)
}
