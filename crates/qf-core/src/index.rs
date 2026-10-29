//! Build an index from a folder: walk, prepare every file in parallel,
//! embed all chunks in batches, then write the store.

use std::path::Path;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::chunk::{ChunkConfig, Tokenize};
use crate::embed::{BATCH_SIZE, EmbedError, Embedder, MODEL_ID};
use crate::ingest::{Skipped, hash_file, walk};
use crate::prepare::prepare;
use crate::store::{Store, StoreError, VECTORS_FILE, VectorWriter, install, staging_dir};
use crate::{Chunk, Document, IndexMeta};

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("embedding failed: {0}")]
    Embed(#[from] EmbedError),
}

/// What an indexing run did and how long each part took.
#[derive(Debug, Default)]
pub struct IndexReport {
    pub documents: usize,
    pub chunks: usize,
    pub skipped: Vec<Skipped>,
    /// Walking, extracting, normalising and chunking.
    pub prepare_time: Duration,
    /// The embedding loop alone.
    pub embed_time: Duration,
    pub total_time: Duration,
}

/// The two long stages of an indexing run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Extracting and chunking files; counted in files.
    Preparing,
    /// Embedding chunks; counted in chunks.
    Embedding,
}

/// Receives indexing progress, for a progress bar in the CLI or the UI.
/// Methods may be called from several threads at once.
pub trait Progress: Sync {
    fn start(&self, _stage: Stage, _total: usize) {}
    fn advance(&self, _done: usize) {}
    fn finish(&self, _stage: Stage) {}
}

/// No progress reporting.
impl Progress for () {}

/// The settings an index built now with `embedder` and `config` records.
pub fn index_meta(embedder: &Embedder, config: ChunkConfig) -> IndexMeta {
    IndexMeta {
        model_id: MODEL_ID.to_string(),
        model_rev: embedder.revision().to_string(),
        dim: embedder.dim(),
        max_tokens: config.max_tokens,
        overlap: config.overlap,
    }
}

/// Index every supported file under `folder` into `index_dir`, replacing
/// whatever index was there. The new index is built beside it and only
/// moved into place once complete, so a failed or interrupted run keeps
/// the old one. Files that cannot be read are skipped with a reason and
/// never stop the run.
pub fn build_index(
    folder: &Path,
    index_dir: &Path,
    embedder: &Embedder,
    tokenizer: &(impl Tokenize + Sync),
    config: ChunkConfig,
    progress: &impl Progress,
) -> Result<IndexReport, IndexError> {
    let started = Instant::now();
    let (files, mut skipped) = walk(folder);
    progress.start(Stage::Preparing, files.len());
    let prepared: Vec<_> = files
        .into_par_iter()
        .map(|file| {
            let result = hash_file(&file.path)
                .map_err(|e| e.to_string())
                .and_then(|hash| {
                    prepare(&file.path, file.format, tokenizer, config)
                        .map(|passages| (hash, passages))
                        .map_err(|e| e.to_string())
                });
            progress.advance(1);
            (file, result)
        })
        .collect();
    progress.finish(Stage::Preparing);

    // Number documents and chunks in walk order; chunk ids are also rows of
    // the vector file.
    let mut documents = Vec::new();
    let mut chunks = Vec::new();
    for (file, result) in prepared {
        let (hash, passages) = match result {
            Ok(ok) => ok,
            Err(reason) => {
                skipped.push(Skipped {
                    path: file.path,
                    reason,
                });
                continue;
            }
        };
        let doc_id = documents.len() as u32;
        for (ord, p) in passages.into_iter().enumerate() {
            chunks.push(Chunk {
                id: chunks.len() as u32,
                doc_id,
                ord: ord as u32,
                text: p.text,
                char_start: p.char_start,
                char_end: p.char_end,
                page: p.page,
            });
        }
        documents.push(Document {
            id: doc_id,
            path: file.path,
            format: file.format,
            hash,
            mtime: file.mtime,
        });
    }
    let prepare_time = started.elapsed();

    let staging = staging_dir(index_dir);
    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(StoreError::from)?;
    }
    let mut store = Store::create(&staging)?;
    let embed_started = Instant::now();
    let mut vectors = VectorWriter::create(&staging.join(VECTORS_FILE), embedder.dim())?;
    progress.start(Stage::Embedding, chunks.len());
    for batch in chunks.chunks(BATCH_SIZE) {
        let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
        for vector in embedder.embed(&texts)? {
            vectors.push(&vector)?;
        }
        progress.advance(batch.len());
    }
    vectors.finish()?;
    progress.finish(Stage::Embedding);
    let embed_time = embed_started.elapsed();

    store.insert_documents(&documents)?;
    store.insert_chunks(&chunks)?;
    // Written last: an index whose run was cut short has no meta and is
    // refused instead of being searched half-built.
    store.write_meta(&index_meta(embedder, config))?;
    // Close the database before the folder is renamed (Windows refuses to
    // move open files).
    drop(store);
    install(&staging, index_dir)?;

    Ok(IndexReport {
        documents: documents.len(),
        chunks: chunks.len(),
        skipped,
        prepare_time,
        embed_time,
        total_time: started.elapsed(),
    })
}
