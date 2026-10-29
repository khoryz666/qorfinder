//! Build an index from a folder: walk, prepare every file in parallel,
//! embed all chunks in batches, then write the store.

use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::chunk::{ChunkConfig, Tokenize};
use crate::embed::{BATCH_SIZE, EmbedError, Embedder, MODEL_ID};
use crate::ingest::{Skipped, hash_file, walk_except};
use crate::prepare::prepare;
use crate::store::{
    Store, StoreError, VECTORS_FILE, VectorWriter, ensure_replaceable, install,
    is_index_folder_name, staging_dir,
};
use crate::{Chunk, Document, IndexMeta};

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("embedding failed: {0}")]
    Embed(#[from] EmbedError),
    #[error(
        "the indexed folder {} is the index folder {} or inside it; choose \
         another folder for the index",
        folder.display(),
        index.display()
    )]
    FolderInsideIndex { index: PathBuf, folder: PathBuf },
}

/// What an indexing run did and how long each part took.
#[derive(Debug, Default)]
pub struct IndexReport {
    pub documents: usize,
    pub chunks: usize,
    pub skipped: Vec<Skipped>,
    /// Why the replaced index could not be deleted, if it could not. The
    /// new index is in place either way.
    pub cleanup_warning: Option<String>,
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
        weights_sha256: embedder.weights_sha256().to_string(),
        dim: embedder.dim(),
        max_tokens: config.max_tokens,
        overlap: config.overlap,
    }
}

/// Check that indexing `folder` into `index_dir` cannot touch the user's
/// files: the documents may not be in the index folder (or be it), and the
/// index folder and its staging folder must be missing, empty or an index.
/// The index may sit inside the documents, as the default one does when a
/// whole user folder is indexed; [`build_index`] then leaves it out of the
/// walk. Run before any work, so a mistyped path fails at once.
pub fn check_index_dir(folder: &Path, index_dir: &Path) -> Result<(), IndexError> {
    let index = resolve(index_dir).map_err(StoreError::from)?;
    let source = resolve(folder).map_err(StoreError::from)?;
    if source.starts_with(&index) {
        return Err(IndexError::FolderInsideIndex {
            index: index_dir.to_path_buf(),
            folder: folder.to_path_buf(),
        });
    }
    ensure_replaceable(index_dir)?;
    ensure_replaceable(&staging_dir(index_dir))?;
    Ok(())
}

/// Whether `path` is `index_dir` or a folder an install makes beside it
/// (`.partial`, `.old`, `.old-N`), for leaving them out of the walk when
/// the index sits inside the indexed folder.
fn index_folders(index_dir: &Path) -> impl Fn(&Path) -> bool {
    let index = resolve(index_dir).ok();
    move |path| {
        let Some(index) = &index else {
            return false;
        };
        let (Some(name), Some(index_name)) = (path.file_name(), index.file_name()) else {
            return false;
        };
        is_index_folder_name(index_name, name)
            && path.parent().and_then(|p| resolve(p).ok()).as_deref() == index.parent()
    }
}

/// `path` made absolute with symbolic links and `..` resolved, also when it
/// does not exist yet: its nearest existing ancestor is canonicalised and
/// the rest appended. The appended parts do not exist, so they hold no
/// symbolic links and their `..` can be resolved by dropping a component.
fn resolve(path: &Path) -> std::io::Result<PathBuf> {
    let path = std::path::absolute(path)?;
    let mut existing = path.as_path();
    let mut rest = Vec::new();
    let mut resolved = loop {
        match existing.canonicalize() {
            Ok(base) => break base,
            Err(err) => {
                let (Some(parent), Some(last)) =
                    (existing.parent(), existing.components().next_back())
                else {
                    return Err(err);
                };
                rest.push(last);
                existing = parent;
            }
        }
    };
    for part in rest.into_iter().rev() {
        match part {
            Component::ParentDir => {
                resolved.pop();
            }
            Component::CurDir => {}
            part => resolved.push(part),
        }
    }
    Ok(resolved)
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
    check_index_dir(folder, index_dir)?;
    let started = Instant::now();
    let (files, mut skipped) = walk_except(folder, index_folders(index_dir));
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
    // A batch is padded to its longest text, so batches of similar lengths
    // waste less work: embed in order of length, then write every vector
    // back at its chunk's row.
    let mut by_length: Vec<usize> = (0..chunks.len()).collect();
    by_length.sort_by_key(|&i| chunks[i].text.len());
    let mut embedded: Vec<Vec<f32>> = vec![Vec::new(); chunks.len()];
    for batch in by_length.chunks(BATCH_SIZE) {
        let texts: Vec<&str> = batch.iter().map(|&i| chunks[i].text.as_str()).collect();
        for (&i, vector) in batch.iter().zip(embedder.embed(&texts)?) {
            embedded[i] = vector;
        }
        progress.advance(batch.len());
    }
    for vector in &embedded {
        vectors.push(vector)?;
    }
    vectors.finish()?;
    progress.finish(Stage::Embedding);
    let embed_time = embed_started.elapsed();

    store.insert_documents(&documents)?;
    store.insert_chunks(&chunks)?;
    // Written last: an index whose run was cut short has no meta and is
    // refused instead of being searched half-built.
    // Absolute, so a rebuild works whatever folder it is started from.
    let source = std::path::absolute(folder).map_err(StoreError::from)?;
    store.write_source_folder(&source)?;
    store.write_meta(&index_meta(embedder, config))?;
    // Close the database before the folder is renamed (Windows refuses to
    // move open files).
    drop(store);
    let cleanup_warning = install(&staging, index_dir)?.map(|err| err.to_string());

    Ok(IndexReport {
        documents: documents.len(),
        chunks: chunks.len(),
        skipped,
        cleanup_warning,
        prepare_time,
        embed_time,
        total_time: started.elapsed(),
    })
}
