use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::chunker::chunk_text;
use crate::embedder::Embedder;
use crate::parser::{is_supported, parse_file};
use crate::store::{Store, file_fingerprint};

/// Chunks embedded per model call (fastembed splits each call into its own
/// parallel sub-batches internally). Embedding batches may span multiple
/// files; `Store::stage_chunk` accumulates a file's chunks incrementally so
/// that's safe regardless of where a batch boundary falls.
const EMBED_BATCH: usize = 256;

#[derive(Debug, Default)]
pub struct DirStats {
    pub indexed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub skipped: usize,
    pub failed: usize,
}

pub struct Indexer {
    store: Store,
    embedder: Embedder,
    chunk_size: usize,
    overlap: usize,
}

enum FileOutcome {
    Unchanged,
    Empty,
    Changed(Vec<String>),
    Failed,
}

impl Indexer {
    pub fn new(store: Store, embedder: Embedder, chunk_size: usize, overlap: usize) -> Self {
        Self {
            store,
            embedder,
            chunk_size,
            overlap,
        }
    }

    /// The store this indexer writes to — exposed so a caller (tests,
    /// mainly) can query the same store right after indexing into it.
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The embedder this indexer embeds chunks with — see [`Self::store`].
    pub fn embedder(&self) -> &Embedder {
        &self.embedder
    }

    /// Walk `root` and index every supported file. Files whose (mtime, size)
    /// match the fingerprint already stored are skipped, so re-running the
    /// command only re-embeds what actually changed. Fingerprints of files
    /// that no longer exist under `root` are removed. Parsing/chunking runs
    /// on a rayon pool, embeddings run in batches; the whole scan commits
    /// once at the end (no network round trips to amortize per-batch).
    pub async fn index_dir(&self, root: &Path, force: bool) -> Result<DirStats> {
        let started = Instant::now();
        let mut stats = DirStats::default();

        // Collect supported files as canonical identities.
        let mut files: Vec<PathBuf> = Vec::new();
        for entry in WalkDir::new(root).follow_links(false) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(err) => {
                    stats.failed += 1;
                    tracing::warn!("failed to walk entry: {err}");
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            if !is_supported(entry.path()) {
                stats.skipped += 1;
                continue;
            }
            match canonical_identity(entry.path()) {
                Some(identity) => files.push(identity),
                None => {
                    stats.failed += 1;
                    tracing::warn!("failed to canonicalize {}", entry.path().display());
                }
            }
        }

        // What does the store already know about the indexed files?
        let fingerprints = if force {
            HashMap::new()
        } else {
            self.store.file_fingerprints()?
        };

        // Remove fingerprints of files that vanished from disk (restricted
        // to paths under `root`, so other directories in the same index
        // survive).
        if !force {
            let known: HashSet<&str> = files.iter().filter_map(|p| p.to_str()).collect();
            for path_str in fingerprints.keys() {
                if known.contains(path_str.as_str()) {
                    continue;
                }
                let path = Path::new(path_str);
                if path.starts_with(root) {
                    if let Err(err) = self.store.delete_file(path) {
                        tracing::warn!("failed to remove vanished file {path_str}: {err:#}");
                    } else {
                        stats.removed += 1;
                        tracing::info!("removed vanished file {path_str}");
                    }
                }
            }
        }

        // Parse + chunk in parallel; skip files whose fingerprint is current.
        let outcomes: Vec<(PathBuf, FileOutcome)> = files
            .into_par_iter()
            .map(|path| {
                let outcome = match std::fs::metadata(&path) {
                    Err(_) => FileOutcome::Failed,
                    Ok(meta) => {
                        let known = fingerprints.get(&path.display().to_string());
                        if known.is_some_and(|fp| fp.matches(&meta)) {
                            FileOutcome::Unchanged
                        } else {
                            match parse_file(&path) {
                                Err(err) => {
                                    tracing::warn!("failed to index {}: {err}", path.display());
                                    FileOutcome::Failed
                                }
                                Ok(text) => {
                                    let chunks = chunk_text(&text, self.chunk_size, self.overlap);
                                    if chunks.is_empty() {
                                        FileOutcome::Empty
                                    } else {
                                        FileOutcome::Changed(chunks)
                                    }
                                }
                            }
                        }
                    }
                };
                (path, outcome)
            })
            .collect();

        let mut changed: Vec<(PathBuf, Vec<String>)> = Vec::new();
        let mut emptied: Vec<PathBuf> = Vec::new();
        for (path, outcome) in outcomes {
            match outcome {
                FileOutcome::Unchanged => stats.unchanged += 1,
                FileOutcome::Empty => {
                    emptied.push(path);
                }
                FileOutcome::Changed(chunks) => {
                    stats.indexed += 1;
                    changed.push((path, chunks));
                }
                FileOutcome::Failed => {
                    stats.failed += 1;
                }
            }
        }

        // Clear old chunks before staging new ones for changed files, and
        // remove emptied files outright.
        for (path, _) in &changed {
            self.store.delete_file(path)?;
        }
        for path in &emptied {
            self.store.delete_file(path)?;
        }

        // Embed in big batches (spanning files) and stage as we go.
        let total_chunks: usize = changed.iter().map(|(_, c)| c.len()).sum();
        let file_fps: HashMap<&Path, (i64, i64)> = changed
            .iter()
            .map(|(path, _)| {
                let fp = std::fs::metadata(path)
                    .map(|m| file_fingerprint(&m))
                    .unwrap_or((0, 0));
                (path.as_path(), fp)
            })
            .collect();
        let mut texts: Vec<&str> = Vec::with_capacity(EMBED_BATCH);
        let mut refs: Vec<(&Path, u64)> = Vec::with_capacity(EMBED_BATCH);
        for (path, chunks) in &changed {
            for (i, chunk) in chunks.iter().enumerate() {
                texts.push(chunk.as_str());
                refs.push((path.as_path(), i as u64));
                if texts.len() >= EMBED_BATCH {
                    let batch_texts: Vec<String> = std::mem::take(&mut texts)
                        .into_iter()
                        .map(str::to_string)
                        .collect();
                    let batch_refs: Vec<(&Path, u64)> = std::mem::take(&mut refs);
                    self.embed_and_stage(&batch_refs, &batch_texts, &file_fps)?;
                }
            }
        }
        if !texts.is_empty() {
            let batch_texts: Vec<String> = std::mem::take(&mut texts)
                .into_iter()
                .map(str::to_string)
                .collect();
            let batch_refs: Vec<(&Path, u64)> = std::mem::take(&mut refs);
            self.embed_and_stage(&batch_refs, &batch_texts, &file_fps)?;
        }

        self.store.commit()?;

        tracing::info!(
            "indexed {} file(s) ({} chunks), {} unchanged, {} removed, {} skipped, {} failed in {:?}",
            stats.indexed,
            total_chunks,
            stats.unchanged,
            stats.removed,
            stats.skipped,
            stats.failed,
            started.elapsed()
        );
        Ok(stats)
    }

    /// Embed `texts` and stage the resulting chunks (uncommitted).
    fn embed_and_stage(
        &self,
        refs: &[(&Path, u64)],
        texts: &[String],
        file_fps: &HashMap<&Path, (i64, i64)>,
    ) -> Result<()> {
        let vectors = self.embedder.embed_passages(texts)?;
        for (((path, chunk_index), text), vector) in refs.iter().zip(texts).zip(vectors) {
            let fp = file_fps.get(path).copied().unwrap_or((0, 0));
            self.store
                .stage_chunk(path, *chunk_index, text, &vector, fp)?;
        }
        Ok(())
    }

    /// Parse, chunk, embed and stage a single file, then commit immediately
    /// (interactive freshness matters more than throughput for watch mode).
    /// Skips the file when its (mtime, size) matches what's already stored;
    /// replaces any existing chunks otherwise.
    pub async fn index_file(&self, path: &Path) -> Result<()> {
        let started = Instant::now();
        let meta = std::fs::metadata(path)
            .with_context(|| format!("failed to stat {}", path.display()))?;
        let old = self.store.file_info(path)?;
        if old.as_ref().is_some_and(|fp| fp.matches(&meta)) {
            tracing::debug!("unchanged, skipping {}", path.display());
            return Ok(());
        }
        let text = parse_file(path).with_context(|| format!("parsing {}", path.display()))?;
        if text.trim().is_empty() {
            self.delete_file(path).await?;
            tracing::debug!("skipped {}: no text", path.display());
            return Ok(());
        }
        let chunks = chunk_text(&text, self.chunk_size, self.overlap);
        let fp = file_fingerprint(&meta);
        self.store.delete_file(path)?;
        for start in (0..chunks.len()).step_by(EMBED_BATCH) {
            let end = (start + EMBED_BATCH).min(chunks.len());
            let batch = &chunks[start..end];
            let vectors = self.embedder.embed_passages(batch)?;
            for (offset, vector) in vectors.into_iter().enumerate() {
                let chunk_index = (start + offset) as u64;
                self.store
                    .stage_chunk(path, chunk_index, &batch[offset], &vector, fp)?;
            }
        }
        self.store.commit()?;
        tracing::info!(
            "indexed {} ({} chunks) in {:?}",
            path.display(),
            chunks.len(),
            started.elapsed()
        );
        Ok(())
    }

    /// Remove every chunk belonging to `path` from the store and commit.
    pub async fn delete_file(&self, path: &Path) -> Result<()> {
        self.store.delete_file(path)?;
        self.store.commit()?;
        Ok(())
    }
}

/// Resolve `path` to an absolute, canonical path used as the identity for
/// store lookups. Falls back to canonicalizing the parent directory so
/// deleted files can still be matched against their indexed identity.
pub fn canonical_identity(path: &Path) -> Option<PathBuf> {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return Some(canonical);
    }
    let parent = path.parent()?;
    let name = path.file_name()?;
    let canonical_parent = std::fs::canonicalize(parent).ok()?;
    Some(canonical_parent.join(name))
}
