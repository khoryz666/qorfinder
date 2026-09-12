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
/// parallel sub-batches internally).
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

    /// Walk `root` and index every supported file. Files whose (mtime, size)
    /// match the points already in the store are skipped, so re-running the
    /// command only re-embeds what actually changed. Points of files that no
    /// longer exist under `root` are removed. Parsing/chunking runs on a
    /// rayon pool, embeddings and upserts are batched.
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
            self.store.file_fingerprints().await?
        };

        // Remove points of files that vanished from disk (restricted to paths
        // under `root`, so other directories in the same collection survive).
        if !force {
            let known: HashSet<&str> = files.iter().filter_map(|p| p.to_str()).collect();
            for path_str in fingerprints.keys() {
                if known.contains(path_str.as_str()) {
                    continue;
                }
                let path = Path::new(path_str);
                if path.starts_with(root) {
                    if let Err(err) = self.store.delete_file(path).await {
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

        // Remove stale points of changed files via deterministic IDs (no scan).
        for (path, _) in &changed {
            if let Some(count) = fingerprints
                .get(&path.display().to_string())
                .and_then(|fp| fp.chunk_count)
            {
                self.store.delete_file_ids(path, count).await?;
            }
        }
        for path in &emptied {
            self.store.delete_file(path).await?;
        }

        // Embed in big batches and upsert as we go; only the last request
        // waits for durability.
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
        let mut embedded = 0usize;
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
                    embedded += batch_texts.len();
                    self.embed_and_upsert(
                        &batch_refs,
                        &batch_texts,
                        &file_fps,
                        embedded == total_chunks,
                    )
                    .await?;
                }
            }
        }
        if !texts.is_empty() {
            let batch_texts: Vec<String> = std::mem::take(&mut texts)
                .into_iter()
                .map(str::to_string)
                .collect();
            let batch_refs: Vec<(&Path, u64)> = std::mem::take(&mut refs);
            self.embed_and_upsert(&batch_refs, &batch_texts, &file_fps, true)
                .await?;
        }

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

    /// Embed `texts` and upsert the resulting points in batches.
    async fn embed_and_upsert(
        &self,
        refs: &[(&Path, u64)],
        texts: &[String],
        file_fps: &HashMap<&Path, (i64, i64)>,
        wait: bool,
    ) -> Result<()> {
        let vectors = self.embedder.embed_passages(texts)?;
        let points: Vec<_> = refs
            .iter()
            .zip(texts)
            .zip(vectors)
            .map(|(((path, chunk_index), text), vector)| {
                let fp = file_fps.get(path).copied().unwrap_or((0, 0));
                Store::point(&path.display().to_string(), *chunk_index, text, vector, fp)
            })
            .collect();
        self.store.upsert_points(points, wait).await?;
        Ok(())
    }

    /// Parse, chunk, embed and upsert a single file. Skips the file when its
    /// (mtime, size) matches the points already stored; replaces any existing
    /// points otherwise.
    pub async fn index_file(&self, path: &Path) -> Result<()> {
        let started = Instant::now();
        let meta = std::fs::metadata(path)
            .with_context(|| format!("failed to stat {}", path.display()))?;
        let old = self.store.file_info(path).await?;
        if old.matches(&meta) {
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
        let mut vectors = Vec::with_capacity(chunks.len());
        for chunk_batch in chunks.chunks(EMBED_BATCH) {
            let batch_vectors = self.embedder.embed_passages(chunk_batch)?;
            vectors.extend(batch_vectors);
        }
        if let Some(count) = old.chunk_count {
            self.store.delete_file_ids(path, count).await?;
        }
        self.store
            .upsert_chunks(path, &chunks, &vectors, file_fingerprint(&meta))
            .await?;
        tracing::info!(
            "indexed {} ({} chunks) in {:?}",
            path.display(),
            chunks.len(),
            started.elapsed()
        );
        Ok(())
    }

    /// Remove every point belonging to `path` from the store.
    pub async fn delete_file(&self, path: &Path) -> Result<()> {
        self.store.delete_file(path).await
    }
}

/// Resolve `path` to an absolute, canonical path used as the payload identity.
/// Falls back to canonicalizing the parent directory so deleted files can
/// still be matched against their indexed identity. Uses `dunce` so Windows
/// paths don't carry the `\\?\` verbatim prefix.
pub fn canonical_identity(path: &Path) -> Option<PathBuf> {
    if let Ok(canonical) = dunce::canonicalize(path) {
        return Some(canonical);
    }
    let parent = path.parent()?;
    let name = path.file_name()?;
    let canonical_parent = dunce::canonicalize(parent).ok()?;
    Some(canonical_parent.join(name))
}
