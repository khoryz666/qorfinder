use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};
use tokio::sync::mpsc;

use crate::embedder::Embedder;
use crate::indexer::{Indexer, canonical_identity};
use crate::store::Store;

/// Watch `root` for file changes, debouncing events into one batch per
/// `debounce` interval, and re-index (or un-index) affected files.
///
/// The store is opened fresh for each batch and dropped once it's applied,
/// rather than held open for the watcher's whole lifetime: `Store::open`
/// takes an exclusive lock (tantivy's `IndexWriter` lock), so holding it
/// continuously would permanently lock out `query`/`stats`/`eval`/`forget`
/// from another process for as long as the watcher runs. Closing it between
/// batches means those commands only collide with the watcher during the
/// brief window it takes to apply one batch, not for its entire lifetime.
pub async fn watch(
    root: PathBuf,
    index_dir: PathBuf,
    embedder: Arc<Embedder>,
    chunk_size: usize,
    chunk_overlap: usize,
    debounce: Duration,
) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<PathBuf>();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        let Ok(event) = res else { return };
        if event.kind.is_access() {
            return;
        }
        for path in event.paths {
            let _ = tx.send(path);
        }
    })
    .context("failed to create file watcher")?;
    watcher
        .watch(&root, RecursiveMode::Recursive)
        .with_context(|| format!("failed to watch {}", root.display()))?;
    tracing::info!(
        "watching {} for changes (debounce {:?})",
        root.display(),
        debounce
    );

    let mut pending: HashSet<PathBuf> = HashSet::new();
    let mut ticker = tokio::time::interval(debounce);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("received Ctrl+C, stopping watcher");
                break;
            }
            Some(path) = rx.recv() => {
                pending.insert(path);
            }
            _ = ticker.tick() => {
                if pending.is_empty() {
                    continue;
                }
                let batch = std::mem::take(&mut pending);
                apply_pending(&index_dir, &embedder, chunk_size, chunk_overlap, batch).await;
            }
        }
    }
    Ok(())
}

/// Re-open the store, apply every pending path, then let it drop (releasing
/// the lock) at the end of this call.
async fn apply_pending(
    index_dir: &Path,
    embedder: &Arc<Embedder>,
    chunk_size: usize,
    chunk_overlap: usize,
    pending: HashSet<PathBuf>,
) {
    let store = match Store::open(index_dir, embedder.dims()) {
        Ok(store) => store,
        Err(err) => {
            tracing::warn!("failed to reopen index for incremental update: {err:#}");
            return;
        }
    };
    let indexer = Indexer::new(store, embedder.clone(), chunk_size, chunk_overlap);
    for path in pending {
        if path.is_dir() {
            continue;
        }
        let Some(identity) = canonical_identity(&path) else {
            continue;
        };
        if identity.is_file() && crate::parser::is_supported(&identity) {
            if let Err(err) = indexer.index_file(&identity).await {
                tracing::warn!("failed to re-index {}: {err:#}", identity.display());
            }
        } else if let Err(err) = indexer.delete_file(&identity).await {
            tracing::warn!("failed to un-index {}: {err:#}", identity.display());
        }
    }
}
