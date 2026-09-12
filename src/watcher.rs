use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};
use tokio::sync::mpsc;

use crate::indexer::{Indexer, canonical_identity};

/// Watch `root` for file changes, debouncing events into one batch per
/// `debounce` interval, and re-index (or un-index) affected files.
pub async fn watch(root: PathBuf, indexer: Arc<Indexer>, debounce: Duration) -> Result<()> {
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
                for path in pending.drain() {
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
        }
    }
    Ok(())
}
