use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use crate::embedder::Embedder;
use crate::indexer::Indexer;
use crate::store::Store;

#[derive(Parser)]
#[command(
    name = "qorfinder",
    version,
    about = "Local-first semantic search: index a directory into an embedded hybrid store and query it with sentence embeddings"
)]
pub struct Cli {
    /// Directory holding the embedded index (tantivy + usearch + redb).
    /// Defaults to ~/.cache/qorfinder/indexes/default
    #[arg(long, global = true, env = "QORFINDER_INDEX_DIR")]
    index_dir: Option<PathBuf>,

    /// Directory where the embedding model is cached
    #[arg(long, global = true, env = "QORFINDER_MODEL_CACHE")]
    model_cache: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

/// Stable per-user default index location: `~/.cache/qorfinder/indexes/default`.
fn default_index_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache")
        .join("qorfinder")
        .join("indexes")
        .join(crate::store::DEFAULT_INDEX_DIR_NAME)
}

#[derive(Subcommand)]
enum Command {
    /// Index all supported files in a directory, then keep watching for changes
    Index {
        /// Directory to index (and watch)
        dir: PathBuf,

        /// Chunk size in characters
        #[arg(long, default_value_t = 512)]
        chunk_size: usize,

        /// Overlap between consecutive chunks in characters
        #[arg(long, default_value_t = 64)]
        chunk_overlap: usize,

        /// Index once and exit (skip watching)
        #[arg(long)]
        once: bool,

        /// Re-index every file even if its stored fingerprint is unchanged
        #[arg(long)]
        force: bool,
    },
    /// Search the index for the top-k matching chunks
    Query {
        /// The search query
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: u64,
    },
    /// Show the number of indexed points
    Stats,
    /// Evaluate retrieval quality against qrels (nDCG@k, Recall@k, MRR@k)
    Eval {
        /// Directory with one file per document (doc id = file stem); must already be indexed
        corpus: PathBuf,

        /// TSV file: query id, query text
        queries: PathBuf,

        /// Qrels in TREC or BEIR format: query id, [0,] doc id, relevance
        qrels: PathBuf,

        /// Cut-off for nDCG@k / Recall@k / MRR@k
        #[arg(short = 'k', long, default_value_t = 10)]
        top_k: usize,

        /// Evaluate only the first N queries (quick smoke runs)
        #[arg(long)]
        limit: Option<usize>,

        /// Retrieval path to score: the dense side alone, the lexical side
        /// alone, or the hybrid fusion the `query` command actually uses
        #[arg(long, value_enum, default_value = "hybrid")]
        mode: crate::eval::EvalMode,
    },
    /// Remove all indexed files under a directory from the index, without
    /// touching anything on disk (use this to stop tracking a directory you
    /// no longer want searched)
    Forget {
        /// Directory whose indexed files should be removed from the index
        dir: PathBuf,
    },
    /// Download and prepare a benchmark corpus (no server needed)
    Corpus {
        #[command(subcommand)]
        corpus: CorpusKind,
    },
    /// Download the embedding model and print its info
    Warm,
}

#[derive(Subcommand)]
enum CorpusKind {
    /// BEIR retrieval benchmark (scifact or nfcorpus)
    Beir {
        /// Dataset name
        #[arg(default_value = "scifact")]
        dataset: String,
        /// Output directory (default: data)
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
    },
    /// Quran corpus: Tanzil Uthmani text + English translation, one file per ayah
    Quran {
        /// Output directory (default: data)
        #[arg(short, long, default_value = "data")]
        out: PathBuf,
    },
}

pub fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let rt = tokio::runtime::Runtime::new().context("failed to start async runtime")?;
    rt.block_on(async move { execute(cli).await })
}

async fn execute(cli: Cli) -> Result<()> {
    match &cli.command {
        Command::Warm => return warm_model(cli.model_cache.clone()),
        Command::Corpus { corpus } => {
            return match corpus {
                CorpusKind::Beir { dataset, out } => {
                    let dir = crate::corpus::prepare_beir(dataset, out)?;
                    println!("corpus ready: {}", dir.display());
                    Ok(())
                }
                CorpusKind::Quran { out } => {
                    let dir = crate::corpus::prepare_quran(out)?;
                    println!("corpus ready: {}", dir.display());
                    Ok(())
                }
            };
        }
        _ => {}
    }

    let embedder = Arc::new(
        Embedder::try_new(cli.model_cache.clone()).context("failed to load embedding model")?,
    );
    let index_dir = cli.index_dir.clone().unwrap_or_else(default_index_dir);
    let store = Store::open(&index_dir, embedder.dims()).with_context(|| {
        format!(
            "failed to open index at {} (delete it to rebuild from scratch)",
            index_dir.display()
        )
    })?;

    match cli.command {
        Command::Index {
            dir,
            chunk_size,
            chunk_overlap,
            once,
            force,
        } => {
            if chunk_size == 0 {
                bail!("--chunk-size must be positive");
            }
            if chunk_overlap >= chunk_size {
                bail!("--chunk-overlap must be smaller than --chunk-size");
            }
            let dir = std::fs::canonicalize(&dir)
                .with_context(|| format!("target directory not found: {}", dir.display()))?;
            let indexer = Indexer::new(store, embedder.clone(), chunk_size, chunk_overlap);
            let stats = indexer.index_dir(&dir, force).await?;
            tracing::info!(
                "indexing done: {} indexed, {} unchanged, {} removed, {} skipped, {} failed",
                stats.indexed,
                stats.unchanged,
                stats.removed,
                stats.skipped,
                stats.failed
            );
            // Release the index lock before watching: the watcher reopens
            // the store only for the brief window it takes to apply a batch
            // of changes, so `query`/`stats`/etc. from another process can
            // run the rest of the time (see watcher::watch's doc comment).
            drop(indexer);
            if !once {
                crate::watcher::watch(
                    dir,
                    index_dir,
                    embedder,
                    chunk_size,
                    chunk_overlap,
                    Duration::from_secs(2),
                )
                .await?;
            }
        }
        Command::Query { query, top_k } => {
            let started = Instant::now();
            let hits = crate::query::run_query(&store, &embedder, &query, top_k)?;
            print!(
                "{}",
                crate::format::format_hits(&query, &hits, started.elapsed())
            );
        }
        Command::Stats => {
            let count = store.count()?;
            println!("index '{}': {} chunk(s)", index_dir.display(), count);
        }
        Command::Eval {
            corpus,
            queries,
            qrels,
            top_k,
            limit,
            mode,
        } => {
            let report = crate::eval::run_eval(
                &store,
                &embedder,
                &corpus,
                &queries,
                &qrels,
                crate::eval::EvalOptions {
                    k: top_k,
                    limit,
                    mode,
                },
            )?;
            println!(
                "mode: {mode:?}, queries evaluated: {} (skipped, no qrels: {})",
                report.evaluated, report.skipped_no_qrels
            );
            println!("nDCG@{}:  {:.4}", report.k, report.ndcg);
            println!("Recall@{}: {:.4}", report.k, report.recall);
            println!("MRR@{}:    {:.4}", report.k, report.mrr);
            println!(
                "time: {:.2} s ({:.1} ms/query)",
                report.total_seconds,
                report.total_seconds * 1000.0 / report.evaluated.max(1) as f64
            );
        }
        Command::Forget { dir } => {
            let dir = std::fs::canonicalize(&dir)
                .with_context(|| format!("target directory not found: {}", dir.display()))?;
            let removed = store.forget_dir(&dir)?;
            println!(
                "removed {removed} file(s) under {} from the index",
                dir.display()
            );
        }
        Command::Corpus { .. } | Command::Warm => unreachable!("handled above"),
    }
    Ok(())
}

fn warm_model(model_cache: Option<PathBuf>) -> Result<()> {
    let embedder = Embedder::try_new(model_cache).context("failed to load embedding model")?;
    println!(
        "model ready: intfloat/multilingual-e5-small, {} dims",
        embedder.dims()
    );
    println!("cache: {}", embedder.cache_dir().display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reopening_after_the_lock_holder_drops_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        {
            let _held = Store::open(dir.path(), 3).unwrap();
            assert!(Store::open(dir.path(), 3).is_err());
        }
        // `_held` (and its tantivy IndexWriter) is dropped now, so the lock
        // it held is released — this is the mechanism the watcher relies on
        // to only hold the lock for the duration of one incremental update.
        assert!(Store::open(dir.path(), 3).is_ok());
    }
}
