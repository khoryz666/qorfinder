use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use qf_core::chunk::{ChunkConfig, ModelTokenizer};
use qf_core::embed::Embedder;
use qf_core::format::snippet;
use qf_core::index::build_index;
use qf_core::ingest::{Skipped, walk};
use qf_core::prepare::prepare;

use crate::progress::Bars;

#[derive(Parser)]
#[command(
    name = "qorfinder",
    version,
    about = "Offline semantic search over local legal documents"
)]
pub struct Cli {
    /// Folder holding the index. Defaults to qorfinder/index in the
    /// user's local data folder
    #[arg(long, global = true, env = "QORFINDER_INDEX_DIR")]
    index_dir: Option<PathBuf>,

    /// Folder holding the embedding model files (config.json,
    /// tokenizer.json, model.safetensors)
    #[arg(
        long,
        global = true,
        env = "QORFINDER_MODEL_DIR",
        default_value = "models/all-MiniLM-L6-v2"
    )]
    model_dir: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Index every PDF, DOCX and TXT file under a folder, replacing the
    /// previous index
    Index {
        /// Folder to index
        folder: PathBuf,

        /// Print the chunks of every file instead of indexing them
        #[arg(long)]
        dry_run: bool,

        #[command(flatten)]
        chunking: ChunkArgs,
    },
}

/// Chunk settings. An index can only be searched with the settings it was
/// built with.
#[derive(Args)]
struct ChunkArgs {
    /// Most tokens in one chunk
    #[arg(long, default_value_t = ChunkConfig::default().max_tokens)]
    max_tokens: usize,

    /// Tokens shared by consecutive windows of an over-long paragraph
    #[arg(long, default_value_t = ChunkConfig::default().overlap)]
    overlap: usize,
}

impl ChunkArgs {
    fn config(&self) -> Result<ChunkConfig> {
        if self.overlap >= self.max_tokens {
            bail!("--overlap must be smaller than --max-tokens");
        }
        Ok(ChunkConfig {
            max_tokens: self.max_tokens,
            overlap: self.overlap,
        })
    }
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Index {
            folder,
            dry_run,
            chunking,
        } => {
            let config = chunking.config()?;
            if dry_run {
                return dry_run_index(&folder, &cli.model_dir, config);
            }
            let index_dir = cli.index_dir.unwrap_or_else(default_index_dir);
            index(&folder, &index_dir, &cli.model_dir, config)
        }
    }
}

fn default_index_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("qorfinder")
        .join("index")
}

fn index(folder: &Path, index_dir: &Path, model_dir: &Path, config: ChunkConfig) -> Result<()> {
    if !folder.is_dir() {
        bail!("{} is not a folder", folder.display());
    }
    let embedder = load_embedder(model_dir)?;
    let tokenizer = load_tokenizer(model_dir)?;
    let report = build_index(
        folder,
        index_dir,
        &embedder,
        &tokenizer,
        config,
        &Bars::default(),
    )?;
    for s in &report.skipped {
        println!("skipped {}: {}", s.path.display(), s.reason);
    }
    let embed_secs = report.embed_time.as_secs_f64();
    println!(
        "indexed {} document(s), {} chunk(s), {} skipped, in {:.1} s \
         (prepare {:.1} s, embed {:.1} s, {:.1} chunks/s)",
        report.documents,
        report.chunks,
        report.skipped.len(),
        report.total_time.as_secs_f64(),
        report.prepare_time.as_secs_f64(),
        embed_secs,
        report.chunks as f64 / embed_secs.max(f64::EPSILON),
    );
    println!("index saved to {}", index_dir.display());
    Ok(())
}

/// Print every chunk the indexer would store, with its offsets into the
/// extracted text and its page, without embedding anything.
fn dry_run_index(folder: &Path, model_dir: &Path, config: ChunkConfig) -> Result<()> {
    let tokenizer = load_tokenizer(model_dir)?;
    let (files, mut skipped) = walk(folder);
    let (mut prepared, mut chunks) = (0, 0);
    for file in &files {
        match prepare(&file.path, file.format, &tokenizer, config) {
            Ok(passages) => {
                println!("{} ({} chunks)", file.path.display(), passages.len());
                for (ord, p) in passages.iter().enumerate() {
                    let page = p.page.map_or(String::new(), |n| format!(" p.{n}"));
                    println!(
                        "  #{ord} [{}..{}]{page} {}",
                        p.char_start,
                        p.char_end,
                        snippet(&p.text, 100)
                    );
                }
                prepared += 1;
                chunks += passages.len();
            }
            Err(err) => skipped.push(Skipped {
                path: file.path.clone(),
                reason: err.to_string(),
            }),
        }
    }
    for s in &skipped {
        println!("skipped {}: {}", s.path.display(), s.reason);
    }
    println!(
        "{prepared} file(s), {chunks} chunk(s), {} skipped",
        skipped.len()
    );
    Ok(())
}

fn load_embedder(model_dir: &Path) -> Result<Embedder> {
    Embedder::load(model_dir).with_context(|| {
        format!(
            "cannot load the model from {} (run scripts/fetch_model.py once to download it)",
            model_dir.display()
        )
    })
}

fn load_tokenizer(model_dir: &Path) -> Result<ModelTokenizer> {
    let path = model_dir.join("tokenizer.json");
    ModelTokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!(e))
        .with_context(|| {
            format!(
                "cannot load {} (run scripts/fetch_model.py once to download the model)",
                path.display()
            )
        })
}
