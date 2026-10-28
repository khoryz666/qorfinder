use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use qf_core::chunk::{ChunkConfig, ModelTokenizer};
use qf_core::format::snippet;
use qf_core::ingest::{Skipped, walk};
use qf_core::prepare::prepare;

#[derive(Parser)]
#[command(
    name = "qorfinder",
    version,
    about = "Offline semantic search over local legal documents"
)]
pub struct Cli {
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
    /// Index every PDF, DOCX and TXT file under a folder
    Index {
        /// Folder to index
        folder: PathBuf,

        /// Print the chunks of every file instead of indexing them
        #[arg(long)]
        dry_run: bool,

        /// Most tokens in one chunk
        #[arg(long, default_value_t = ChunkConfig::default().max_tokens)]
        max_tokens: usize,

        /// Tokens shared by consecutive windows of an over-long paragraph
        #[arg(long, default_value_t = ChunkConfig::default().overlap)]
        overlap: usize,
    },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Index {
            folder,
            dry_run,
            max_tokens,
            overlap,
        } => {
            if overlap >= max_tokens {
                bail!("--overlap must be smaller than --max-tokens");
            }
            let config = ChunkConfig {
                max_tokens,
                overlap,
            };
            if !dry_run {
                bail!("only --dry-run is available so far");
            }
            dry_run_index(&folder, &cli.model_dir, config)
        }
    }
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
