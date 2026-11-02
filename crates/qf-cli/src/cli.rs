use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use qf_core::chunk::{ChunkConfig, ModelTokenizer};
use qf_core::embed::{DEFAULT_MODEL, Embedder, MODELS, Model};
use qf_core::format::{snippet, wrap};
use qf_core::fusion::RRF_KAPPA;
use qf_core::index::{build_index, check_index_dir, index_meta};
use qf_core::ingest::{Skipped, walk};
use qf_core::prepare::prepare;
use qf_core::search::Searcher;

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

    /// Embedding model, by Hugging Face id. An index is searched with the
    /// model it was built with
    #[arg(
        long,
        global = true,
        env = "QORFINDER_MODEL",
        default_value = DEFAULT_MODEL.id,
        value_parser = parse_model,
    )]
    model: &'static Model,

    /// Folder holding the embedding model files (config.json,
    /// tokenizer.json, model.safetensors). Defaults to models/<name>, where
    /// <name> is the last part of the --model id, next to the executable,
    /// or in the current folder when running from the repository
    #[arg(long, global = true, env = "QORFINDER_MODEL_DIR")]
    model_dir: Option<PathBuf>,

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
    /// Find the passages closest in meaning to a query
    Search {
        /// What to look for, in plain English
        query: String,

        /// Number of passages to return
        #[arg(short, long, default_value_t = 10)]
        k: usize,

        #[command(flatten)]
        chunking: ChunkArgs,
    },
    /// Score BM25, semantic search and their RRF fusion on the CUAD test
    /// set: writes TREC qrels and run files for scripts/score.py
    Eval {
        /// Folder holding queries.jsonl and spans.tsv
        #[arg(long, default_value = "data/testset")]
        testset: PathBuf,

        /// Folder of the sampled contracts the spans annotate. Never all
        /// 510 contracts: spans.tsv covers the sample only
        #[arg(long, default_value = "data/cuad/sample")]
        docs: PathBuf,

        /// Folder the TREC files (or timing.csv) are written to
        #[arg(long, default_value = "results/eval")]
        out: PathBuf,

        /// Rebuild the evaluation index even if it is up to date
        #[arg(long)]
        reindex: bool,

        /// Instead, time every query (one warm-up, then 5 runs) against
        /// the index in --index-dir, such as one of all 510 CUAD contracts
        #[arg(long)]
        timing: bool,

        /// RRF constant for the hybrid run
        #[arg(long, default_value_t = RRF_KAPPA)]
        kappa: u32,

        #[command(flatten)]
        chunking: ChunkArgs,
    },
    /// Open the search page in the browser, served on 127.0.0.1 only
    Serve {
        /// Port to listen on; 0 picks a free one
        #[arg(long, default_value_t = 0)]
        port: u16,

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

/// The `--model` id as a supported model.
fn parse_model(id: &str) -> Result<&'static Model, String> {
    Model::find(id).ok_or_else(|| {
        let ids: Vec<&str> = MODELS.iter().map(|m| m.id).collect();
        format!("unknown model; supported: {}", ids.join(", "))
    })
}

/// The chosen model and the folder holding its files.
pub(crate) struct ModelFiles {
    pub model: &'static Model,
    pub dir: PathBuf,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let model_files = &ModelFiles {
        model: cli.model,
        dir: cli
            .model_dir
            .unwrap_or_else(|| default_model_dir(cli.model)),
    };
    match cli.command {
        Command::Index {
            folder,
            dry_run,
            chunking,
        } => {
            let config = chunking.config()?;
            if dry_run {
                return dry_run_index(&folder, model_files, config);
            }
            let index_dir = cli.index_dir.unwrap_or_else(default_index_dir);
            index(&folder, &index_dir, model_files, config)
        }
        Command::Search { query, k, chunking } => {
            let index_dir = cli.index_dir.unwrap_or_else(default_index_dir);
            search(&query, k, &index_dir, model_files, chunking.config()?)
        }
        Command::Eval {
            testset,
            docs,
            out,
            reindex,
            timing,
            kappa,
            chunking,
        } => {
            let config = chunking.config()?;
            if timing {
                let index_dir = cli.index_dir.unwrap_or_else(default_index_dir);
                return crate::eval::timing(&testset, &out, &index_dir, model_files, config);
            }
            // Its own index, so evaluating never replaces the user's.
            let index_dir = cli
                .index_dir
                .unwrap_or_else(|| PathBuf::from(EVAL_INDEX_DIR));
            crate::eval::effectiveness(
                &testset,
                &docs,
                &out,
                &index_dir,
                model_files,
                config,
                reindex,
                kappa,
            )
        }
        Command::Serve { port, chunking } => {
            let index_dir = cli.index_dir.unwrap_or_else(default_index_dir);
            crate::serve::run(port, &index_dir, model_files, chunking.config()?)
        }
    }
}

/// The evaluation index, built from the CUAD sample.
const EVAL_INDEX_DIR: &str = "data/eval-index";

/// The model's folder next to the executable, so a release works from any
/// working folder; otherwise the repository copy, for `cargo run`. Either
/// way models/<name>, with <name> from the model id.
fn default_model_dir(model: &Model) -> PathBuf {
    let subdir = Path::new("models").join(model.dir_name());
    std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.join(&subdir)))
        .filter(|dir| dir.is_dir())
        .unwrap_or(subdir)
}

fn default_index_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("qorfinder")
        .join("index")
}

fn index(
    folder: &Path,
    index_dir: &Path,
    model_files: &ModelFiles,
    config: ChunkConfig,
) -> Result<()> {
    let mut out = std::io::stdout().lock();
    if !folder.is_dir() {
        bail!("{} is not a folder", folder.display());
    }
    // Before the model loads, so a wrong --index-dir fails at once.
    check_index_dir(folder, index_dir)?;
    let embedder = load_embedder(model_files)?;
    let tokenizer = load_tokenizer(model_files)?;
    let report = build_index(
        folder,
        index_dir,
        &embedder,
        &tokenizer,
        config,
        &Bars::default(),
    )?;
    for s in &report.skipped {
        writeln!(out, "skipped {}: {}", s.path.display(), s.reason)?;
    }
    let embed_secs = report.embed_time.as_secs_f64();
    writeln!(
        out,
        "indexed {} document(s), {} chunk(s), {} skipped, in {:.1} s \
         (prepare {:.1} s, embed {:.1} s, {:.1} chunks/s)",
        report.documents,
        report.chunks,
        report.skipped.len(),
        report.total_time.as_secs_f64(),
        report.prepare_time.as_secs_f64(),
        embed_secs,
        report.chunks as f64 / embed_secs.max(f64::EPSILON),
    )?;
    writeln!(out, "index saved to {}", index_dir.display())?;
    if let Some(line) = crate::memory::peak_line() {
        writeln!(out, "{line}")?;
    }
    if let Some(warning) = &report.cleanup_warning {
        eprintln!("warning: could not delete the previous index: {warning}");
    }
    Ok(())
}

/// Lines under a search result line up after its rank and score.
const RESULT_INDENT: &str = "           ";
/// Width of the wrapped passage under a result.
const RESULT_WIDTH: usize = 72;

fn search(
    query: &str,
    k: usize,
    index_dir: &Path,
    model_files: &ModelFiles,
    config: ChunkConfig,
) -> Result<()> {
    let mut out = std::io::stdout().lock();
    let embedder = load_embedder(model_files)?;
    let searcher = Searcher::open(index_dir, &embedder, &index_meta(&embedder, config))?;
    let result = searcher.search(query, k)?;
    let store = searcher.store();
    for (rank, hit) in result.hits.iter().enumerate() {
        let chunk = store
            .chunk(hit.chunk_id)?
            .with_context(|| format!("chunk {} missing from the index", hit.chunk_id))?;
        let doc = store
            .document(chunk.doc_id)?
            .with_context(|| format!("document {} missing from the index", chunk.doc_id))?;
        // File name first, where the eye looks; then where it is, and the
        // passage wrapped under it.
        let name = doc.path.file_name().unwrap_or(doc.path.as_os_str());
        let folder = doc.path.parent().unwrap_or(Path::new(""));
        let mut place = format!(
            "{} · characters {}–{}",
            folder.display(),
            chunk.char_start,
            chunk.char_end
        );
        if let Some(page) = chunk.page {
            place.push_str(&format!(" · page {page}"));
        }
        writeln!(out, "{:>2}. {:.3}  {}", rank + 1, hit.score, name.display())?;
        writeln!(out, "{RESULT_INDENT}{place}")?;
        for line in wrap(&snippet(&chunk.text, 280), RESULT_WIDTH) {
            writeln!(out, "{RESULT_INDENT}{line}")?;
        }
        writeln!(out)?;
    }
    let mut summary = format!(
        "{} result(s) in {:.1} ms (encode {:.1} ms, scan {:.1} ms)",
        result.hits.len(),
        (result.encode_time + result.scan_time).as_secs_f64() * 1e3,
        result.encode_time.as_secs_f64() * 1e3,
        result.scan_time.as_secs_f64() * 1e3,
    );
    if let Some(line) = crate::memory::peak_line() {
        summary.push_str(&format!(" · {line}"));
    }
    writeln!(out, "{summary}")?;
    Ok(())
}

/// Print every chunk the indexer would store, with its offsets into the
/// extracted text and its page, without embedding anything.
fn dry_run_index(folder: &Path, model_files: &ModelFiles, config: ChunkConfig) -> Result<()> {
    let mut out = std::io::stdout().lock();
    let tokenizer = load_tokenizer(model_files)?;
    let (files, mut skipped) = walk(folder);
    let (mut prepared, mut chunks) = (0, 0);
    for file in &files {
        match prepare(&file.path, file.format, &tokenizer, config) {
            Ok(passages) => {
                writeln!(out, "{} ({} chunks)", file.path.display(), passages.len())?;
                for (ord, p) in passages.iter().enumerate() {
                    let page = p.page.map_or(String::new(), |n| format!(" p.{n}"));
                    writeln!(
                        out,
                        "  #{ord} [{}..{}]{page} {}",
                        p.char_start,
                        p.char_end,
                        snippet(&p.text, 100)
                    )?;
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
        writeln!(out, "skipped {}: {}", s.path.display(), s.reason)?;
    }
    writeln!(
        out,
        "{prepared} file(s), {chunks} chunk(s), {} skipped",
        skipped.len()
    )?;
    Ok(())
}

pub(crate) fn load_embedder(model_files: &ModelFiles) -> Result<Embedder> {
    Embedder::load(model_files.model, &model_files.dir).with_context(|| {
        format!(
            "cannot load {} from {} (run scripts/fetch_model.py --model {} once to download it)",
            model_files.model.id,
            model_files.dir.display(),
            model_files.model.id,
        )
    })
}

pub(crate) fn load_tokenizer(model_files: &ModelFiles) -> Result<ModelTokenizer> {
    let path = model_files.dir.join("tokenizer.json");
    ModelTokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!(e))
        .with_context(|| {
            format!(
                "cannot load {} (run scripts/fetch_model.py --model {} once to download the model)",
                path.display(),
                model_files.model.id,
            )
        })
}
