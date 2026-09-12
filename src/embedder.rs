use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Vector dimensionality of the chosen embedding model. Changing the model
/// means changing this constant and re-indexing everything.
pub const MODEL_DIMS: usize = 384;

const PASSAGE_PREFIX: &str = "passage: ";
const QUERY_PREFIX: &str = "query: ";

/// Prefix a document chunk for an e5-style model.
pub fn passage_text(chunk: &str) -> String {
    format!("{PASSAGE_PREFIX}{chunk}")
}

/// Prefix a user query for an e5-style model.
pub fn query_text(query: &str) -> String {
    format!("{QUERY_PREFIX}{query}")
}

pub struct Embedder {
    model: TextEmbedding,
    cache_dir: PathBuf,
}

impl Embedder {
    /// Load the embedding model. Downloads the ONNX model from HuggingFace on
    /// first use; afterwards the cache is fully offline.
    pub fn try_new(cache_dir: Option<PathBuf>) -> Result<Self> {
        // fastembed defaults to a CWD-relative cache, which would re-download
        // the model for every directory the CLI is invoked from. Pin it to a
        // stable per-user location instead.
        let cache_dir = cache_dir.unwrap_or_else(default_cache_dir);
        let options = InitOptions::new(EmbeddingModel::MultilingualE5Small)
            .with_cache_dir(cache_dir.clone())
            .with_show_download_progress(false);
        let model = TextEmbedding::try_new(options).context(
            "failed to initialize embedding model (first run needs network to download it)",
        )?;
        Ok(Self { model, cache_dir })
    }

    pub fn dims(&self) -> usize {
        MODEL_DIMS
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn embed_passages(&self, chunks: &[String]) -> Result<Vec<Vec<f32>>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let prefixed: Vec<String> = chunks.iter().map(|c| passage_text(c)).collect();
        let vectors = self
            .model
            .embed(prefixed, None)
            .context("embedding passages failed")?;
        anyhow::ensure!(
            vectors.len() == chunks.len(),
            "embedder returned {} vectors for {} chunks",
            vectors.len(),
            chunks.len()
        );
        Ok(vectors)
    }

    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let vectors = self
            .model
            .embed(vec![query_text(query)], None)
            .context("embedding query failed")?;
        vectors
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("embedder returned no vector for query"))
    }
}

/// Stable per-user model cache: `~/.cache/qorfinder/models` on all platforms.
fn default_cache_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("qorfinder").join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passage_text_gets_passage_prefix() {
        assert_eq!(passage_text("hello"), "passage: hello");
    }

    #[test]
    fn query_text_gets_query_prefix() {
        assert_eq!(query_text("hello"), "query: hello");
    }

    #[test]
    fn prefixes_differ() {
        assert_ne!(passage_text("x"), query_text("x"));
    }
}
