//! Sentence embeddings with all-MiniLM-L6-v2, run on the CPU with candle.

use std::path::Path;

use candle_core::{D, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::chunk::TokenizeError;

/// Hugging Face id of the model QorFinder is built and evaluated with.
pub const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Texts embedded in one forward pass. Larger batches are a little faster
/// but hold more activations in memory at once.
pub const BATCH_SIZE: usize = 32;

/// sentence-transformers truncates this model's input at 256 tokens
/// (`max_seq_length` in its `sentence_bert_config.json`); doing the same
/// keeps the Rust embeddings identical to the Python ones.
pub const MAX_SEQ_LEN: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid config.json: {0}")]
    Config(#[from] serde_json::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(TokenizeError),
    #[error("model: {0}")]
    Model(#[from] candle_core::Error),
}

/// A loaded embedding model and its tokenizer.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
}

impl Embedder {
    /// Load `config.json`, `tokenizer.json` and `model.safetensors` from
    /// `model_dir`. Nothing is downloaded (NFR1); the files come from
    /// `scripts/fetch_model.py`.
    pub fn load(model_dir: &Path) -> Result<Self, EmbedError> {
        let read = |name: &str| {
            let path = model_dir.join(name);
            std::fs::read(&path).map_err(|source| EmbedError::Io {
                path: path.display().to_string(),
                source,
            })
        };
        let config: Config = serde_json::from_slice(&read("config.json")?)?;

        let mut tokenizer =
            Tokenizer::from_bytes(read("tokenizer.json")?).map_err(EmbedError::Tokenizer)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(EmbedError::Tokenizer)?
            .with_padding(Some(PaddingParams::default()));

        let device = Device::Cpu;
        let weights = model_dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(EmbedError::Io {
                path: weights.display().to_string(),
                source: std::io::ErrorKind::NotFound.into(),
            });
        }
        // SAFETY: the weights file is memory-mapped read-only; it must not be
        // modified while the model is loaded, which nothing in QorFinder does.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            dim: config.hidden_size,
        })
    }

    /// Length of every embedding vector (384 for all-MiniLM-L6-v2).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embed one batch of texts (up to [`BATCH_SIZE`] at a time is best).
    /// Each text is tokenised and truncated to [`MAX_SEQ_LEN`] tokens, the
    /// batch is padded to its longest text, run through BERT, and every
    /// text's token vectors are mean-pooled over its real (unpadded) tokens
    /// and scaled to unit length, as sentence-transformers does for this
    /// model. Unit length makes the dot product equal cosine similarity.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(EmbedError::Tokenizer)?;
        let stack = |rows: Vec<&[u32]>| -> Result<Tensor, EmbedError> {
            let rows = rows
                .into_iter()
                .map(|row| Tensor::new(row, &self.device))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Tensor::stack(&rows, 0)?)
        };
        let ids = stack(encodings.iter().map(|e| e.get_ids()).collect())?;
        let type_ids = stack(encodings.iter().map(|e| e.get_type_ids()).collect())?;
        let mask = stack(encodings.iter().map(|e| e.get_attention_mask()).collect())?;

        // [batch, tokens, dim]
        let hidden = self.model.forward(&ids, &type_ids, Some(&mask))?;
        let mask = mask.to_dtype(hidden.dtype())?.unsqueeze(D::Minus1)?;
        let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
        let counts = mask.sum(1)?;
        let mean = summed.broadcast_div(&counts)?;
        let norms = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
        Ok(mean.broadcast_div(&norms)?.to_vec2()?)
    }

    /// Embed a search query with exactly the same code path as a chunk.
    /// all-MiniLM-L6-v2 uses no query prefix.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.embed(&[query])?.remove(0))
    }
}
