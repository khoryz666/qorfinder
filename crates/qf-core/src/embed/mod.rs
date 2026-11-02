//! Sentence embeddings with one of the models in [`MODELS`], run on the CPU
//! with candle.

mod models;
mod mpnet;

pub use models::{Architecture, DEFAULT_MODEL, MODELS, Model, Pooling};

use std::path::Path;

use candle_core::{D, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{self, BertModel, DTYPE};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::chunk::TokenizeError;

/// Texts embedded in one forward pass. Larger batches are a little faster
/// but hold more activations in memory at once.
pub const BATCH_SIZE: usize = 32;

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
    #[error("the files are not {model}: config.json has {found}, expected {expected}")]
    WrongModel {
        model: &'static str,
        found: String,
        expected: String,
    },
}

/// The transformer, chosen by `model_type` in `config.json`.
enum Encoder {
    Bert(BertModel),
    MPNet(mpnet::MPNetModel),
}

/// The fields of `config.json` checked against the [`Model`].
#[derive(serde::Deserialize)]
struct ModelType {
    model_type: String,
    hidden_size: usize,
    #[serde(default)]
    pad_token_id: u32,
}

/// A loaded embedding model and its tokenizer.
pub struct Embedder {
    model: &'static Model,
    encoder: Encoder,
    tokenizer: Tokenizer,
    device: Device,
    weights_sha256: String,
}

impl Embedder {
    /// Load `model`'s `config.json`, `tokenizer.json` and
    /// `model.safetensors` from `model_dir`. Nothing is downloaded (NFR1);
    /// the files come from `scripts/fetch_model.py`. Fails if the files are
    /// of another architecture or size than `model`.
    pub fn load(model: &'static Model, model_dir: &Path) -> Result<Self, EmbedError> {
        let read = |name: &str| {
            let path = model_dir.join(name);
            std::fs::read(&path).map_err(|source| EmbedError::Io {
                path: path.display().to_string(),
                source,
            })
        };
        let config = read("config.json")?;
        let model_type: ModelType = serde_json::from_slice(&config)?;
        let wrong = |found: String, expected: String| EmbedError::WrongModel {
            model: model.id,
            found,
            expected,
        };
        if model_type.model_type != model.architecture.model_type() {
            return Err(wrong(
                format!("model_type {:?}", model_type.model_type),
                format!("{:?}", model.architecture.model_type()),
            ));
        }
        if model_type.hidden_size != model.dim {
            return Err(wrong(
                format!("hidden_size {}", model_type.hidden_size),
                model.dim.to_string(),
            ));
        }

        let mut tokenizer =
            Tokenizer::from_bytes(read("tokenizer.json")?).map_err(EmbedError::Tokenizer)?;
        // Pad with the model's own padding token (MPNet's is not id 0).
        let pad_id = model_type.pad_token_id;
        let pad_token = tokenizer.id_to_token(pad_id).unwrap_or_default();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: model.max_seq_len,
                ..Default::default()
            }))
            .map_err(EmbedError::Tokenizer)?
            .with_padding(Some(PaddingParams {
                pad_id,
                pad_token,
                ..Default::default()
            }));

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
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DTYPE, &device)? };
        let encoder = match model.architecture {
            Architecture::MPNet => {
                let config: mpnet::Config = serde_json::from_slice(&config)?;
                Encoder::MPNet(mpnet::MPNetModel::load(vb, &config)?)
            }
            Architecture::Bert => {
                let config: bert::Config = serde_json::from_slice(&config)?;
                Encoder::Bert(BertModel::load(vb, &config)?)
            }
        };
        let weights_sha256 = crate::ingest::hash_file(&weights)
            .map_err(|source| EmbedError::Io {
                path: weights.display().to_string(),
                source,
            })?
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        Ok(Self {
            model,
            encoder,
            tokenizer,
            device,
            weights_sha256,
        })
    }

    /// The model this embedder runs.
    pub fn model(&self) -> &'static Model {
        self.model
    }

    /// Length of every embedding vector.
    pub fn dim(&self) -> usize {
        self.model.dim
    }

    /// SHA-256 of `model.safetensors` in hex: identifies the exact weights
    /// an index was built with.
    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    /// Embed one batch of chunks (up to [`BATCH_SIZE`] at a time is best),
    /// each with the model's passage prefix.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.embed_with_prefix(self.model.passage_prefix, texts)
    }

    /// Embed a search query, with the model's query prefix.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self
            .embed_with_prefix(self.model.query_prefix, &[query])?
            .remove(0))
    }

    /// Each text is prefixed, tokenised and truncated to the model's
    /// `max_seq_len` tokens, the batch is padded to its longest text and
    /// run through the encoder, and every text's token vectors are pooled
    /// as the model says (a mean counts only the real, unpadded tokens) and
    /// scaled to unit length, as sentence-transformers does. Unit length
    /// makes the dot product equal cosine similarity.
    fn embed_with_prefix(&self, prefix: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
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
        let hidden = match &self.encoder {
            Encoder::Bert(model) => model.forward(&ids, &type_ids, Some(&mask))?,
            Encoder::MPNet(model) => model.forward(&ids, &mask)?,
        };
        let pooled = match self.model.pooling {
            Pooling::Cls => hidden.narrow(1, 0, 1)?.squeeze(1)?,
            Pooling::Mean => {
                let mask = mask.to_dtype(hidden.dtype())?.unsqueeze(D::Minus1)?;
                let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
                summed.broadcast_div(&mask.sum(1)?)?
            }
        };
        let norms = pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        Ok(pooled.broadcast_div(&norms)?.to_vec2()?)
    }
}
