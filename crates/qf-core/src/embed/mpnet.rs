//! The MPNet encoder (Song et al., 2020), which candle-transformers does
//! not provide. A port of Hugging Face `modeling_mpnet.py` for inference:
//! BERT-like layers plus a learned relative-position bias shared by all
//! layers, and position ids that start after the padding id.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear};

/// The fields of an MPNet `config.json` the encoder needs.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub relative_attention_num_buckets: usize,
}

/// Relative distances beyond this share the last bucket.
const MAX_DISTANCE: f64 = 128.0;

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    attention_norm: LayerNorm,
    intermediate: Linear,
    output: Linear,
    output_norm: LayerNorm,
    heads: usize,
    head_dim: usize,
}

impl Layer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let h = config.hidden_size;
        let eps = config.layer_norm_eps;
        let attn = vb.pp("attention.attn");
        Ok(Self {
            q: linear(h, h, attn.pp("q"))?,
            k: linear(h, h, attn.pp("k"))?,
            v: linear(h, h, attn.pp("v"))?,
            o: linear(h, h, attn.pp("o"))?,
            attention_norm: layer_norm(h, eps, vb.pp("attention.LayerNorm"))?,
            intermediate: linear(h, config.intermediate_size, vb.pp("intermediate.dense"))?,
            output: linear(config.intermediate_size, h, vb.pp("output.dense"))?,
            output_norm: layer_norm(h, eps, vb.pp("output.LayerNorm"))?,
            heads: config.num_attention_heads,
            head_dim: h / config.num_attention_heads,
        })
    }

    /// `bias` is the position bias plus the padding mask, broadcastable to
    /// [batch, heads, tokens, tokens].
    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let (batch, tokens, hidden) = xs.dims3()?;
        let split = |t: Tensor| {
            t.reshape((batch, tokens, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.q.forward(xs)?)?;
        let k = split(self.k.forward(xs)?)?;
        let v = split(self.v.forward(xs)?)?;
        let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?.broadcast_add(bias)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let context = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, tokens, hidden))?;
        let attended = self
            .attention_norm
            .forward(&(self.o.forward(&context)? + xs)?)?;
        let inner = self.intermediate.forward(&attended)?.gelu_erf()?;
        self.output_norm
            .forward(&(self.output.forward(&inner)? + attended)?)
    }
}

pub struct MPNetModel {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    embedding_norm: LayerNorm,
    layers: Vec<Layer>,
    relative_attention_bias: Embedding,
    buckets: usize,
    pad_token_id: u32,
}

impl MPNetModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let h = config.hidden_size;
        let emb = vb.pp("embeddings");
        let enc = vb.pp("encoder");
        Ok(Self {
            word_embeddings: embedding(config.vocab_size, h, emb.pp("word_embeddings"))?,
            position_embeddings: embedding(
                config.max_position_embeddings,
                h,
                emb.pp("position_embeddings"),
            )?,
            embedding_norm: layer_norm(h, config.layer_norm_eps, emb.pp("LayerNorm"))?,
            layers: (0..config.num_hidden_layers)
                .map(|i| Layer::load(enc.pp(format!("layer.{i}")), config))
                .collect::<Result<_>>()?,
            relative_attention_bias: embedding(
                config.relative_attention_num_buckets,
                config.num_attention_heads,
                enc.pp("relative_attention_bias"),
            )?,
            buckets: config.relative_attention_num_buckets,
            pad_token_id: config.pad_token_id,
        })
    }

    /// Token vectors, [batch, tokens, hidden], for `ids` and `mask` of
    /// shape [batch, tokens] (mask 1 for real tokens, 0 for padding).
    pub fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let device = ids.device();
        let tokens = ids.dim(1)?;

        // Real tokens are numbered from pad_token_id + 1; padding gets
        // pad_token_id.
        let real = ids.ne(self.pad_token_id)?.to_dtype(DType::F32)?;
        let positions = (real.cumsum(1)? * &real)?
            .affine(1.0, self.pad_token_id as f64)?
            .to_dtype(DType::U32)?;
        let embedded =
            (self.word_embeddings.forward(ids)? + self.position_embeddings.forward(&positions)?)?;
        let mut xs = self.embedding_norm.forward(&embedded)?;

        // [1, heads, tokens, tokens] position bias plus [batch, 1, 1, tokens]
        // mask, so padding keys get no attention.
        let position_bias = self
            .relative_attention_bias
            .forward(&self.buckets_for(tokens, device)?)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?;
        let padding = ((mask.to_dtype(DType::F32)? - 1.0)? * f32::MAX as f64)?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let bias = position_bias.broadcast_add(&padding)?;

        for layer in &self.layers {
            xs = layer.forward(&xs, &bias)?;
        }
        Ok(xs)
    }

    /// [tokens, tokens] bucket of each key's offset from each query, as
    /// `relative_position_bucket` in the reference implementation.
    fn buckets_for(&self, tokens: usize, device: &Device) -> Result<Tensor> {
        let half = self.buckets / 2;
        let max_exact = half / 2;
        let mut buckets = Vec::with_capacity(tokens * tokens);
        for query in 0..tokens as i64 {
            for key in 0..tokens as i64 {
                let n = query - key;
                let base = if n < 0 { half } else { 0 };
                let n = n.unsigned_abs() as usize;
                let offset = if n < max_exact {
                    n
                } else {
                    let large = max_exact
                        + ((n as f32 / max_exact as f32).ln()
                            / (MAX_DISTANCE as f32 / max_exact as f32).ln()
                            * (half - max_exact) as f32) as usize;
                    large.min(half - 1)
                };
                buckets.push((base + offset) as u32);
            }
        }
        Tensor::from_vec(buckets, (tokens, tokens), device)
    }
}
