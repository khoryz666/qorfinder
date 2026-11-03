//! The embedding models QorFinder can run, and what it must know about each
//! one beyond its `config.json` to reproduce Python sentence-transformers.
//! Every value comes from the model's sentence-transformers files at the
//! revision pinned in `scripts/fetch_model.py`.

/// The transformer architecture: `model_type` in `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Bert,
    MPNet,
}

impl Architecture {
    /// The `model_type` value in `config.json`.
    pub fn model_type(self) -> &'static str {
        match self {
            Architecture::Bert => "bert",
            Architecture::MPNet => "mpnet",
        }
    }
}

/// How a text's token vectors become one vector, as the model's
/// `1_Pooling/config.json` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Mean over the real (unpadded) tokens.
    Mean,
    /// The first ([CLS]) token's vector.
    Cls,
}

/// One supported embedding model.
#[derive(Debug, PartialEq, Eq)]
pub struct Model {
    /// Hugging Face id, as passed to `--model`.
    pub id: &'static str,
    pub architecture: Architecture,
    /// Length of every embedding vector: `hidden_size` in `config.json`.
    pub dim: usize,
    /// Input is truncated here, as `max_seq_length` in
    /// `sentence_bert_config.json` makes sentence-transformers do.
    pub max_seq_len: usize,
    pub pooling: Pooling,
    /// Put before every search query.
    pub query_prefix: &'static str,
    /// Put before every chunk.
    pub passage_prefix: &'static str,
}

impl Model {
    /// The model with this Hugging Face id, if QorFinder supports it.
    pub fn find(id: &str) -> Option<&'static Model> {
        MODELS.iter().find(|model| model.id == id)
    }

    /// Folder name of the model's files under `models/`: the last part of
    /// its id.
    pub fn dir_name(&self) -> &'static str {
        self.id.rsplit('/').next().unwrap_or(self.id)
    }
}

/// Every supported model. The first is the default, the model QorFinder
/// is built and evaluated with.
pub const MODELS: &[Model] = &[
    Model {
        id: "sentence-transformers/all-MiniLM-L6-v2",
        architecture: Architecture::Bert,
        dim: 384,
        max_seq_len: 256,
        pooling: Pooling::Mean,
        query_prefix: "",
        passage_prefix: "",
    },
    // MiniLM fine-tuned on the clauses of the CUAD contracts outside the test
    // sample (scripts/finetune_minilm.py). It is trained locally, not fetched
    // from Hugging Face, so scripts/fetch_model.py has no entry for it. Same
    // architecture, size and pooling as the base, so candle loads it as is.
    // The weights are the lowest-validation-loss epoch, pinned in the embed test.
    Model {
        id: "all-MiniLM-L6-v2-ft-cuad",
        architecture: Architecture::Bert,
        dim: 384,
        max_seq_len: 256,
        pooling: Pooling::Mean,
        query_prefix: "",
        passage_prefix: "",
    },
    Model {
        id: "sentence-transformers/all-mpnet-base-v2",
        architecture: Architecture::MPNet,
        dim: 768,
        max_seq_len: 384,
        pooling: Pooling::Mean,
        query_prefix: "",
        passage_prefix: "",
    },
    // The query instruction is the one the model card gives for retrieval.
    Model {
        id: "BAAI/bge-base-en-v1.5",
        architecture: Architecture::Bert,
        dim: 768,
        max_seq_len: 512,
        pooling: Pooling::Cls,
        query_prefix: "Represent this sentence for searching relevant passages: ",
        passage_prefix: "",
    },
    Model {
        id: "intfloat/e5-base-v2",
        architecture: Architecture::Bert,
        dim: 768,
        max_seq_len: 512,
        pooling: Pooling::Mean,
        query_prefix: "query: ",
        passage_prefix: "passage: ",
    },
];

/// The model used unless another is chosen.
pub const DEFAULT_MODEL: &Model = &MODELS[0];
