use std::path::PathBuf;

use qf_core::embed::{DEFAULT_MODEL, EmbedError, Embedder, MODELS, Model, Pooling};

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(DEFAULT_MODEL.dir_name())
}

fn load() -> Embedder {
    Embedder::load(DEFAULT_MODEL, &model_dir()).unwrap()
}

#[test]
fn models_are_found_by_id() {
    let e5 = Model::find("intfloat/e5-base-v2").unwrap();
    assert_eq!(e5.dir_name(), "e5-base-v2");
    assert_eq!(e5.query_prefix, "query: ");
    assert_eq!(e5.passage_prefix, "passage: ");
    assert_eq!(
        Model::find("BAAI/bge-base-en-v1.5").unwrap().pooling,
        Pooling::Cls
    );
    assert!(Model::find("all-MiniLM-L6-v2").is_none());
    assert_eq!(DEFAULT_MODEL.id, "sentence-transformers/all-MiniLM-L6-v2");
    // Ids, and so model folders, are unique.
    for (i, a) in MODELS.iter().enumerate() {
        for b in &MODELS[i + 1..] {
            assert_ne!(a.dir_name(), b.dir_name());
        }
    }
}

#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn files_of_another_model_are_refused() {
    // MiniLM's files are BERT with 384 dimensions, not MPNet with 768.
    let mpnet = Model::find("sentence-transformers/all-mpnet-base-v2").unwrap();
    let err = Embedder::load(mpnet, &model_dir()).err().unwrap();
    assert!(matches!(err, EmbedError::WrongModel { .. }), "got {err}");
}

#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn loads_minilm() {
    let embedder = load();
    assert_eq!(embedder.dim(), 384);
    // The weights pinned in scripts/fetch_model.py.
    assert_eq!(
        embedder.weights_sha256(),
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
    );
}

#[test]
fn missing_model_dir_is_an_io_error() {
    let err = Embedder::load(
        DEFAULT_MODEL,
        std::path::Path::new("/nonexistent/qorfinder"),
    )
    .err()
    .unwrap();
    assert!(matches!(err, EmbedError::Io { .. }), "got {err}");
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn embeddings_are_unit_length_and_meaningful() {
    let embedder = load();
    let texts = [
        "The employee may not work for a competitor for two years.",
        "After leaving, staff are barred from joining rival firms for 24 months.",
        "The invoice is payable within thirty days of receipt.",
    ];
    let vectors = embedder.embed(&texts).unwrap();
    assert_eq!(vectors.len(), 3);
    for v in &vectors {
        assert_eq!(v.len(), 384);
        assert!((dot(v, v) - 1.0).abs() < 1e-4);
    }
    // The paraphrase is closer than the unrelated clause.
    assert!(dot(&vectors[0], &vectors[1]) > dot(&vectors[0], &vectors[2]) + 0.2);
}

#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn padding_does_not_change_an_embedding() {
    let embedder = load();
    let short = "Governing law is England.";
    let alone = embedder.embed(&[short]).unwrap().remove(0);
    let batched = embedder
        .embed(&[
            short,
            "A much longer clause that forces the short one to be padded in the batch.",
        ])
        .unwrap();
    assert!(dot(&alone, &batched[0]) > 0.9999);
    assert!(embedder.embed(&[]).unwrap().is_empty());
}

/// Correctness check for the report: candle must reproduce Python
/// sentence-transformers. The reference vectors come from
/// scripts/reference_embeddings.py.
#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn embeddings_match_sentence_transformers() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minilm_reference.json");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let sentences: Vec<&str> = reference["sentences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    let expected: Vec<Vec<f32>> = serde_json::from_value(reference["embeddings"].clone()).unwrap();

    let actual = load().embed(&sentences).unwrap();
    for ((sentence, a), e) in sentences.iter().zip(&actual).zip(&expected) {
        let cosine = dot(a, e) / (dot(a, a).sqrt() * dot(e, e).sqrt());
        println!("cosine {cosine:.6}  {sentence}");
        assert!(cosine > 0.99, "cosine {cosine} for {sentence:?}");
    }
}
