use std::path::PathBuf;

use qf_core::embed::{EmbedError, Embedder};

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/all-MiniLM-L6-v2")
}

#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn loads_minilm() {
    let embedder = Embedder::load(&model_dir()).unwrap();
    assert_eq!(embedder.dim(), 384);
}

#[test]
fn missing_model_dir_is_an_io_error() {
    let err = Embedder::load(std::path::Path::new("/nonexistent/qorfinder"))
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
    let embedder = Embedder::load(&model_dir()).unwrap();
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
    let embedder = Embedder::load(&model_dir()).unwrap();
    let short = "Governing law is England.";
    let alone = embedder.embed_query(short).unwrap();
    let batched = embedder
        .embed(&[
            short,
            "A much longer clause that forces the short one to be padded in the batch.",
        ])
        .unwrap();
    assert!(dot(&alone, &batched[0]) > 0.9999);
    assert!(embedder.embed(&[]).unwrap().is_empty());
}
