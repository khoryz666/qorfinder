use std::fs;
use std::path::PathBuf;

use qf_core::chunk::{ChunkConfig, ModelTokenizer};
use qf_core::embed::Embedder;
use qf_core::index::{build_index, index_meta};
use qf_core::store::{Store, VECTORS_FILE, read_vectors};

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/all-MiniLM-L6-v2")
}

#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn indexes_a_folder_and_skips_bad_files() {
    let docs = tempfile::tempdir().unwrap();
    fs::write(
        docs.path().join("nda.txt"),
        "1. Confidentiality\nThe Recipient shall keep the information secret.\n\n\
         2. Term\nThis Agreement lasts two years.",
    )
    .unwrap();
    fs::write(
        docs.path().join("lease.txt"),
        "The tenant pays rent monthly.",
    )
    .unwrap();
    fs::write(docs.path().join("broken.pdf"), "not a pdf").unwrap();
    let index_dir = tempfile::tempdir().unwrap();

    let embedder = Embedder::load(&model_dir()).unwrap();
    let tokenizer = ModelTokenizer::from_file(&model_dir().join("tokenizer.json")).unwrap();
    let config = ChunkConfig {
        max_tokens: 12,
        overlap: 2,
    };
    let report = build_index(docs.path(), index_dir.path(), &embedder, &tokenizer, config).unwrap();

    assert_eq!(report.documents, 2);
    assert_eq!(report.skipped.len(), 1);
    assert!(report.skipped[0].path.ends_with("broken.pdf"));

    let store = Store::open(index_dir.path()).unwrap();
    store.check_meta(&index_meta(&embedder, config)).unwrap();
    assert_eq!(store.chunk_count().unwrap(), report.chunks);
    // Walk order: broken.pdf is skipped, lease.txt is document 0.
    assert_eq!(
        store
            .document(0)
            .unwrap()
            .unwrap()
            .path
            .file_name()
            .unwrap(),
        "lease.txt"
    );
    let first_nda_chunk = store.chunk(1).unwrap().unwrap();
    assert_eq!((first_nda_chunk.doc_id, first_nda_chunk.ord), (1, 0));
    assert!(first_nda_chunk.text.starts_with("1. Confidentiality"));

    let vectors = read_vectors(&index_dir.path().join(VECTORS_FILE), 384).unwrap();
    assert_eq!(vectors.len(), report.chunks * 384);
}
