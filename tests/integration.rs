//! End-to-end coverage of the real pipeline: parse -> chunk -> embed ->
//! index -> query, against the repo's small fixture corpus.
//!
//! Unlike `cargo test --lib`, this needs a real `Embedder`, which downloads
//! the ONNX model from HuggingFace on first use (afterwards it's cached and
//! fully offline, same as the CLI itself). That network dependency is why
//! this lives in `tests/` and is `#[ignore]`d rather than in `src/`'s fast,
//! fully-offline unit tests: run it explicitly with
//!
//! ```bash
//! cargo test --test integration -- --ignored
//! ```

use std::path::Path;
use std::sync::Arc;

use qorfinder::embedder::Embedder;
use qorfinder::indexer::Indexer;
use qorfinder::query::run_query;
use qorfinder::store::Store;

#[tokio::test]
#[ignore = "downloads/loads the real embedding model on first run"]
async fn indexes_and_queries_the_fixture_corpus() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/small_corpus");
    let index_dir = tempfile::tempdir().unwrap();

    let embedder = Arc::new(Embedder::try_new(None).expect("failed to load embedding model"));
    let store = Store::open(index_dir.path(), embedder.dims()).unwrap();
    let indexer = Indexer::new(store, embedder, 512, 64);

    let stats = indexer.index_dir(&corpus, false).await.unwrap();
    assert_eq!(
        stats.indexed, 3,
        "expected all 3 fixture files to be indexed"
    );
    assert_eq!(stats.failed, 0);

    let hits = run_query(
        indexer.store(),
        indexer.embedder(),
        "database for vectors",
        5,
    )
    .unwrap();
    assert!(
        hits.iter().any(|h| h.file_path.ends_with("doc2.md")),
        "expected doc2.md (Vector Databases) among hits, got {:?}",
        hits.iter().map(|h| &h.file_path).collect::<Vec<_>>()
    );

    // Re-running unchanged should be a pure no-op on the fingerprint side.
    let rescan = indexer.index_dir(&corpus, false).await.unwrap();
    assert_eq!(rescan.indexed, 0);
    assert_eq!(rescan.unchanged, 3);
}
