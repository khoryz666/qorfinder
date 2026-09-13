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
use std::time::Duration;

use clap::Parser;
use qorfinder::cli::{Cli, execute};
use qorfinder::embedder::{Embedder, MODEL_DIMS};
use qorfinder::indexer::Indexer;
use qorfinder::query::run_query;
use qorfinder::store::Store;
use qorfinder::watcher::watch;

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

#[tokio::test]
#[ignore = "downloads/loads the real embedding model; runs the file watcher for several seconds"]
async fn watcher_reindexes_and_unindexes_files_while_releasing_the_lock() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/small_corpus");
    let watch_dir = tempfile::tempdir().unwrap();
    for name in ["doc1.txt", "doc2.md", "doc3.txt"] {
        std::fs::copy(source.join(name), watch_dir.path().join(name)).unwrap();
    }

    let index_dir = tempfile::tempdir().unwrap();
    let embedder = Arc::new(Embedder::try_new(None).expect("failed to load embedding model"));
    let dims = embedder.dims();

    // Initial scan, then release the lock exactly like the CLI does before
    // handing off to the watcher (see cli.rs's Index command).
    let store = Store::open(index_dir.path(), dims).unwrap();
    let indexer = Indexer::new(store, embedder.clone(), 512, 64);
    let stats = indexer.index_dir(watch_dir.path(), false).await.unwrap();
    assert_eq!(stats.indexed, 3);
    drop(indexer);

    let handle = tokio::spawn(watch(
        watch_dir.path().to_path_buf(),
        index_dir.path().to_path_buf(),
        embedder,
        512,
        64,
        Duration::from_millis(300),
    ));

    // Give the watcher a moment to install its filesystem hooks.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // The watcher should be idle now, holding no lock: a fresh open (what
    // `query`/`stats`/etc. do from another process) must succeed.
    let store =
        Store::open(index_dir.path(), dims).expect("open should succeed while watcher is idle");
    assert_eq!(store.count().unwrap(), 3);
    drop(store);

    // Modify a file; the watcher should pick it up and re-index it.
    std::fs::write(
        watch_dir.path().join("doc1.txt"),
        "a brand new sentence about watchers and debouncing",
    )
    .unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;

    let store = Store::open(index_dir.path(), dims).unwrap();
    let hits = store.search_lexical("debouncing", 5).unwrap();
    assert!(
        hits.iter().any(|h| h.text.contains("debouncing")),
        "expected the watcher to have re-indexed the modified file"
    );
    drop(store);

    // Delete a file; the watcher should un-index it.
    std::fs::remove_file(watch_dir.path().join("doc3.txt")).unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;

    let store = Store::open(index_dir.path(), dims).unwrap();
    let fingerprints = store.file_fingerprints().unwrap();
    assert!(
        !fingerprints.keys().any(|p| p.ends_with("doc3.txt")),
        "expected the watcher to have un-indexed the deleted file"
    );

    handle.abort();
}

#[tokio::test]
#[ignore = "downloads/loads the real embedding model on first run"]
async fn forget_command_removes_a_directory_from_the_index() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/small_corpus");
    let index_dir = tempfile::tempdir().unwrap();
    let index_dir_str = index_dir.path().to_str().unwrap();
    let corpus_str = corpus.to_str().unwrap();

    let index_cli = Cli::try_parse_from([
        "qorfinder",
        "--index-dir",
        index_dir_str,
        "index",
        corpus_str,
        "--once",
    ])
    .unwrap();
    execute(index_cli).await.unwrap();

    let store = Store::open(index_dir.path(), MODEL_DIMS).unwrap();
    assert_eq!(store.count().unwrap(), 3);
    drop(store);

    let forget_cli = Cli::try_parse_from([
        "qorfinder",
        "--index-dir",
        index_dir_str,
        "forget",
        corpus_str,
    ])
    .unwrap();
    execute(forget_cli).await.unwrap();

    let store = Store::open(index_dir.path(), MODEL_DIMS).unwrap();
    assert_eq!(
        store.count().unwrap(),
        0,
        "expected `forget` to remove every chunk under the given directory"
    );
}
