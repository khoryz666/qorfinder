use std::fs;
use std::path::PathBuf;

use qf_core::chunk::{ChunkConfig, ModelTokenizer};
use qf_core::embed::Embedder;
use std::sync::Mutex;

use qf_core::IndexMeta;
use qf_core::index::{Progress, Stage, build_index, index_meta};
use qf_core::search::Searcher;
use qf_core::store::{Store, VECTORS_FILE, read_vectors};

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/all-MiniLM-L6-v2")
}

/// Records progress events, summing `advance` calls per stage.
#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<String>>,
    done: Mutex<usize>,
}

impl Progress for Recorder {
    fn start(&self, stage: Stage, total: usize) {
        *self.done.lock().unwrap() = 0;
        self.events
            .lock()
            .unwrap()
            .push(format!("start {stage:?} {total}"));
    }
    fn advance(&self, done: usize) {
        *self.done.lock().unwrap() += done;
    }
    fn finish(&self, stage: Stage) {
        let done = *self.done.lock().unwrap();
        self.events
            .lock()
            .unwrap()
            .push(format!("done {stage:?} {done}"));
    }
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
    let progress = Recorder::default();
    let report = build_index(
        docs.path(),
        index_dir.path(),
        &embedder,
        &tokenizer,
        config,
        &progress,
    )
    .unwrap();

    assert_eq!(report.documents, 2);
    assert!(!qf_core::store::staging_dir(index_dir.path()).exists());
    assert_eq!(report.skipped.len(), 1);
    assert!(report.skipped[0].path.ends_with("broken.pdf"));

    let store = Store::open(index_dir.path()).unwrap();
    store.check_meta(&index_meta(&embedder, config)).unwrap();
    assert_eq!(
        store.source_folder().unwrap(),
        Some(std::path::absolute(docs.path()).unwrap())
    );
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

    assert_eq!(
        *progress.events.lock().unwrap(),
        [
            "start Preparing 3".to_string(),
            "done Preparing 3".to_string(),
            format!("start Embedding {}", report.chunks),
            format!("done Embedding {}", report.chunks),
        ]
    );

    let vectors = read_vectors(&index_dir.path().join(VECTORS_FILE), 384).unwrap();
    assert_eq!(vectors.len(), report.chunks * 384);
}

/// T4: an index survives a restart and is searched without re-indexing.
#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn saved_index_is_searched_after_reopening() {
    let docs = tempfile::tempdir().unwrap();
    fs::write(
        docs.path().join("employment.txt"),
        "1. Non-Compete\nFor two years after termination the Employee shall not work for a competitor.",
    )
    .unwrap();
    fs::write(
        docs.path().join("supply.txt"),
        "1. Governing Law\nThis Agreement is governed by the laws of Delaware.",
    )
    .unwrap();
    let index_dir = tempfile::tempdir().unwrap();
    let config = ChunkConfig::default();
    {
        let embedder = Embedder::load(&model_dir()).unwrap();
        let tokenizer = ModelTokenizer::from_file(&model_dir().join("tokenizer.json")).unwrap();
        build_index(
            docs.path(),
            index_dir.path(),
            &embedder,
            &tokenizer,
            config,
            &(),
        )
        .unwrap();
    }

    // A fresh "process": nothing but the files on disk.
    let embedder = Embedder::load(&model_dir()).unwrap();
    let searcher =
        Searcher::open(index_dir.path(), &embedder, &index_meta(&embedder, config)).unwrap();
    let result = searcher
        .search("can I join a rival company after I quit?", 2)
        .unwrap();
    assert_eq!(result.hits.len(), 2);
    let best = searcher
        .store()
        .chunk(result.hits[0].chunk_id)
        .unwrap()
        .unwrap();
    let doc = searcher.store().document(best.doc_id).unwrap().unwrap();
    assert!(doc.path.ends_with("employment.txt"), "best hit was {doc:?}");
    assert!(result.hits[0].score > result.hits[1].score);

    let other = IndexMeta {
        overlap: 10,
        ..index_meta(&embedder, config)
    };
    assert!(Searcher::open(index_dir.path(), &embedder, &other).is_err());
}

#[test]
fn documents_may_not_be_inside_the_index_folder() {
    use qf_core::index::{IndexError, check_index_dir};

    let root = tempfile::tempdir().unwrap();
    let docs = root.path().join("contracts");
    fs::create_dir(&docs).unwrap();
    let refused = |index: PathBuf| {
        matches!(
            check_index_dir(&docs, &index),
            Err(IndexError::FolderInsideIndex { .. })
        )
    };
    assert!(refused(docs.clone()));
    assert!(refused(docs.join(".")));
    assert!(refused(root.path().to_path_buf()));
    assert!(check_index_dir(&docs, &root.path().join("index")).is_ok());
    // A sibling whose name merely starts the same is not a parent.
    assert!(check_index_dir(&docs, &root.path().join("contracts-index")).is_ok());
    // The index may sit inside the documents, as the default one does when
    // a whole user folder is indexed.
    assert!(check_index_dir(&docs, &docs.join("index")).is_ok());
    assert!(check_index_dir(&docs, &docs.join("sub/../index")).is_ok());
}

/// With the index inside the indexed folder, rebuilding twice indexes the
/// documents only: the index, its staging folder and old copies are left
/// out of the walk, not read as documents or listed as skipped.
#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn index_inside_the_indexed_folder_is_left_out() {
    let docs = tempfile::tempdir().unwrap();
    fs::write(
        docs.path().join("lease.txt"),
        "The tenant pays rent monthly.",
    )
    .unwrap();
    let index_dir = docs.path().join(".qorfinder/index");
    let embedder = Embedder::load(&model_dir()).unwrap();
    let tokenizer = ModelTokenizer::from_file(&model_dir().join("tokenizer.json")).unwrap();

    for _ in 0..2 {
        let report = build_index(
            docs.path(),
            &index_dir,
            &embedder,
            &tokenizer,
            ChunkConfig::default(),
            &(),
        )
        .unwrap();
        assert_eq!(report.documents, 1);
        assert!(report.skipped.is_empty(), "{:?}", report.skipped);
    }
    // A user's text file beside the index is still a document.
    fs::write(docs.path().join(".qorfinder/notes.txt"), "Rent is due.").unwrap();
    let report = build_index(
        docs.path(),
        &index_dir,
        &embedder,
        &tokenizer,
        ChunkConfig::default(),
        &(),
    )
    .unwrap();
    assert_eq!(report.documents, 2);
}

#[test]
fn index_folder_of_other_files_is_refused_before_any_work() {
    use qf_core::index::{IndexError, check_index_dir};
    use qf_core::store::StoreError;

    let root = tempfile::tempdir().unwrap();
    let docs = root.path().join("contracts");
    fs::create_dir(&docs).unwrap();
    let work = root.path().join("work");
    fs::create_dir(&work).unwrap();
    fs::write(work.join("memo.txt"), "mine").unwrap();
    assert!(matches!(
        check_index_dir(&docs, &work),
        Err(IndexError::Store(StoreError::NotAnIndex(_)))
    ));
    let partial = root.path().join("index.partial");
    fs::create_dir(&partial).unwrap();
    fs::write(partial.join("memo.txt"), "mine").unwrap();
    assert!(check_index_dir(&docs, &root.path().join("index")).is_err());
    assert_eq!(fs::read_to_string(work.join("memo.txt")).unwrap(), "mine");
}
