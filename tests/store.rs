use std::path::Path;

use qorfinder::store::Store;

fn open(dir: &Path) -> Store {
    Store::open(dir, 3).unwrap()
}

#[test]
fn stage_commit_search_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    let path = Path::new("/a.txt");
    store
        .stage_chunk(path, 0, "hello world", &[1.0, 0.0, 0.0], (1, 2))
        .unwrap();
    store.commit().unwrap();

    let hits = store.search(vec![1.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].file_path, "/a.txt");
    assert_eq!(hits[0].text, "hello world");

    let lexical_hits = store.search_lexical("hello", 5).unwrap();
    assert_eq!(lexical_hits.len(), 1);
}

#[test]
fn chunks_across_embedding_batches_accumulate_not_replace() {
    // Simulates indexer.rs staging one file's chunks interleaved with
    // another file's, as happens when an embed batch spans file
    // boundaries. A naive "replace all chunks for path" implementation
    // would drop earlier chunks when later ones for the same path
    // arrive in a later call.
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    let a = Path::new("/a.txt");
    let b = Path::new("/b.txt");
    store
        .stage_chunk(a, 0, "a chunk zero", &[1.0, 0.0, 0.0], (1, 1))
        .unwrap();
    store
        .stage_chunk(b, 0, "b chunk zero", &[0.0, 1.0, 0.0], (1, 1))
        .unwrap();
    store
        .stage_chunk(a, 1, "a chunk one", &[0.0, 0.0, 1.0], (1, 1))
        .unwrap();
    store.commit().unwrap();

    let fps = store.file_fingerprints().unwrap();
    assert_eq!(fps["/a.txt"].chunk_count, 2);
    assert_eq!(fps["/b.txt"].chunk_count, 1);
    let hits = store.search_lexical("chunk", 10).unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn delete_file_removes_staged_and_durable_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    let path = Path::new("/a.txt");
    store
        .stage_chunk(path, 0, "first version", &[1.0, 0.0, 0.0], (1, 1))
        .unwrap();
    store.commit().unwrap();

    store.delete_file(path).unwrap();
    store
        .stage_chunk(path, 0, "second version", &[0.0, 1.0, 0.0], (2, 2))
        .unwrap();
    store.commit().unwrap();

    let hits = store.search_lexical("version", 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].text, "second version");
    assert_eq!(store.file_fingerprints().unwrap()["/a.txt"].chunk_count, 1);
}

#[test]
fn forget_dir_removes_only_files_under_root() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store
        .stage_chunk(
            Path::new("/keep/root/a.txt"),
            0,
            "a",
            &[1.0, 0.0, 0.0],
            (1, 1),
        )
        .unwrap();
    store
        .stage_chunk(
            Path::new("/keep/root/b.txt"),
            0,
            "b",
            &[0.0, 1.0, 0.0],
            (1, 1),
        )
        .unwrap();
    store
        .stage_chunk(
            Path::new("/keep/elsewhere.txt"),
            0,
            "c",
            &[0.0, 0.0, 1.0],
            (1, 1),
        )
        .unwrap();
    store.commit().unwrap();

    let removed = store.forget_dir(Path::new("/keep/root")).unwrap();
    assert_eq!(removed, 2);
    let fps = store.file_fingerprints().unwrap();
    assert!(!fps.contains_key("/keep/root/a.txt"));
    assert!(!fps.contains_key("/keep/root/b.txt"));
    assert!(fps.contains_key("/keep/elsewhere.txt"));
    assert_eq!(store.count().unwrap(), 1);
}

#[test]
fn forget_dir_on_untracked_root_is_a_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    assert_eq!(store.forget_dir(Path::new("/nothing/here")).unwrap(), 0);
}

#[test]
fn delete_of_unknown_file_is_a_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store.delete_file(Path::new("/missing.txt")).unwrap();
}

#[test]
fn file_info_reflects_committed_fingerprint() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    let path = Path::new("/a.txt");
    assert!(store.file_info(path).unwrap().is_none());
    store
        .stage_chunk(path, 0, "text", &[1.0, 0.0, 0.0], (10, 20))
        .unwrap();
    store.commit().unwrap();
    let info = store.file_info(path).unwrap().unwrap();
    assert_eq!(info.mtime_secs, 10);
    assert_eq!(info.size_bytes, 20);
    assert_eq!(info.chunk_count, 1);
}

#[test]
fn count_reflects_vector_count() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store
        .stage_chunk(Path::new("/a.txt"), 0, "x", &[1.0, 0.0, 0.0], (1, 1))
        .unwrap();
    store
        .stage_chunk(Path::new("/a.txt"), 1, "y", &[0.0, 1.0, 0.0], (1, 1))
        .unwrap();
    store.commit().unwrap();
    assert_eq!(store.count().unwrap(), 2);
}

#[test]
fn reopening_existing_index_rejects_mismatched_dims() {
    let dir = tempfile::tempdir().unwrap();
    Store::open(dir.path(), 3).unwrap();
    match Store::open(dir.path(), 4) {
        Err(err) => assert!(err.to_string().contains("dims")),
        Ok(_) => panic!("expected a dims mismatch error"),
    }
}

#[test]
fn reopening_existing_index_with_same_dims_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    {
        let store = Store::open(dir.path(), 3).unwrap();
        store
            .stage_chunk(
                Path::new("/a.txt"),
                0,
                "persisted",
                &[1.0, 0.0, 0.0],
                (1, 1),
            )
            .unwrap();
        store.commit().unwrap();
    }
    let store = Store::open(dir.path(), 3).unwrap();
    assert_eq!(store.count().unwrap(), 1);
}
