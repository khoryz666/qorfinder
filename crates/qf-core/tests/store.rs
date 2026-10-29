use std::path::PathBuf;

use qf_core::store::{Store, StoreError, VectorWriter, install, read_vectors, staging_dir};
use qf_core::{Chunk, Document, Format, IndexMeta};

fn document(id: u32) -> Document {
    Document {
        id,
        path: PathBuf::from(format!("/contracts/{id}.pdf")),
        format: Format::Pdf,
        hash: [id as u8; 32],
        mtime: 1_700_000_000,
    }
}

fn chunk(id: u32, doc_id: u32, ord: u32) -> Chunk {
    Chunk {
        id,
        doc_id,
        ord,
        text: format!("chunk {id}"),
        char_start: 100 * ord as usize,
        char_end: 100 * ord as usize + 90,
        page: ord.is_multiple_of(2).then_some(ord + 1),
    }
}

#[test]
fn documents_and_chunks_survive_reopening() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut store = Store::create(dir.path()).unwrap();
        store.insert_documents(&[document(0), document(1)]).unwrap();
        store
            .insert_chunks(&[chunk(0, 0, 0), chunk(1, 0, 1), chunk(2, 1, 0)])
            .unwrap();
    }
    let store = Store::open(dir.path()).unwrap();
    assert_eq!(store.chunk_count().unwrap(), 3);
    assert_eq!(store.document(1).unwrap(), Some(document(1)));
    assert_eq!(store.chunk(1).unwrap(), Some(chunk(1, 0, 1)));
    assert_eq!(store.chunk(2).unwrap(), Some(chunk(2, 1, 0)));
    assert_eq!(store.chunk(3).unwrap(), None);
    assert_eq!(store.document(7).unwrap(), None);
}

#[test]
fn create_replaces_an_existing_index() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::create(dir.path()).unwrap();
    store.insert_documents(&[document(0)]).unwrap();
    store.insert_chunks(&[chunk(0, 0, 0)]).unwrap();
    drop(store);

    let store = Store::create(dir.path()).unwrap();
    assert_eq!(store.chunk_count().unwrap(), 0);
}

#[test]
fn opening_a_missing_index_fails() {
    let dir = tempfile::tempdir().unwrap();
    let err = Store::open(dir.path()).err().unwrap();
    assert!(
        err.to_string().contains("run `qorfinder index` first"),
        "{err}"
    );
}

#[test]
fn vectors_round_trip_as_little_endian_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors.f32");
    let mut writer = VectorWriter::create(&path, 3).unwrap();
    writer.push(&[1.0, -2.5, 0.0]).unwrap();
    writer.push(&[f32::MIN_POSITIVE, 3.25, -0.0]).unwrap();
    assert!(writer.push(&[1.0]).is_err());
    assert_eq!(writer.finish().unwrap(), 2);

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len(), 2 * 3 * 4);
    assert_eq!(&bytes[4..8], &(-2.5f32).to_le_bytes());
    assert_eq!(
        read_vectors(&path, 3).unwrap(),
        [1.0, -2.5, 0.0, f32::MIN_POSITIVE, 3.25, -0.0]
    );
}

#[test]
fn truncated_vector_file_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors.f32");
    std::fs::write(&path, [0u8; 13]).unwrap();
    assert!(read_vectors(&path, 3).is_err());
}

fn meta() -> IndexMeta {
    IndexMeta {
        model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
        model_rev: "53aa5117".into(),
        dim: 384,
        max_tokens: 200,
        overlap: 40,
    }
}

#[test]
fn meta_round_trips_and_matching_settings_pass() {
    let dir = tempfile::tempdir().unwrap();
    Store::create(dir.path())
        .unwrap()
        .write_meta(&meta())
        .unwrap();
    let store = Store::open(dir.path()).unwrap();
    assert_eq!(store.read_meta().unwrap(), meta());
    assert_eq!(store.check_meta(&meta()).unwrap(), meta());
}

#[test]
fn index_built_with_other_settings_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    Store::create(dir.path())
        .unwrap()
        .write_meta(&meta())
        .unwrap();
    let store = Store::open(dir.path()).unwrap();

    let other_chunking = IndexMeta {
        max_tokens: 120,
        ..meta()
    };
    let err = store.check_meta(&other_chunking).unwrap_err();
    assert!(matches!(
        err,
        StoreError::Mismatch {
            field: "max_tokens",
            ..
        }
    ));
    assert!(
        err.to_string()
            .contains("max_tokens 200, but this run uses 120"),
        "{err}"
    );

    let other_model = IndexMeta {
        model_id: "BAAI/bge-small-en-v1.5".into(),
        ..meta()
    };
    assert!(matches!(
        store.check_meta(&other_model),
        Err(StoreError::Mismatch { field: "model", .. })
    ));
}

#[test]
fn index_without_meta_is_corrupt() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::create(dir.path()).unwrap();
    assert!(matches!(store.read_meta(), Err(StoreError::Corrupt(_))));
}

#[test]
fn staging_folder_sits_beside_the_index() {
    let dir = PathBuf::from("/data/qorfinder/index");
    assert_eq!(
        staging_dir(&dir),
        PathBuf::from("/data/qorfinder/index.partial")
    );
}

#[test]
fn install_swaps_in_the_new_index_and_removes_the_old() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("index");
    let staging = staging_dir(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("which"), "old").unwrap();
    std::fs::create_dir_all(&staging).unwrap();
    std::fs::write(staging.join("which"), "new").unwrap();

    install(&staging, &dir).unwrap();

    assert_eq!(std::fs::read_to_string(dir.join("which")).unwrap(), "new");
    let mut left: Vec<_> = std::fs::read_dir(root.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    left.sort();
    assert_eq!(left, ["index"]);
}

#[test]
fn install_works_when_there_is_no_old_index() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("index");
    let staging = staging_dir(&dir);
    std::fs::create_dir_all(&staging).unwrap();
    std::fs::write(staging.join("which"), "new").unwrap();

    install(&staging, &dir).unwrap();

    assert_eq!(std::fs::read_to_string(dir.join("which")).unwrap(), "new");
    assert!(!staging.exists());
}
