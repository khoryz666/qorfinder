use std::path::PathBuf;

use qf_core::store::Store;
use qf_core::{Chunk, Document, Format};

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
        store.insert_document(&document(0)).unwrap();
        store.insert_document(&document(1)).unwrap();
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
    store.insert_document(&document(0)).unwrap();
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
