use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use tantivy::collector::TopDocs;
use tantivy::query::{QueryParser, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, STORED, STRING, Schema, TEXT, Value};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

/// A lexical (BM25) search hit.
pub struct LexicalHit {
    pub chunk_id: String,
    pub file_path: String,
    pub chunk_index: u64,
    pub text: String,
    pub score: f32,
}

/// A chunk resolved by its usearch vector key, for the dense side of a
/// hybrid query.
pub struct ResolvedChunk {
    pub file_path: String,
    pub chunk_index: u64,
    pub text: String,
}

struct Fields {
    chunk_id: Field,
    file_path: Field,
    chunk_index: Field,
    text: Field,
    mtime_secs: Field,
    size_bytes: Field,
    vector_key: Field,
}

/// Embedded lexical (BM25) index (tantivy), one document per chunk. Owns the
/// full chunk payload (path, chunk index, text, fingerprint fields, and the
/// linked vector key) — the vector store holds only raw vector geometry, so
/// every hit is ultimately resolved through here.
pub struct LexicalStore {
    index: Index,
    // tantivy's IndexWriter::commit needs &mut self; everything else the
    // writer exposes is already &self. Wrapping it is what lets the whole
    // Store facade stay &self-based, matching Indexer's existing signatures.
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
    fields: Fields,
}

impl LexicalStore {
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create index dir {}", dir.display()))?;

        let mut schema_builder = Schema::builder();
        let chunk_id = schema_builder.add_text_field("chunk_id", STRING | STORED);
        let file_path = schema_builder.add_text_field("file_path", STRING | STORED);
        let chunk_index = schema_builder.add_u64_field(
            "chunk_index",
            tantivy::schema::NumericOptions::default().set_stored().set_fast(),
        );
        // Default (unstemmed) tokenizer: this tool indexes arbitrary
        // personal files, not just English prose, and an English stemmer
        // would silently corrupt matching on non-English text.
        let text = schema_builder.add_text_field("text", TEXT | STORED);
        let mtime_secs = schema_builder
            .add_i64_field("mtime_secs", tantivy::schema::NumericOptions::default().set_stored());
        let size_bytes = schema_builder
            .add_i64_field("size_bytes", tantivy::schema::NumericOptions::default().set_stored());
        let vector_key = schema_builder.add_u64_field(
            "vector_key",
            tantivy::schema::NumericOptions::default()
                .set_stored()
                .set_indexed(),
        );
        let schema = schema_builder.build();

        let index = match Index::open_in_dir(dir) {
            Ok(index) => index,
            Err(_) => Index::create_in_dir(dir, schema)
                .with_context(|| format!("failed to create tantivy index at {}", dir.display()))?,
        };
        let writer: IndexWriter = index
            .writer(50_000_000)
            .context("failed to create tantivy index writer")?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("failed to build tantivy reader")?;

        Ok(Self {
            index,
            writer: Mutex::new(writer),
            reader,
            fields: Fields {
                chunk_id,
                file_path,
                chunk_index,
                text,
                mtime_secs,
                size_bytes,
                vector_key,
            },
        })
    }

    /// Stage an add for one chunk, replacing any existing document with the
    /// same `chunk_id`. Not visible to searches until `commit()`.
    pub fn upsert_chunk(
        &self,
        chunk_id: &str,
        file_path: &str,
        chunk_index: u64,
        text: &str,
        mtime_secs: i64,
        size_bytes: i64,
        vector_key: u64,
    ) -> Result<()> {
        let writer = self.writer.lock().unwrap();
        writer.delete_term(Term::from_field_text(self.fields.chunk_id, chunk_id));
        let mut doc = TantivyDocument::default();
        doc.add_text(self.fields.chunk_id, chunk_id);
        doc.add_text(self.fields.file_path, file_path);
        doc.add_u64(self.fields.chunk_index, chunk_index);
        doc.add_text(self.fields.text, text);
        doc.add_i64(self.fields.mtime_secs, mtime_secs);
        doc.add_i64(self.fields.size_bytes, size_bytes);
        doc.add_u64(self.fields.vector_key, vector_key);
        writer
            .add_document(doc)
            .context("failed to add tantivy document")?;
        Ok(())
    }

    pub fn delete_chunk(&self, chunk_id: &str) {
        self.writer
            .lock()
            .unwrap()
            .delete_term(Term::from_field_text(self.fields.chunk_id, chunk_id));
    }

    pub fn delete_file(&self, file_path: &str) {
        self.writer
            .lock()
            .unwrap()
            .delete_term(Term::from_field_text(self.fields.file_path, file_path));
    }

    pub fn commit(&self) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .commit()
            .context("tantivy commit failed")?;
        self.reader.reload().context("tantivy reader reload failed")?;
        Ok(())
    }

    /// BM25 search over `text`, ranked highest score first.
    pub fn search(&self, query_text: &str, k: usize) -> Result<Vec<LexicalHit>> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.fields.text]);
        let query = query_parser
            .parse_query(query_text)
            .context("failed to parse lexical query")?;
        let top_docs: Vec<(f32, tantivy::DocAddress)> = searcher
            .search(&query, &TopDocs::with_limit(k).order_by_score())
            .context("tantivy search failed")?;
        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(addr)?;
            hits.push(LexicalHit {
                chunk_id: self.get_text(&doc, self.fields.chunk_id),
                file_path: self.get_text(&doc, self.fields.file_path),
                chunk_index: self.get_u64(&doc, self.fields.chunk_index),
                text: self.get_text(&doc, self.fields.text),
                score,
            });
        }
        Ok(hits)
    }

    /// Resolve a usearch vector key back to the chunk it belongs to (for the
    /// dense side of a hybrid query, which only has keys and distances).
    pub fn resolve_vector_key(&self, vector_key: u64) -> Result<Option<ResolvedChunk>> {
        let searcher = self.reader.searcher();
        let term = Term::from_field_u64(self.fields.vector_key, vector_key);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let top_docs: Vec<(f32, tantivy::DocAddress)> = searcher
            .search(&query, &TopDocs::with_limit(1).order_by_score())
            .context("vector-key lookup failed")?;
        match top_docs.into_iter().next() {
            Some((_, addr)) => {
                let doc: TantivyDocument = searcher.doc(addr)?;
                Ok(Some(ResolvedChunk {
                    file_path: self.get_text(&doc, self.fields.file_path),
                    chunk_index: self.get_u64(&doc, self.fields.chunk_index),
                    text: self.get_text(&doc, self.fields.text),
                }))
            }
            None => Ok(None),
        }
    }

    fn get_text(&self, doc: &TantivyDocument, field: Field) -> String {
        doc.get_first(field)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    }

    fn get_u64(&self, doc: &TantivyDocument, field: Field) -> u64 {
        doc.get_first(field).and_then(|v| v.as_u64()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(dir: &Path) -> LexicalStore {
        LexicalStore::open(dir).unwrap()
    }

    #[test]
    fn upsert_commit_search_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        s.upsert_chunk("a.txt:0", "a.txt", 0, "the quick brown fox", 1, 2, 10)
            .unwrap();
        s.commit().unwrap();
        let hits = s.search("quick fox", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file_path, "a.txt");
        assert_eq!(hits[0].chunk_index, 0);
    }

    #[test]
    fn upsert_same_chunk_id_replaces_not_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        s.upsert_chunk("a.txt:0", "a.txt", 0, "original text", 1, 2, 10)
            .unwrap();
        s.commit().unwrap();
        s.upsert_chunk("a.txt:0", "a.txt", 0, "updated text", 3, 4, 10)
            .unwrap();
        s.commit().unwrap();
        let hits = s.search("updated", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "updated text");
        assert!(s.search("original", 5).unwrap().is_empty());
    }

    #[test]
    fn delete_chunk_removes_it() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        s.upsert_chunk("a.txt:0", "a.txt", 0, "hello world", 1, 2, 10)
            .unwrap();
        s.commit().unwrap();
        s.delete_chunk("a.txt:0");
        s.commit().unwrap();
        assert!(s.search("hello", 5).unwrap().is_empty());
    }

    #[test]
    fn delete_file_removes_all_its_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        s.upsert_chunk("a.txt:0", "a.txt", 0, "hello world", 1, 2, 10)
            .unwrap();
        s.upsert_chunk("a.txt:1", "a.txt", 1, "hello again", 1, 2, 11)
            .unwrap();
        s.upsert_chunk("b.txt:0", "b.txt", 0, "hello elsewhere", 1, 2, 12)
            .unwrap();
        s.commit().unwrap();
        s.delete_file("a.txt");
        s.commit().unwrap();
        let hits = s.search("hello", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file_path, "b.txt");
    }

    #[test]
    fn resolve_vector_key_finds_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        s.upsert_chunk("a.txt:0", "a.txt", 0, "hello world", 1, 2, 42)
            .unwrap();
        s.commit().unwrap();
        let resolved = s.resolve_vector_key(42).unwrap().unwrap();
        assert_eq!(resolved.file_path, "a.txt");
        assert_eq!(resolved.text, "hello world");
    }

    #[test]
    fn resolve_missing_vector_key_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        assert!(s.resolve_vector_key(999).unwrap().is_none());
    }

    #[test]
    fn reopening_existing_index_preserves_documents() {
        let dir = tempfile::tempdir().unwrap();
        {
            let s = store(dir.path());
            s.upsert_chunk("a.txt:0", "a.txt", 0, "persisted text", 1, 2, 10)
                .unwrap();
            s.commit().unwrap();
        }
        let s = store(dir.path());
        assert_eq!(s.search("persisted", 5).unwrap().len(), 1);
    }
}
