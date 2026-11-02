//! BM25 keyword baseline: a tantivy index over the same chunks as semantic
//! search, so the two methods differ only in how they score.

use tantivy::collector::TopDocs;
use tantivy::columnar::Column;
use tantivy::query::QueryParser;
use tantivy::schema::{FAST, IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
use tantivy::{Index, IndexReader, doc};

use crate::{Chunk, Hit};

#[derive(Debug, thiserror::Error)]
#[error("BM25 index failed: {0}")]
pub struct Bm25Error(#[from] tantivy::TantivyError);

/// Memory tantivy may use while adding chunks.
const WRITER_MEMORY_BYTES: usize = 50_000_000;

/// An in-memory BM25 index of chunk texts. It is built from the store when
/// needed: indexing the 30 evaluation contracts takes well under a second,
/// so nothing is saved to disk.
pub struct Bm25 {
    reader: IndexReader,
    parser: QueryParser,
}

impl Bm25 {
    /// Index the text of every chunk with tantivy's English analyser
    /// (`en_stem`: lowercase, then Porter stemming), so "terminate" also
    /// matches "termination".
    pub fn build(chunks: &[Chunk]) -> Result<Self, Bm25Error> {
        let mut schema = Schema::builder();
        let indexing = TextFieldIndexing::default()
            .set_tokenizer("en_stem")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text = schema.add_text_field(
            "text",
            TextOptions::default().set_indexing_options(indexing),
        );
        let chunk_id = schema.add_u64_field("chunk_id", FAST);
        let index = Index::create_in_ram(schema.build());
        // One thread writes one segment, so the same chunks always give the
        // same index.
        let mut writer = index.writer_with_num_threads(1, WRITER_MEMORY_BYTES)?;
        for chunk in chunks {
            writer
                .add_document(doc!(text => chunk.text.as_str(), chunk_id => u64::from(chunk.id)))?;
        }
        writer.commit()?;
        let reader = index.reader()?;
        let parser = QueryParser::for_index(&index, vec![text]);
        Ok(Self { reader, parser })
    }

    /// The `k` chunks with the highest BM25 score for `query`, best first.
    /// Chunks sharing no term with the query are not returned, so the list
    /// can be shorter than `k`. Scoring uses tantivy's defaults, k1 = 1.2
    /// and b = 0.75, untuned. Equal scores are ordered by chunk id, so
    /// results are the same every run.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<Hit>, Bm25Error> {
        if k == 0 {
            return Ok(Vec::new());
        }
        // Lenient: a query with characters the query syntax uses (a colon,
        // quotes, brackets) is still searched as plain words.
        let (query, _) = self.parser.parse_query_lenient(query);
        let searcher = self.reader.searcher();
        let top = searcher.search(&query, &TopDocs::with_limit(k).order_by_score())?;
        let ids: Vec<Column<u64>> = searcher
            .segment_readers()
            .iter()
            .map(|segment| segment.fast_fields().u64("chunk_id"))
            .collect::<Result<_, _>>()?;
        let mut hits: Vec<Hit> = top
            .into_iter()
            .filter_map(|(score, address)| {
                let id = ids[address.segment_ord as usize].first(address.doc_id)?;
                Some(Hit {
                    chunk_id: id as u32,
                    score,
                })
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.chunk_id.cmp(&b.chunk_id))
        });
        Ok(hits)
    }
}
