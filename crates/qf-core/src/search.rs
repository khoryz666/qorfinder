//! Exact semantic search: score every stored vector against the query.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::embed::{EmbedError, Embedder};
use crate::store::{Store, StoreError, VECTORS_FILE, read_vectors};
use crate::{Hit, IndexMeta};

#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("embedding the query failed: {0}")]
    Embed(#[from] EmbedError),
}

/// Hits for one query, with the two parts of its latency timed apart.
#[derive(Debug)]
pub struct SearchResult {
    pub hits: Vec<Hit>,
    /// Embedding the query.
    pub encode_time: Duration,
    /// Scoring every stored vector.
    pub scan_time: Duration,
}

/// An opened index ready to answer queries: the store for chunk text and
/// all vectors in memory.
pub struct Searcher<'a> {
    store: Store,
    vectors: Vec<f32>,
    dim: usize,
    embedder: &'a Embedder,
}

impl<'a> Searcher<'a> {
    /// Open the index in `index_dir`. It must have been built with exactly
    /// the settings in `current` (same model, weights and chunking).
    pub fn open(
        index_dir: &Path,
        embedder: &'a Embedder,
        current: &IndexMeta,
    ) -> Result<Self, SearchError> {
        let store = Store::open(index_dir)?;
        let meta = store.check_meta(current)?;
        let vectors = read_vectors(&index_dir.join(VECTORS_FILE), meta.dim)?;
        let rows = vectors.len() / meta.dim;
        let chunks = store.chunk_count()?;
        if rows != chunks {
            return Err(StoreError::Corrupt(format!("{rows} vectors for {chunks} chunks")).into());
        }
        Ok(Self {
            store,
            vectors,
            dim: meta.dim,
            embedder,
        })
    }

    /// The store, to look up the chunks and documents of hits.
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The `k` chunks closest in meaning to `query`.
    pub fn search(&self, query: &str, k: usize) -> Result<SearchResult, SearchError> {
        let started = Instant::now();
        let vector = self.embedder.embed_query(query)?;
        let encode_time = started.elapsed();
        let started = Instant::now();
        let hits = top_k(&vector, &self.vectors, self.dim, k);
        Ok(SearchResult {
            hits,
            encode_time,
            scan_time: started.elapsed(),
        })
    }
}

/// The `k` rows of `vectors` (row-major, `dim` floats each) with the
/// highest dot product with `query`, best first. Vectors are unit length,
/// so the dot product is the cosine similarity. A min-heap holding the best
/// `k` so far keeps the scan at O(n log k) time and O(k) extra memory.
/// Equal scores are ordered by chunk id, so results are the same every run.
pub fn top_k(query: &[f32], vectors: &[f32], dim: usize, k: usize) -> Vec<Hit> {
    assert_eq!(query.len(), dim, "query has the wrong dimension");
    if k == 0 {
        return Vec::new();
    }
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (id, row) in vectors.chunks_exact(dim).enumerate() {
        let score: f32 = row.iter().zip(query).map(|(a, b)| a * b).sum();
        heap.push(Reverse(Ranked(Hit {
            chunk_id: id as u32,
            score,
        })));
        if heap.len() > k {
            heap.pop(); // drop the worst of the k + 1
        }
    }
    // Ascending order of Reverse(...) is best first.
    heap.into_sorted_vec()
        .into_iter()
        .map(|Reverse(Ranked(hit))| hit)
        .collect()
}

/// A hit ordered by score, then by lower chunk id first.
struct Ranked(Hit);

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .score
            .total_cmp(&other.0.score)
            .then_with(|| other.0.chunk_id.cmp(&self.0.chunk_id))
    }
}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked {}
