//! Exact semantic search: score every stored vector against the query.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::Hit;

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
