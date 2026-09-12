use std::collections::HashMap;

use anyhow::Result;

use crate::embedder::Embedder;
use crate::store::{SearchHit, Store};

/// Standard Reciprocal Rank Fusion constant.
const RRF_K: f64 = 60.0;

/// Run a hybrid query: dense (usearch, cosine) and lexical (tantivy, BM25)
/// results are each retrieved, then fused by Reciprocal Rank Fusion so a
/// chunk that ranks well on either side contributes, and one that ranks
/// well on both is boosted. `SearchHit::score` on the result is the fused
/// RRF score, not a raw similarity/BM25 value.
pub fn run_query(
    store: &Store,
    embedder: &Embedder,
    query: &str,
    top_k: u64,
) -> Result<Vec<SearchHit>> {
    let vector = embedder.embed_query(query)?;
    let retrieve = (top_k as usize).saturating_mul(4).max(50);

    let mut fused: HashMap<String, (f64, SearchHit)> = HashMap::new();

    for (rank, (key, _distance)) in store
        .search_vector(&vector, retrieve)?
        .into_iter()
        .enumerate()
    {
        let Some(chunk) = store.resolve_vector_key(key)? else {
            continue;
        };
        let entry = fused.entry(chunk.chunk_id).or_insert_with(|| {
            (
                0.0,
                SearchHit {
                    file_path: chunk.file_path,
                    chunk_index: chunk.chunk_index,
                    score: 0.0,
                    text: chunk.text,
                },
            )
        });
        entry.0 += rrf_score(rank);
    }

    for (rank, hit) in store
        .search_lexical(query, retrieve)?
        .into_iter()
        .enumerate()
    {
        let entry = fused.entry(hit.chunk_id).or_insert_with(|| {
            (
                0.0,
                SearchHit {
                    file_path: hit.file_path,
                    chunk_index: hit.chunk_index,
                    score: 0.0,
                    text: hit.text,
                },
            )
        });
        entry.0 += rrf_score(rank);
    }

    let mut ranked: Vec<SearchHit> = fused
        .into_values()
        .map(|(score, mut hit)| {
            hit.score = score as f32;
            hit
        })
        .collect();
    ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
    ranked.truncate(top_k as usize);
    Ok(ranked)
}

fn rrf_score(rank: usize) -> f64 {
    1.0 / (RRF_K + rank as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_score_decreases_with_rank() {
        assert!(rrf_score(0) > rrf_score(1));
        assert!(rrf_score(1) > rrf_score(10));
    }

    // `run_query` itself needs a real Embedder, which downloads a model on
    // first use, so it isn't unit-tested here (see design/EVALUATION.md-style
    // integration coverage in eval.rs instead). What's exercised in this
    // module is the fusion math (above) and, via store's own tests, that
    // search_vector/search_lexical/resolve_vector_key behave as run_query
    // assumes.
}
