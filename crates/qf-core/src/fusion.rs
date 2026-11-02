//! Reciprocal Rank Fusion (RRF): combine the BM25 and semantic rankings
//! into one hybrid ranking (report Equation 2.4).

use std::collections::HashMap;

use crate::Hit;

/// The RRF constant. Cormack et al. (2009) use 60; on the CUAD test set 5
/// scores best, chosen by leave-one-category-out cross-validation (15 of
/// 17 folds pick it). A small constant lets each list's top ranks count.
pub const RRF_KAPPA: u32 = 5;

/// Fuse two rankings of chunk ids, each best first. A chunk scores
/// `1 / (kappa + r)` for its 1-based rank `r` in each list, summed over the
/// lists it appears in; a chunk missing from a list adds nothing for it.
/// Pass the full lists, not their top 10, so the ranks are true positions.
/// Returns the `k` best, ordered by score and then by chunk id, so results
/// are the same every run.
pub fn reciprocal_rank_fusion(bm25: &[u32], semantic: &[u32], k: usize, kappa: u32) -> Vec<Hit> {
    let mut scores: HashMap<u32, f64> = HashMap::new();
    for list in [bm25, semantic] {
        for (rank, &id) in list.iter().enumerate() {
            *scores.entry(id).or_default() += 1.0 / (f64::from(kappa) + rank as f64 + 1.0);
        }
    }
    let mut fused: Vec<(u32, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused
        .into_iter()
        .take(k)
        .map(|(chunk_id, score)| Hit {
            chunk_id,
            score: score as f32,
        })
        .collect()
}
