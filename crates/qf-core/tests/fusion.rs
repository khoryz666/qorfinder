use qf_core::fusion::{RRF_KAPPA, reciprocal_rank_fusion};

fn ids(hits: &[qf_core::Hit]) -> Vec<u32> {
    hits.iter().map(|h| h.chunk_id).collect()
}

#[test]
fn a_chunk_first_in_both_lists_scores_two_over_61() {
    let hits = reciprocal_rank_fusion(&[7, 3, 5], &[7, 5, 9], 10, RRF_KAPPA);
    assert_eq!(hits[0].chunk_id, 7);
    assert!((hits[0].score - 2.0 / 61.0).abs() < 1e-7);
    assert!((hits[0].score - 0.0328).abs() < 1e-4);
}

#[test]
fn a_chunk_found_by_one_method_still_scores() {
    let hits = reciprocal_rank_fusion(&[1, 2], &[3], 10, RRF_KAPPA);
    // 1 and 3 are both first in one list only: equal scores, lower id first.
    assert_eq!(ids(&hits), [1, 3, 2]);
    assert!((hits[1].score - 1.0 / 61.0).abs() < 1e-7);
    assert!((hits[2].score - 1.0 / 62.0).abs() < 1e-7);
}

#[test]
fn agreement_between_lists_beats_one_top_rank() {
    // 5 is second in both lists: 2/62 > 1/61 for the chunks first in one.
    let hits = reciprocal_rank_fusion(&[1, 5], &[2, 5], 10, RRF_KAPPA);
    assert_eq!(ids(&hits), [5, 1, 2]);
}

#[test]
fn k_limits_the_result_and_empty_lists_give_nothing() {
    // 1 and 3 each score 1/61 + 1/63, just above 2's 2/62.
    let hits = reciprocal_rank_fusion(&[1, 2, 3], &[3, 2, 1], 2, RRF_KAPPA);
    assert_eq!(ids(&hits), [1, 3]);
    assert!(reciprocal_rank_fusion(&[], &[], 10, RRF_KAPPA).is_empty());
}
