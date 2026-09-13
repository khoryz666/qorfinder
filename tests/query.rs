use qorfinder::query::rrf_score;

#[test]
fn rrf_score_decreases_with_rank() {
    assert!(rrf_score(0) > rrf_score(1));
    assert!(rrf_score(1) > rrf_score(10));
}
