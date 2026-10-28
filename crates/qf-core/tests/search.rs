use qf_core::search::top_k;

fn ids(hits: &[qf_core::Hit]) -> Vec<u32> {
    hits.iter().map(|h| h.chunk_id).collect()
}

#[test]
fn returns_the_k_best_rows_best_first() {
    let vectors = [
        1.0, 0.0, // 0: score 1.0
        0.0, 1.0, // 1: score 0.0
        0.6, 0.8, // 2: score 0.6
        -1.0, 0.0, // 3: score -1.0
        0.8, 0.6, // 4: score 0.8
    ];
    let hits = top_k(&[1.0, 0.0], &vectors, 2, 3);
    assert_eq!(ids(&hits), [0, 4, 2]);
    assert!((hits[1].score - 0.8).abs() < 1e-6);
}

#[test]
fn matches_a_full_sort() {
    let dim = 8;
    let n = 500;
    // Deterministic pseudo-random vectors.
    let mut state = 42u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let vectors: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    let query: Vec<f32> = (0..dim).map(|_| next()).collect();

    let mut all: Vec<(u32, f32)> = vectors
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, row)| (i as u32, row.iter().zip(&query).map(|(a, b)| a * b).sum()))
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let expected: Vec<u32> = all.iter().take(10).map(|&(id, _)| id).collect();

    assert_eq!(ids(&top_k(&query, &vectors, dim, 10)), expected);
}

#[test]
fn ties_are_broken_by_chunk_id_and_k_can_exceed_n() {
    let vectors = [0.5, 0.5, 0.5];
    assert_eq!(ids(&top_k(&[1.0], &vectors, 1, 10)), [0, 1, 2]);
    assert!(top_k(&[1.0], &vectors, 1, 0).is_empty());
    assert!(top_k(&[1.0], &[], 1, 5).is_empty());
}
