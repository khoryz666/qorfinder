use qf_core::Chunk;
use qf_core::bm25::Bm25;

fn chunks(texts: &[&str]) -> Vec<Chunk> {
    texts
        .iter()
        .enumerate()
        .map(|(i, text)| Chunk {
            id: i as u32,
            doc_id: 0,
            ord: i as u32,
            text: text.to_string(),
            char_start: 0,
            char_end: text.chars().count(),
            page: None,
        })
        .collect()
}

fn ids(hits: &[qf_core::Hit]) -> Vec<u32> {
    hits.iter().map(|h| h.chunk_id).collect()
}

#[test]
fn stemmed_terms_find_the_matching_chunk_first() {
    let bm25 = Bm25::build(&chunks(&[
        "The lessee shall pay the rent monthly.",
        "Either party may terminate this agreement on notice.",
        "This agreement is governed by the laws of Malaysia.",
    ]))
    .unwrap();
    // "termination" and "terminate" share the stem "termin".
    let hits = bm25.search("termination", 10).unwrap();
    assert_eq!(ids(&hits), [1]);
    assert!(hits[0].score > 0.0);
}

#[test]
fn rarer_terms_and_more_matches_rank_higher() {
    let bm25 = Bm25::build(&chunks(&[
        "agreement agreement agreement",
        "confidential information under this agreement",
        "the governing law",
    ]))
    .unwrap();
    // "confidential" is in one chunk only, so it outweighs "agreement".
    assert_eq!(
        ids(&bm25.search("confidential agreement", 10).unwrap()),
        [1, 0]
    );
}

#[test]
fn query_syntax_characters_are_searched_as_words() {
    let bm25 = Bm25::build(&chunks(&["Section 5: non-compete", "other text"])).unwrap();
    assert_eq!(
        ids(&bm25.search("\"section 5: (non-compete", 10).unwrap()),
        [0]
    );
}

#[test]
fn equal_scores_are_ordered_by_chunk_id() {
    let bm25 = Bm25::build(&chunks(&["royalty fee", "other", "royalty fee"])).unwrap();
    assert_eq!(ids(&bm25.search("royalty", 10).unwrap()), [0, 2]);
    assert!(bm25.search("royalty", 0).unwrap().is_empty());
    assert!(bm25.search("absent", 10).unwrap().is_empty());
}
