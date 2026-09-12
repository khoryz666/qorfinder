use anyhow::Result;

use crate::embedder::Embedder;
use crate::store::{SearchHit, Store};

pub async fn run_query(
    store: &Store,
    embedder: &Embedder,
    query: &str,
    top_k: u64,
) -> Result<Vec<SearchHit>> {
    let vector = embedder.embed_query(query)?;
    store.search(vector, top_k).await
}
