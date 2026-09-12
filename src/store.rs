use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter,
    PointId, PointStruct, PointsIdsList, ScoredPoint, ScrollPointsBuilder, SearchPointsBuilder,
    UpsertPointsBuilder, Value, VectorParamsBuilder, points_selector::PointsSelectorOneOf,
};
use uuid::Uuid;

/// Qdrant's default gRPC port (the Rust client speaks gRPC; REST is 6333).
pub const DEFAULT_URL: &str = "http://localhost:6334";
pub const DEFAULT_COLLECTION: &str = "qorfinder";

/// Points per upsert request; keeps requests small so Qdrant stays responsive.
const UPSERT_BATCH: usize = 512;
/// Points per delete-by-id request.
const DELETE_BATCH: usize = 4096;

pub struct SearchHit {
    pub file_path: String,
    pub chunk_index: u64,
    pub score: f32,
    pub text: String,
}

/// (mtime seconds, size bytes) fingerprint of a file on disk.
pub fn file_fingerprint(meta: &std::fs::Metadata) -> (i64, i64) {
    let secs = meta
        .modified()
        .ok()
        .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    (secs, meta.len() as i64)
}

/// What the store knows about a previously indexed file, recovered from its
/// points' payloads. `chunk_count` is None when the file has no points.
#[derive(Debug, Clone, Copy, Default)]
pub struct FileFingerprint {
    pub mtime_secs: Option<i64>,
    pub size_bytes: Option<i64>,
    pub chunk_count: Option<u64>,
}

impl FileFingerprint {
    /// True when the stored points carry a usable fingerprint (points written
    /// by an older version don't, so the file must be re-indexed).
    pub fn matches(&self, meta: &std::fs::Metadata) -> bool {
        let (mtime, size) = file_fingerprint(meta);
        Some(mtime) == self.mtime_secs && Some(size) == self.size_bytes
    }
}

pub struct Store {
    client: Qdrant,
    collection: String,
}

impl Store {
    /// Connect to Qdrant and ensure the collection exists with the given
    /// vector size. Errors out if an existing collection has different dims.
    pub async fn connect(url: &str, collection: &str, dims: usize) -> Result<Self> {
        let client = Qdrant::from_url(url)
            .skip_compatibility_check()
            .build()
            .with_context(|| format!("failed to connect to Qdrant at {url}"))?;
        let store = Self {
            client,
            collection: collection.to_string(),
        };
        store.ensure_collection(dims).await?;
        Ok(store)
    }

    async fn ensure_collection(&self, dims: usize) -> Result<()> {
        if !self.client.collection_exists(&self.collection).await? {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection)
                        .vectors_config(VectorParamsBuilder::new(dims as u64, Distance::Cosine))
                        .build(),
                )
                .await
                .with_context(|| format!("failed to create collection '{}'", self.collection))?;
            tracing::info!(
                "created collection '{}' ({} dims, cosine)",
                self.collection,
                dims
            );
            return Ok(());
        }
        if let Some(info) = self.client.collection_info(&self.collection).await?.result {
            let actual = info
                .config
                .and_then(|c| c.params)
                .and_then(|p| p.vectors_config)
                .and_then(|v| v.config)
                .and_then(|c| match c {
                    qdrant_client::qdrant::vectors_config::Config::Params(p) => {
                        Some(p.size as usize)
                    }
                    _ => None,
                });
            if let Some(actual) = actual {
                anyhow::ensure!(
                    actual == dims,
                    "collection '{}' has {} dims but the embedding model produces {}; \
                     recreate the collection or use a matching model",
                    self.collection,
                    actual,
                    dims
                );
            }
        }
        Ok(())
    }

    /// Deterministic point ID for a (file, chunk index) pair.
    pub fn point_id(path_str: &str, chunk_index: u64) -> PointId {
        let id = Uuid::new_v5(
            &Uuid::NAMESPACE_OID,
            format!("{path_str}:{chunk_index}").as_bytes(),
        );
        PointId::from(id.to_string())
    }

    /// Build one point for a chunk, carrying path, chunk index, raw text and
    /// the file fingerprint in the payload.
    pub fn point(
        path_str: &str,
        chunk_index: u64,
        text: &str,
        vector: Vec<f32>,
        fingerprint: (i64, i64),
    ) -> PointStruct {
        PointStruct::new(
            Self::point_id(path_str, chunk_index),
            vector,
            payload(path_str, chunk_index, text, fingerprint),
        )
    }

    /// Replace all points belonging to `path` with one point per chunk.
    /// Point IDs are deterministic (UUID v5 of path + chunk index), so
    /// re-indexing the same file is idempotent.
    pub async fn upsert_chunks(
        &self,
        path: &Path,
        chunks: &[String],
        vectors: &[Vec<f32>],
        fingerprint: (i64, i64),
    ) -> Result<usize> {
        anyhow::ensure!(
            chunks.len() == vectors.len(),
            "chunks/vectors length mismatch: {} vs {}",
            chunks.len(),
            vectors.len()
        );
        if chunks.is_empty() {
            return Ok(0);
        }
        let path_str = path.display().to_string();
        let points: Vec<PointStruct> = chunks
            .iter()
            .zip(vectors)
            .enumerate()
            .map(|(i, (text, vector))| {
                Self::point(&path_str, i as u64, text, vector.clone(), fingerprint)
            })
            .collect();
        self.upsert_points(points, true).await
    }

    /// Upsert points in batches. Only the final request waits for durability;
    /// intermediate batches are applied asynchronously by Qdrant.
    pub async fn upsert_points(&self, points: Vec<PointStruct>, wait: bool) -> Result<usize> {
        if points.is_empty() {
            return Ok(0);
        }
        let mut sent = 0usize;
        for batch in points.chunks(UPSERT_BATCH) {
            sent += batch.len();
            let is_last = sent == points.len();
            self.client
                .upsert_points(
                    UpsertPointsBuilder::new(&self.collection, batch.to_vec())
                        .wait(wait && is_last)
                        .build(),
                )
                .await
                .context("failed to upsert points")?;
        }
        Ok(sent)
    }

    /// Remove the points of `path` with chunk indices 0..chunk_count via
    /// deterministic IDs. Much cheaper than a filtered delete (no scan).
    pub async fn delete_file_ids(&self, path: &Path, chunk_count: u64) -> Result<()> {
        let path_str = path.display().to_string();
        let ids: Vec<PointId> = (0..chunk_count)
            .map(|i| Self::point_id(&path_str, i))
            .collect();
        self.delete_ids(ids).await
    }

    /// Delete a batch of points by ID.
    pub async fn delete_ids(&self, ids: Vec<PointId>) -> Result<()> {
        for batch in ids.chunks(DELETE_BATCH) {
            self.client
                .delete_points(
                    DeletePointsBuilder::new(&self.collection)
                        .points(PointsSelectorOneOf::Points(PointsIdsList {
                            ids: batch.to_vec(),
                        }))
                        .wait(true)
                        .build(),
                )
                .await
                .context("failed to delete points")?;
        }
        Ok(())
    }

    /// Remove every point belonging to `path`. Used when the file is gone and
    /// its previous chunk count is unknown.
    pub async fn delete_file(&self, path: &Path) -> Result<()> {
        let filter = Filter::must([Condition::matches("file_path", path.display().to_string())]);
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection)
                    .points(PointsSelectorOneOf::Filter(filter))
                    .wait(true)
                    .build(),
            )
            .await
            .with_context(|| format!("failed to delete points for {}", path.display()))?;
        Ok(())
    }

    /// Page through all points (payload only, no vectors) in the collection.
    async fn scroll_all(&self, filter: Option<Filter>) -> Result<Vec<HashMap<String, Value>>> {
        let mut out = Vec::new();
        let mut offset: Option<PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(&self.collection)
                .limit(1000)
                .with_payload(true)
                .with_vectors(false);
            if let Some(filter) = filter.clone() {
                builder = builder.filter(filter);
            }
            if let Some(offset) = offset.clone() {
                builder = builder.offset(offset);
            }
            let res = self
                .client
                .scroll(builder.build())
                .await
                .context("failed to scroll points")?;
            let page = res.result;
            let page_len = page.len();
            out.extend(page.into_iter().map(|p| p.payload));
            if page_len < 1000 {
                break;
            }
            match res.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }
        Ok(out)
    }

    /// Fingerprint of every indexed file, recovered from point payloads.
    pub async fn file_fingerprints(&self) -> Result<HashMap<String, FileFingerprint>> {
        let mut map: HashMap<String, FileFingerprint> = HashMap::new();
        for payload in self.scroll_all(None).await? {
            if let Some(path) = payload_string(&payload, "file_path") {
                fold_fingerprint(&mut map, path, &payload);
            }
        }
        Ok(map)
    }

    /// Fingerprint of a single file (None when it has no points).
    pub async fn file_info(&self, path: &Path) -> Result<FileFingerprint> {
        let path_str = path.display().to_string();
        let filter = Filter::must([Condition::matches("file_path", path_str.clone())]);
        let mut map: HashMap<String, FileFingerprint> = HashMap::new();
        for payload in self.scroll_all(Some(filter)).await? {
            fold_fingerprint(&mut map, path_str.clone(), &payload);
        }
        Ok(map.remove(&path_str).unwrap_or_default())
    }

    pub async fn search(&self, vector: Vec<f32>, top_k: u64) -> Result<Vec<SearchHit>> {
        let request = SearchPointsBuilder::new(&self.collection, vector, top_k)
            .with_payload(true)
            .build();
        let res = self
            .client
            .search_points(request)
            .await
            .context("qdrant search failed")?;
        Ok(res.result.into_iter().map(hit_from_scored).collect())
    }

    pub async fn count(&self) -> Result<u64> {
        let res = self
            .client
            .count(
                CountPointsBuilder::new(&self.collection)
                    .exact(true)
                    .build(),
            )
            .await
            .context("qdrant count failed")?;
        Ok(res.result.map(|r| r.count).unwrap_or(0))
    }
}

fn payload(
    file_path: &str,
    chunk_index: u64,
    text: &str,
    fingerprint: (i64, i64),
) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    map.insert(
        "file_path".to_string(),
        Value {
            kind: Some(Kind::StringValue(file_path.to_string())),
        },
    );
    map.insert(
        "chunk_index".to_string(),
        Value {
            kind: Some(Kind::IntegerValue(chunk_index as i64)),
        },
    );
    map.insert(
        "text".to_string(),
        Value {
            kind: Some(Kind::StringValue(text.to_string())),
        },
    );
    map.insert(
        "mtime_secs".to_string(),
        Value {
            kind: Some(Kind::IntegerValue(fingerprint.0)),
        },
    );
    map.insert(
        "size_bytes".to_string(),
        Value {
            kind: Some(Kind::IntegerValue(fingerprint.1)),
        },
    );
    map
}

fn payload_string(payload: &HashMap<String, Value>, key: &str) -> Option<String> {
    payload.get(key).and_then(|v| match &v.kind {
        Some(Kind::StringValue(s)) => Some(s.clone()),
        _ => None,
    })
}

fn payload_i64(payload: &HashMap<String, Value>, key: &str) -> Option<i64> {
    payload.get(key).and_then(|v| match &v.kind {
        Some(Kind::IntegerValue(i)) => Some(*i),
        _ => None,
    })
}

fn fold_fingerprint(
    map: &mut HashMap<String, FileFingerprint>,
    path: String,
    payload: &HashMap<String, Value>,
) {
    let entry = map.entry(path).or_default();
    let chunk_index = payload_i64(payload, "chunk_index").unwrap_or(0).max(0) as u64;
    entry.chunk_count = Some(entry.chunk_count.unwrap_or(0).max(chunk_index + 1));
    if entry.mtime_secs.is_none() {
        entry.mtime_secs = payload_i64(payload, "mtime_secs");
    }
    if entry.size_bytes.is_none() {
        entry.size_bytes = payload_i64(payload, "size_bytes");
    }
}

fn hit_from_scored(point: ScoredPoint) -> SearchHit {
    let get_string = |key: &str| -> Option<String> {
        point.payload.get(key).and_then(|v| match &v.kind {
            Some(Kind::StringValue(s)) => Some(s.clone()),
            _ => None,
        })
    };
    let get_u64 = |key: &str| -> Option<u64> {
        point.payload.get(key).and_then(|v| match &v.kind {
            Some(Kind::IntegerValue(i)) => Some(*i as u64),
            _ => None,
        })
    };
    SearchHit {
        file_path: get_string("file_path").unwrap_or_default(),
        chunk_index: get_u64("chunk_index").unwrap_or(0),
        score: point.score,
        text: get_string("text").unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_ids_are_deterministic() {
        let a = Store::point_id("C:\\docs\\a.txt", 3);
        let b = Store::point_id("C:\\docs\\a.txt", 3);
        let c = Store::point_id("C:\\docs\\a.txt", 4);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn fingerprint_matches_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "hello world").unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        let (mtime, size) = file_fingerprint(&meta);
        assert_eq!(size, 11);
        let fp = FileFingerprint {
            mtime_secs: Some(mtime),
            size_bytes: Some(size),
            chunk_count: Some(1),
        };
        assert!(fp.matches(&meta));
    }

    #[test]
    fn fingerprint_rejects_changed_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "hello").unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        let (mtime, size) = file_fingerprint(&meta);
        let fp = FileFingerprint {
            mtime_secs: Some(mtime),
            size_bytes: Some(size),
            chunk_count: Some(1),
        };
        std::fs::write(&path, "hello world, now different").unwrap();
        let new_meta = std::fs::metadata(&path).unwrap();
        assert!(!fp.matches(&new_meta));
    }

    #[test]
    fn fingerprint_without_stored_metadata_never_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "x").unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert!(!FileFingerprint::default().matches(&meta));
    }

    #[test]
    fn fold_fingerprint_recovers_count_and_mtime() {
        let mut map = HashMap::new();
        let p0 = payload("C:\\a.txt", 0, "t0", (111, 5));
        let p1 = payload("C:\\a.txt", 1, "t1", (111, 5));
        fold_fingerprint(&mut map, "C:\\a.txt".to_string(), &p0);
        fold_fingerprint(&mut map, "C:\\a.txt".to_string(), &p1);
        let fp = map["C:\\a.txt"];
        assert_eq!(fp.chunk_count, Some(2));
        assert_eq!(fp.mtime_secs, Some(111));
        assert_eq!(fp.size_bytes, Some(5));
    }

    #[test]
    fn legacy_points_without_fingerprint_are_reindexed() {
        let mut map = HashMap::new();
        let mut legacy = payload("C:\\a.txt", 0, "t0", (0, 0));
        legacy.remove("mtime_secs");
        legacy.remove("size_bytes");
        fold_fingerprint(&mut map, "C:\\a.txt".to_string(), &legacy);
        let fp = map["C:\\a.txt"];
        assert_eq!(fp.chunk_count, Some(1));
        assert_eq!(fp.mtime_secs, None);
        assert_eq!(fp.size_bytes, None);
    }

    #[test]
    fn payload_carries_identity_and_fingerprint() {
        let p = payload("/docs/note.txt", 2, "text", (100, 9));
        assert_eq!(
            payload_string(&p, "file_path").as_deref(),
            Some("/docs/note.txt")
        );
        assert_eq!(payload_i64(&p, "chunk_index"), Some(2));
        assert_eq!(payload_i64(&p, "mtime_secs"), Some(100));
        assert_eq!(payload_i64(&p, "size_bytes"), Some(9));
    }
}
