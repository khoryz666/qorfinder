/// Split text into fixed-size windows of `chunk_size` characters with `overlap`
/// characters shared between consecutive chunks. Runs of whitespace (including
/// newlines) are collapsed to single spaces first.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    assert!(chunk_size > 0, "chunk_size must be positive");
    assert!(
        overlap < chunk_size,
        "overlap must be smaller than chunk_size"
    );

    let cleaned: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = cleaned.chars().collect();
    if chars.len() <= chunk_size {
        return vec![cleaned];
    }

    let step = chunk_size - overlap;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk.trim().to_string());
        if end == chars.len() {
            break;
        }
        start += step;
    }
    chunks.into_iter().filter(|c| !c.is_empty()).collect()
}
