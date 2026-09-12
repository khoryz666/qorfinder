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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_text_is_a_single_chunk() {
        assert_eq!(
            chunk_text("hello world", 512, 64),
            vec!["hello world".to_string()]
        );
    }

    #[test]
    fn empty_and_whitespace_text_yield_no_chunks() {
        assert!(chunk_text("", 512, 64).is_empty());
        assert!(chunk_text("   \n\t  ", 512, 64).is_empty());
    }

    #[test]
    fn splits_long_text_and_overlaps() {
        let text = "a".repeat(1000);
        let chunks = chunk_text(&text, 512, 64);
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 512);
        }
        let overlap_end: String = chunks[0].chars().skip(512 - 64).collect();
        let overlap_start: String = chunks[1].chars().take(64).collect();
        assert_eq!(overlap_end, overlap_start);
    }

    #[test]
    fn collapses_internal_whitespace() {
        assert_eq!(
            chunk_text("a\n\nb   c\t d", 512, 64),
            vec!["a b c d".to_string()]
        );
    }

    #[test]
    fn covers_full_text_without_gaps() {
        let text = "x".repeat(1500);
        let chunks = chunk_text(&text, 512, 64);
        let total: usize = chunks.iter().map(|c| c.chars().count()).sum();
        assert!(total >= 1500, "chunks must cover at least the full input");
        let joined: String = chunks.join("");
        assert!(joined.starts_with(&text[..512]));
        assert!(joined.ends_with(&text[text.len() - 512..]));
    }

    #[test]
    #[should_panic]
    fn overlap_must_be_smaller_than_chunk_size() {
        chunk_text("abc", 10, 10);
    }
}
