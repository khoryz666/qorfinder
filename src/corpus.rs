use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use md5::{Digest as Md5Digest, Md5};
use sha2::Sha256;
use walkdir::WalkDir;

/// Pinned corpus sources (reproducible dev env).
pub const SCIFACT_ZIP_URL: &str =
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip";
pub const SCIFACT_ZIP_MD5: &str = "5f7d1de60b170fc8027bb7898e2efca1";
pub const QURAN_JSON_COMMIT: &str = "791a3cf1376e2851706e9cf01c226f8d3c9d7355";
pub const QURAN_SURAHS: u32 = 114;
pub const QURAN_AYAHS: usize = 6236;

/// Download `url` to `dest`. Skips if the file already exists.
pub fn download(url: &str, dest: &Path) -> Result<()> {
    if dest.exists() {
        tracing::info!("using existing {}", dest.display());
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    tracing::info!("downloading {url}");
    let response = ureq::get(url)
        .timeout(Duration::from_secs(900))
        .call()
        .with_context(|| format!("download failed: {url}"))?;
    let mut reader = response.into_reader();
    let mut file =
        fs::File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    let written = std::io::copy(&mut reader, &mut file)?;
    tracing::info!("downloaded {written} bytes to {}", dest.display());
    Ok(())
}

pub fn md5_hex(path: &Path) -> Result<String> {
    let mut hasher = Md5::new();
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn sha256_hex(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Replace characters that are invalid in file names on any OS.
pub fn sanitize_doc_id(id: &str) -> String {
    id.chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            other => other,
        })
        .collect()
}

fn beir_body(title: Option<&str>, text: &str) -> String {
    match title {
        Some(title) if !title.is_empty() => format!("{title}\n{text}"),
        _ => text.to_string(),
    }
}

fn quran_file_name(surah: u64, ayah: u64) -> String {
    format!("surah-{surah}-ayah-{ayah}.txt")
}

/// Download and unpack the pinned BEIR dataset into
/// `<out>/<dataset>/{corpus,queries.tsv,qrels.tsv}`. Idempotent.
pub fn prepare_beir(dataset: &str, out_dir: &Path) -> Result<PathBuf> {
    anyhow::ensure!(
        dataset == "scifact" || dataset == "nfcorpus",
        "supported BEIR datasets: scifact, nfcorpus (got {dataset})"
    );
    let root = out_dir.join(dataset);
    let corpus_dir = root.join("corpus");
    if count_txt_files(&corpus_dir) > 0
        && root.join("queries.tsv").exists()
        && root.join("qrels.tsv").exists()
    {
        tracing::info!("{dataset} already prepared, skipping");
        return Ok(corpus_dir);
    }

    let zip_path = out_dir.join(format!("{dataset}.zip"));
    let url =
        format!("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip");
    download(&url, &zip_path)?;
    if dataset == "scifact" {
        let actual = md5_hex(&zip_path)?;
        anyhow::ensure!(
            actual == SCIFACT_ZIP_MD5,
            "scifact.zip md5 mismatch: expected {}, got {actual}; delete {} and retry",
            SCIFACT_ZIP_MD5,
            zip_path.display()
        );
    }

    let extract_dir = root.join("extracted");
    fs::create_dir_all(&extract_dir)?;
    unzip(&zip_path, &extract_dir)?;

    let (corpus_json, queries_json, test_tsv) = find_beir_files(&extract_dir)?;

    fs::create_dir_all(&corpus_dir)?;
    let corpus_json = fs::read_to_string(&corpus_json)?;
    let mut count = 0usize;
    for line in corpus_json.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let doc: serde_json::Value = serde_json::from_str(line)?;
        let id = doc["_id"].as_str().unwrap_or_default();
        let title = doc.get("title").and_then(|t| t.as_str());
        let text = doc["text"].as_str().unwrap_or_default();
        let file = corpus_dir.join(format!("{}.txt", sanitize_doc_id(id)));
        fs::write(&file, beir_body(title, text))?;
        count += 1;
    }
    tracing::info!("wrote {count} corpus files to {}", corpus_dir.display());

    let queries_json = fs::read_to_string(&queries_json)?;
    let mut queries_tsv = String::new();
    for line in queries_json.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let q: serde_json::Value = serde_json::from_str(line)?;
        let qid = q["_id"].as_str().unwrap_or_default();
        let text = q["text"].as_str().unwrap_or_default();
        queries_tsv.push_str(&format!("{qid}\t{text}\n"));
    }
    fs::write(root.join("queries.tsv"), queries_tsv)?;
    fs::copy(&test_tsv, root.join("qrels.tsv"))?;
    Ok(corpus_dir)
}

/// Download the pinned quran-json snapshot (Tanzil Uthmani text + English
/// translation) and write one file per ayah. Idempotent.
pub fn prepare_quran(out_dir: &Path) -> Result<PathBuf> {
    let corpus_dir = out_dir.join("quran").join("corpus");
    if count_txt_files(&corpus_dir) == QURAN_AYAHS {
        tracing::info!(
            "quran corpus already prepared ({} files), skipping",
            QURAN_AYAHS
        );
        return Ok(corpus_dir);
    }
    fs::create_dir_all(&corpus_dir)?;
    let base = format!(
        "https://raw.githubusercontent.com/risan/quran-json/{QURAN_JSON_COMMIT}/dist/chapters/en"
    );
    let mut count = 0usize;
    for surah in 1..=QURAN_SURAHS {
        let url = format!("{base}/{surah}.json");
        let response = ureq::get(&url)
            .timeout(Duration::from_secs(120))
            .call()
            .with_context(|| format!("download failed: {url}"))?;
        let mut body = String::new();
        response.into_reader().read_to_string(&mut body)?;
        let surah_json: serde_json::Value =
            serde_json::from_str(&body).with_context(|| format!("parsing {url}"))?;
        let surah_id = surah_json["id"].as_u64().unwrap_or(surah as u64);
        let verses = surah_json["verses"]
            .as_array()
            .with_context(|| format!("no verses in {url}"))?;
        for ayah in verses {
            let ayah_id = ayah["id"].as_u64().unwrap_or_default();
            let text = ayah["text"].as_str().unwrap_or_default();
            let translation = ayah["translation"].as_str().unwrap_or_default();
            let file = corpus_dir.join(quran_file_name(surah_id, ayah_id));
            fs::write(&file, format!("{text}\n{translation}"))?;
            count += 1;
        }
        tracing::info!("surah {surah}/{QURAN_SURAHS} done");
    }
    anyhow::ensure!(
        count == QURAN_AYAHS,
        "expected {QURAN_AYAHS} ayahs, wrote {count}"
    );
    Ok(corpus_dir)
}

fn count_txt_files(dir: &Path) -> usize {
    if !dir.is_dir() {
        return 0;
    }
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file()
                && e.path()
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("txt"))
        })
        .count()
}

fn unzip(zip_path: &Path, dest: &Path) -> Result<()> {
    let file = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let Some(out_path) = entry.enclosed_name().map(|n| dest.join(&n)) else {
            continue;
        };
        if entry.is_dir() {
            fs::create_dir_all(&out_path)?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut out = fs::File::create(&out_path)?;
        std::io::copy(&mut entry, &mut out)?;
    }
    Ok(())
}

fn find_beir_files(dir: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let mut corpus_json = None;
    let mut queries_json = None;
    let mut test_tsv = None;
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        match name.as_ref() {
            "corpus.jsonl" => corpus_json = Some(entry.path().to_path_buf()),
            "queries.jsonl" => queries_json = Some(entry.path().to_path_buf()),
            "test.tsv" => test_tsv = Some(entry.path().to_path_buf()),
            _ => {}
        }
    }
    match (corpus_json, queries_json, test_tsv) {
        (Some(c), Some(q), Some(t)) => Ok((c, q, t)),
        _ => bail!("expected corpus.jsonl, queries.jsonl and test.tsv inside the archive"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_forbidden_filename_chars() {
        assert_eq!(sanitize_doc_id("PLAIN-2"), "PLAIN-2");
        assert_eq!(
            sanitize_doc_id("a/b\\c:d*e?f\"g<h>i|j"),
            "a_b_c_d_e_f_g_h_i_j"
        );
    }

    #[test]
    fn beir_body_prefers_title() {
        assert_eq!(beir_body(Some("Title"), "text"), "Title\ntext");
        assert_eq!(beir_body(None, "text"), "text");
        assert_eq!(beir_body(Some(""), "text"), "text");
    }

    #[test]
    fn quran_file_names_are_stable() {
        assert_eq!(quran_file_name(2, 255), "surah-2-ayah-255.txt");
    }

    #[test]
    fn counts_txt_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "x").unwrap();
        fs::write(dir.path().join("b.md"), "y").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub").join("c.txt"), "z").unwrap();
        assert_eq!(count_txt_files(dir.path()), 2);
    }
}
