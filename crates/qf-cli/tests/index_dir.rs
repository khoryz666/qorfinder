use std::fs;
use std::path::Path;
use std::process::Command;

/// `qorfinder index <folder> --index-dir <index>` with no model, so only
/// the checks that run before the model loads are exercised.
fn index(folder: &Path, index_dir: &Path, root: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_qorfinder"))
        .arg("--index-dir")
        .arg(index_dir)
        .arg("--model-dir")
        .arg(root.join("no-model"))
        .arg("index")
        .arg(folder)
        .output()
        .unwrap()
}

/// Pointing the index at the user's own files, by a typo or a wrong path
/// from the UI, must fail and leave every file where it was.
#[test]
fn index_dir_holding_other_files_is_refused_and_kept() {
    let root = tempfile::tempdir().unwrap();
    let docs = root.path().join("contracts");
    let work = root.path().join("work");
    for dir in [&docs, &work] {
        fs::create_dir(dir).unwrap();
        fs::write(dir.join("lease.txt"), "The tenant shall pay rent.").unwrap();
    }

    for index_dir in [&work, &docs] {
        let out = index(&docs, index_dir, root.path());
        assert!(!out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("not a QorFinder index") || stderr.contains("inside it"),
            "{stderr}"
        );
    }
    for dir in [&docs, &work] {
        assert_eq!(
            fs::read_to_string(dir.join("lease.txt")).unwrap(),
            "The tenant shall pay rent."
        );
    }
    let mut left: Vec<_> = fs::read_dir(root.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    left.sort();
    assert_eq!(left, ["contracts", "work"]);
}
