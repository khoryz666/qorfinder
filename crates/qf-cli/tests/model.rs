use std::process::Command;

fn qorfinder(args: &[&str], cwd: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_qorfinder"))
        .args(args)
        .current_dir(cwd)
        .env_remove("QORFINDER_MODEL")
        .env_remove("QORFINDER_MODEL_DIR")
        .output()
        .unwrap()
}

#[test]
fn unknown_model_is_refused_with_the_supported_ids() {
    let dir = tempfile::tempdir().unwrap();
    let out = qorfinder(&["--model", "all-MiniLM-L6-v2", "search", "x"], dir.path());
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown model"), "{stderr}");
    assert!(
        stderr.contains("sentence-transformers/all-MiniLM-L6-v2"),
        "{stderr}"
    );
}

/// Without --model-dir the files are looked for in models/<name>, with
/// <name> from the --model id.
#[test]
fn model_folder_comes_from_the_model_id() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir(&docs).unwrap();
    let out = qorfinder(
        &[
            "--model",
            "intfloat/e5-base-v2",
            "index",
            "--dry-run",
            docs.to_str().unwrap(),
        ],
        dir.path(),
    );
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("models/e5-base-v2/tokenizer.json"),
        "{stderr}"
    );
    assert!(
        stderr.contains("fetch_model.py --model intfloat/e5-base-v2"),
        "{stderr}"
    );
}
