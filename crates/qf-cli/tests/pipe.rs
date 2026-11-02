use std::io::Read;
use std::process::{Command, Stdio};

/// `qorfinder ... | head` closes the pipe after a few lines; the command
/// must then stop quietly instead of panicking with "Broken pipe".
#[test]
#[ignore = "needs models/ from scripts/fetch_model.py"]
fn closed_output_pipe_is_not_an_error() {
    let docs = tempfile::tempdir().unwrap();
    // Far more chunk lines than a pipe buffers, so writes go on after the
    // reader has gone.
    std::fs::write(docs.path().join("long.txt"), "word ".repeat(200_000)).unwrap();
    let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(qf_core::embed::DEFAULT_MODEL.dir_name());

    let mut child = Command::new(env!("CARGO_BIN_EXE_qorfinder"))
        .arg("--model-dir")
        .arg(&model_dir)
        .args(["index", "--dry-run"])
        .arg(docs.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut first = [0u8; 16];
    stdout.read_exact(&mut first).unwrap();
    drop(stdout);
    let out = child.wait_with_output().unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}
