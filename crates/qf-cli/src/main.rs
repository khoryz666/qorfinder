mod cli;
mod eval;
mod memory;
mod progress;
mod serve;

fn main() {
    if let Err(err) = cli::run() {
        // The reader closed the pipe, as `qorfinder ... | head` does once it
        // has its lines: nothing is wrong, so stop quietly.
        let closed_pipe = err
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::BrokenPipe);
        if !closed_pipe {
            eprintln!("error: {}", error_message(&err));
            std::process::exit(1);
        }
    }
}

/// The error and its causes joined by ": ", like `{:#}`, but without a cause
/// that the message already ends with: most library errors already include
/// their source's text.
fn error_message(err: &anyhow::Error) -> String {
    let mut message = err.to_string();
    for cause in err.chain().skip(1) {
        let cause = cause.to_string();
        if !message.ends_with(&cause) {
            message.push_str(": ");
            message.push_str(&cause);
        }
    }
    message
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::error_message;

    #[derive(Debug, thiserror::Error)]
    #[error("index file: {0}")]
    struct Wrapper(#[from] std::io::Error);

    #[test]
    fn a_cause_already_in_the_message_is_not_repeated() {
        let io = std::io::Error::other("no index at x");
        let err = anyhow::Error::from(Wrapper(io));
        assert_eq!(error_message(&err), "index file: no index at x");
    }

    #[test]
    fn context_and_new_causes_are_kept() {
        let err = Err::<(), _>(std::io::Error::other("file missing"))
            .context("cannot load the model")
            .unwrap_err();
        assert_eq!(error_message(&err), "cannot load the model: file missing");
    }
}
