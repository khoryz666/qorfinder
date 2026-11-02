mod cli;
mod eval;
mod memory;
mod progress;

fn main() {
    if let Err(err) = cli::run() {
        // The reader closed the pipe, as `qorfinder ... | head` does once it
        // has its lines: nothing is wrong, so stop quietly.
        let closed_pipe = err
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::BrokenPipe);
        if !closed_pipe {
            eprintln!("error: {err:#}");
            std::process::exit(1);
        }
    }
}
