fn main() {
    if let Err(err) = qorfinder::cli::run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
