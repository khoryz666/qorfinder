//! QorFinder core library. Everything the CLI and the local API do lives
//! here, so evaluation and the UI run exactly the same code.

pub mod chunk;
pub mod embed;
pub mod format;
pub mod ingest;
pub mod prepare;
pub mod types;

pub use types::{Document, Format};
