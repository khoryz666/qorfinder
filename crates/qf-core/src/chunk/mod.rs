//! Clause-aware chunking of normalised document text.

mod units;

pub use units::{is_clause_start, split_units};
