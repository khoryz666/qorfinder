use std::io::IsTerminal;
use std::sync::Mutex;

use indicatif::{ProgressBar, ProgressStyle};
use qf_core::index::{Progress, Stage};

/// Indexing progress as terminal progress bars, one per stage. Nothing is
/// drawn when stderr is not a terminal, so logs and pipes stay clean.
#[derive(Default)]
pub struct Bars {
    bar: Mutex<Option<ProgressBar>>,
}

impl Progress for Bars {
    fn start(&self, stage: Stage, total: usize) {
        if !std::io::stderr().is_terminal() {
            return;
        }
        let (label, unit) = match stage {
            Stage::Preparing => ("reading ", "files"),
            Stage::Embedding => ("embedding", "chunks"),
        };
        let bar = ProgressBar::new(total as u64).with_style(
            ProgressStyle::with_template(&format!(
                "{label} [{{elapsed_precise}}] [{{bar:32}}] {{pos}}/{{len}} {unit} (eta {{eta}})"
            ))
            .expect("valid progress template")
            .progress_chars("=> "),
        );
        *self.bar.lock().unwrap() = Some(bar);
    }

    fn advance(&self, done: usize) {
        if let Some(bar) = &*self.bar.lock().unwrap() {
            bar.inc(done as u64);
        }
    }

    fn finish(&self, _stage: Stage) {
        if let Some(bar) = self.bar.lock().unwrap().take() {
            bar.finish_and_clear();
        }
    }
}
