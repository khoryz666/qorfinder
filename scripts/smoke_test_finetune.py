#!/usr/bin/env python3
"""Smoke-test finetune_minilm.py: a few steps, checkpoint, resume, log.

This proves the three things the full run depends on, in under a minute:

  1. checkpointing  -- a step_* folder appears with model.safetensors,
     config.json, tokenizer.json and trainer.pt;
  2. resume         -- a second invocation picks up where the first stopped
     (its log says "resumed from" and its first step is one past the first
     run's last step);
  3. logging        -- the plain-text log has timestamped lines, "checkpoint
     saved", "resumed from", per-step loss and the final summary.

It is deliberately tiny (a handful of optimizer steps on a few dozen pairs)
and uses a separate --out-dir so it can never be confused with, or resumed
by, the real training run. It does not download anything.

Usage (in the reference devshell, which has PyTorch):

    nix develop .#reference --command python3 scripts/smoke_test_finetune.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINETUNE = REPO / "scripts" / "finetune_minilm.py"
SMOKE_DIR = REPO / "results" / "finetune-minilm" / "smoke"
LOG = SMOKE_DIR / "train.log"

# Common flags for both runs. steps_per_epoch=6, checkpoint every 3 steps,
# batch 8, and a global per-category cap so only ~136 pairs are built.
BASE_ARGS = [
    "--out-dir", str(SMOKE_DIR),
    "--epochs", "2",
    "--batch-size", "8",
    "--steps-per-epoch", "6",
    "--steps-per-checkpoint", "3",
    "--max-pairs-per-category", "8",
    "--threads", "4",
    "--seed", "2026",
]


def run_finetune(extra: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(FINETUNE), *BASE_ARGS, *extra]
    return subprocess.run(cmd, text=True, capture_output=True)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    if LOG.exists():
        print("--- last log ---", file=sys.stderr)
        print(LOG.read_text(encoding="utf-8")[-3000:], file=sys.stderr)
    sys.exit(1)


def expect(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def main() -> int:
    # Clean start, so a previous smoke run cannot mask a regression.
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)

    # Run 1: three steps, checkpoint at step 3, then stop (--max-steps 3).
    print("run 1: 3 steps, checkpoint, stop")
    first = run_finetune(["--max-steps", "3"])
    if first.returncode != 0:
        fail(f"run 1 exited {first.returncode}\nSTDOUT:\n{first.stdout}\nSTDERR:\n{first.stderr}")
    log1 = LOG.read_text(encoding="utf-8")
    expect("checkpoint saved step=3" in log1, "run 1 log has no 'checkpoint saved step=3'")
    expect("--- training summary ---" in log1, "run 1 log has no summary block")
    ckpt3 = SMOKE_DIR / "checkpoints" / "step_0000003"
    expect((ckpt3 / "model.safetensors").is_file(), "step_0000003/model.safetensors missing")
    expect((ckpt3 / "config.json").is_file(), "step_0000003/config.json missing")
    expect((ckpt3 / "tokenizer.json").is_file(), "step_0000003/tokenizer.json missing")
    expect((ckpt3 / "trainer.pt").is_file(), "step_0000003/trainer.pt missing")

    # Run 2: resume from step 3 and run steps 4..6.
    print("run 2: resume from step 3, run steps 4..6")
    second = run_finetune(["--max-steps", "6"])
    if second.returncode != 0:
        fail(f"run 2 exited {second.returncode}\nSTDOUT:\n{second.stdout}\nSTDERR:\n{second.stderr}")
    log2 = LOG.read_text(encoding="utf-8")
    expect("resumed from" in log2, "run 2 log has no 'resumed from'")
    expect("training begins at global step 3" in log2, "run 2 did not begin at step 3")
    expect("checkpoint saved step=6" in log2, "run 2 log has no 'checkpoint saved step=6'")
    ckpt6 = SMOKE_DIR / "checkpoints" / "step_0000006"
    expect((ckpt6 / "model.safetensors").is_file(), "step_0000006/model.safetensors missing")
    expect((ckpt6 / "trainer.pt").is_file(), "step_0000006/trainer.pt missing")

    # Timestamps: every log line begins with YYYY-MM-DD HH:MM:SS.
    import re

    timestamp = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}  ")
    for i, line in enumerate(log2.splitlines(), 1):
        expect(timestamp.match(line) is not None, f"log line {i} has no timestamp: {line!r}")

    # The final model (the path the summary reports) must exist.
    final = SMOKE_DIR / "model" / "model.safetensors"
    expect(final.is_file(), "final model/model.safetensors missing")

    print("smoke test passed: checkpointing, resume and logging all work.")
    print(f"artifacts in {SMOKE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
