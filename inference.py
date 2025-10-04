"""Submission generation pipeline orchestrator.

Runs the required preprocessing steps for test data and calls the submission
script using a trained experiment directory (``outputs/<run_id>``).

Example
-------
    python inference.py --run-id latest --full --output-dir submissions/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXECUTABLE = sys.executable


def run_command(command: Iterable[str], label: str, dry_run: bool = False) -> None:
    """Execute a command with logging and optional dry-run."""

    cmd_list = list(command)
    display = " ".join(cmd_list)
    print(f"\n=== {label} ===")
    print(display)

    if dry_run:
        print("(dry-run) skipping execution")
        return

    result = subprocess.run(cmd_list, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step '{label}' failed with exit code {result.returncode}")


def resolve_run_id(requested: str) -> Path:
    """Resolve the experiment directory to use for inference."""

    outputs_dir = PROJECT_ROOT / "outputs"
    if requested != "latest":
        run_dir = outputs_dir / requested
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {run_dir}")
        return run_dir

    latest_link = outputs_dir / "latest"
    if latest_link.exists():
        run_dir = latest_link.resolve()
        if run_dir.is_dir():
            return run_dir

    latest_txt = outputs_dir / "latest_run_id.txt"
    if latest_txt.exists():
        run_id = latest_txt.read_text(encoding="utf-8").strip()
        if run_id:
            run_dir = outputs_dir / run_id
            if run_dir.is_dir():
                return run_dir

    raise FileNotFoundError(
        "Could not resolve latest run. Ensure train.py finished successfully or pass --run-id explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference pipeline and generate submission"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="latest",
        help="Experiment directory under outputs/ to use (default: latest symlink)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save submission.csv (default: outputs/<run_id>/)",
    )
    parser.add_argument(
        "--full", action="store_true", help="Generate predictions for all dates"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Earliest candle date for data preparation (forwarded to 1_prepare_data.py)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip llm sentiment stage for test data"
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Regenerate llm outputs even if cached files exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )

    args = parser.parse_args()
    dry = args.dry_run

    # Step 0.1 – preprocess raw news (idempotent)
    run_command(
        [PYTHON_EXECUTABLE, "scripts/0_1_news_preprocess.py"],
        "0_1 news preprocess",
        dry,
    )

    # Step 0.2 – optional LLM inference for test news
    if args.skip_llm:
        print("Skipping LLM sentiment stage (--skip-llm)")
    else:
        llm_cmd = [PYTHON_EXECUTABLE, "scripts/0_2_llm_models.py", "--file", "test"]
        if args.force_llm:
            llm_cmd.append("--force-llm")
        run_command(llm_cmd, "0_2 llm sentiment (test)", dry)

    # Step 0.3 – explode tickers only for test data
    run_command(
        [PYTHON_EXECUTABLE, "scripts/0_3_llm_explode.py", "--test-only"],
        "0_3 llm explode (test)",
        dry,
    )

    # Step 0.4 – aggregate news features for test
    run_command(
        [PYTHON_EXECUTABLE, "scripts/0_4_news_ticker_features.py", "--test-only"],
        "0_4 news features (test)",
        dry,
    )

    # Step 1 – prepare data (reuses same script to refresh holdout dataset)
    prep_cmd = [PYTHON_EXECUTABLE, "scripts/1_prepare_data.py"]
    if args.start_date:
        prep_cmd.extend(["--start-date", args.start_date])
    run_command(prep_cmd, "1 prepare data", dry)

    # Step 4 – generate submission
    if dry:
        run_dir_name = args.run_id if args.run_id != "latest" else "latest"
    else:
        run_dir_name = resolve_run_id(args.run_id).name

    submission_cmd = [
        PYTHON_EXECUTABLE,
        "scripts/4_generate_submission.py",
        "--run-id",
        run_dir_name,
    ]
    if args.output_dir:
        submission_cmd.extend(["--output-dir", args.output_dir])
    if args.full:
        submission_cmd.append("--full")

    run_command(submission_cmd, "4 generate submission", dry)

    print("\n✅ Inference pipeline finished")


if __name__ == "__main__":
    main()
