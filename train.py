"""End-to-end training pipeline orchestrator.

This script wraps the individual stage scripts under ``scripts/`` into a single
command, so you can reproduce the README workflow in one go.

Example
-------
    python train.py --exp-name lgbm_with_news --model-type lightgbm \\
        --n-estimators 800 --learning-rate 0.03 --start-date 2024-01-01
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
    """Run a subprocess command with pretty logging.

    Args:
        command: Sequence to execute (passed directly to ``subprocess.run``).
        label: Human readable step name for logging.
        dry_run: If True, only print the command without executing it.
    """

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


def ensure_outputs_dir() -> Path:
    """Make sure outputs directory exists and return it."""

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def detect_new_run(outputs_dir: Path, before: set[str]) -> Path | None:
    """Return the directory for the newly created run, if identifiable."""

    after = {d.name for d in outputs_dir.iterdir() if d.is_dir()}
    new_runs = after - before
    if len(new_runs) == 1:
        return outputs_dir / new_runs.pop()

    if not after:
        return None

    # Fallback: pick the most recently modified directory.
    return max(
        (d for d in outputs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )


def update_latest_symlink(run_dir: Path) -> None:
    """Point ``outputs/latest`` (and helper txt) to the provided run directory."""

    outputs_dir = run_dir.parent
    latest_link = outputs_dir / "latest"

    if latest_link.exists() or latest_link.is_symlink():
        try:
            if latest_link.is_dir() and not latest_link.is_symlink():
                latest_link.rmdir()
            else:
                latest_link.unlink()
        except OSError:
            print("Warning: could not remove previous latest link; continuing.")

    try:
        latest_link.symlink_to(run_dir, target_is_directory=True)
    except OSError:
        print("Warning: creating symlink failed; writing latest_run_id.txt instead.")

    latest_txt = outputs_dir / "latest_run_id.txt"
    latest_txt.write_text(run_dir.name + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full training pipeline")
    parser.add_argument(
        "--exp-name", required=True, help="Experiment name for training"
    )
    parser.add_argument(
        "--model-type",
        choices=["lightgbm", "momentum"],
        default="lightgbm",
        help="Model type to train (default: lightgbm)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Earliest candle date to keep when preparing data",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio for scripts/1_prepare_data.py (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio for scripts/1_prepare_data.py (default: 0.15)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip the LLM sentiment stage"
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Regenerate LLM outputs even if cached files exist",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip metrics aggregation via collect_experiments.py",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them"
    )

    # LightGBM hyper-parameters (forwarded as-is)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    dry = args.dry_run

    # Stage 0.1 – news preprocess
    run_command(
        [PYTHON_EXECUTABLE, "scripts/0_1_news_preprocess.py"],
        "0_1 news preprocess",
        dry,
    )

    # Stage 0.2 – optional LLM sentiment
    if not args.skip_llm:
        llm_cmd = [PYTHON_EXECUTABLE, "scripts/0_2_llm_models.py"]
        if args.force_llm:
            llm_cmd.append("--force-llm")
        run_command(llm_cmd, "0_2 llm sentiment", dry)
    else:
        print("Skipping LLM sentiment stage (--skip-llm)")

    # Stage 0.3 – explode news by tickers
    run_command(
        [PYTHON_EXECUTABLE, "scripts/0_3_llm_explode.py", "--all"],
        "0_3 llm explode",
        dry,
    )

    # Stage 0.4 – aggregate news features
    run_command(
        [PYTHON_EXECUTABLE, "scripts/0_4_news_ticker_features.py"],
        "0_4 news features",
        dry,
    )

    # Stage 1 – prepare structured data
    prep_cmd = [
        PYTHON_EXECUTABLE,
        "scripts/1_prepare_data.py",
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
    ]
    if args.start_date:
        prep_cmd.extend(["--start-date", args.start_date])
    run_command(prep_cmd, "1 prepare data", dry)

    # Stage 2 – train model
    outputs_dir = ensure_outputs_dir()
    before_runs = {d.name for d in outputs_dir.iterdir() if d.is_dir()}

    train_cmd = [
        PYTHON_EXECUTABLE,
        "scripts/2_train_model.py",
        "--exp-name",
        args.exp_name,
        "--model-type",
        args.model_type,
        "--n-estimators",
        str(args.n_estimators),
        "--learning-rate",
        str(args.learning_rate),
        "--max-depth",
        str(args.max_depth),
        "--num-leaves",
        str(args.num_leaves),
        "--min-child-samples",
        str(args.min_child_samples),
        "--subsample",
        str(args.subsample),
        "--colsample-bytree",
        str(args.colsample_bytree),
        "--random-state",
        str(args.random_state),
    ]
    run_command(train_cmd, "2 train model", dry)

    if not dry:
        new_run_dir = detect_new_run(outputs_dir, before_runs)
        if new_run_dir is None:
            print(
                "Warning: could not detect new run directory; latest link not updated"
            )
        else:
            print(f"Detected new run directory: {new_run_dir.name}")
            update_latest_symlink(new_run_dir)
            print(f"Updated outputs/latest → {new_run_dir.name}")

    # Optional Stage 3 – collect experiment metrics
    if args.skip_collect:
        print("Skipping collect_experiments stage (--skip-collect)")
    else:
        run_command(
            [PYTHON_EXECUTABLE, "scripts/collect_experiments.py"],
            "collect experiments",
            dry,
        )

    print("\n✅ Training pipeline finished")


if __name__ == "__main__":
    main()
