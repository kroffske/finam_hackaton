"""Optuna-based hyperparameter search for the LightGBM model.

Usage:
    python scripts/optuna_tune.py --n-trials 25 --study-name lightgbm_mae

The script expects preprocessed data from ``scripts/1_prepare_data.py`` and
stores study artefacts in ``outputs/<timestamp>_optuna/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is on the path
project_root = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(project_root / "src"))

try:
    import optuna
except ImportError as exc:  # pragma: no cover - helpful runtime error
    raise ImportError(
        "Optuna не установлен. Установите пакет: pip install optuna"
    ) from exc

from finam.exclude import exclude_feature_columns
from finam.features_target import extract_targets_dict
from finam.metrics import evaluate_predictions
from finam.model import LightGBMModel


def load_preprocessed_data(
    horizons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load train/val datasets and filter feature columns.

    Args:
        horizons: prediction horizons used in the model (e.g., 1-20 days).

    Returns:
        Tuple of (train_df, val_df, feature_columns).
    """

    preprocessed_dir = project_root / "data" / "preprocessed"
    metadata_path = preprocessed_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            "metadata.json не найден. Сначала запустите scripts/1_prepare_data.py"
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_df = pd.read_csv(preprocessed_dir / "train.csv")
    val_df = pd.read_csv(preprocessed_dir / "val.csv")

    feature_cols = exclude_feature_columns(metadata["feature_columns"])
    if not feature_cols:
        raise ValueError(
            "После исключения признаков не осталось фич. Обновите finam.exclude"
        )

    if val_df.empty:
        raise ValueError(
            "Validation split пуст. Перегенерируйте данные с ненулевой val долей."
        )

    # Ограничиваемся нужными колонками и заменяем NaN (LightGBM сам справится, но
    # явная очистка упрощает дебаг и гарантирует одинаковый пайплайн).
    train_df = train_df[feature_cols + [f"target_return_{h}d" for h in horizons]].copy()
    val_df = val_df[feature_cols + [f"target_return_{h}d" for h in horizons]].copy()

    return train_df, val_df, feature_cols


def build_objective(
    X_train: pd.DataFrame,
    y_train: dict[str, np.ndarray],
    X_val: pd.DataFrame,
    y_val: dict[str, np.ndarray],
    horizons: list[int],
    base_random_state: int,
):
    """Create an Optuna objective that minimises validation MAE."""

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": base_random_state,
        }

        model = LightGBMModel(**params, verbose=-1)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        metrics = evaluate_predictions(y_val, val_preds, horizons=horizons)
        score = metrics.get("mae_mean", np.inf)
        trial.set_user_attr("metrics", metrics)
        return score

    return objective


def save_study(study: optuna.Study, output_dir: Path) -> None:
    """Persist best parameters and all trials to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)

    best_payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "datetime": datetime.utcnow().isoformat(),
        "n_trials": len(study.trials),
    }

    with (output_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs")
    )
    trials_df.to_csv(output_dir / "trials.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna search for LightGBM MAE minimisation"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Количество попыток Optuna (default: 25)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="lightgbm_mae",
        help="Название Optuna study (default: lightgbm_mae)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (например sqlite:///optuna.db). Если не задан, study живёт в памяти.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить существующий study, если он найден в storage",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed для LightGBM (default: 42)"
    )
    parser.add_argument(
        "--horizon-max",
        type=int,
        default=20,
        help="Максимальный горизонт прогнозирования (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = list(range(1, args.horizon_max + 1))

    train_df, val_df, feature_cols = load_preprocessed_data(horizons)

    X_train = train_df[feature_cols]
    y_train = extract_targets_dict(train_df, horizons=horizons)

    X_val = val_df[feature_cols]
    y_val = extract_targets_dict(val_df, horizons=horizons)

    objective = build_objective(X_train, y_train, X_val, y_val, horizons, args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=args.storage,
        load_if_exists=args.resume,
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = project_root / "outputs" / f"{timestamp}_optuna"
    save_study(study, output_dir)

    print("=" * 80)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 80)
    print(f" Best MAE mean: {study.best_value:.6f}")
    print(f" Best params: {study.best_params}")
    print(f" Artefacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
