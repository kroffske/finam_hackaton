"""
Model Training Script

Этот скрипт:
1. Загружает preprocessed данные из data/preprocessed/
2. Обучает модель (LightGBM, Momentum, или другую)
3. Сохраняет в outputs/<timestamp>_<exp_name>/:
   - model_*.pkl (сериализованные модели)
   - config.yaml (параметры эксперимента)
   - metrics.json (метрики на train и val)
   - feature_importance.csv (для LightGBM)
   - predictions_val.csv (предсказания на validation)

Usage:
    python scripts/2_train_model.py --exp-name lgbm_basic
    python scripts/2_train_model.py --exp-name lgbm_calibrated --model-type lightgbm --calibrate
    python scripts/2_train_model.py --exp-name momentum_baseline --model-type momentum
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import json

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import joblib

from finam.features_target import extract_targets_dict
from finam.exclude import exclude_feature_columns
from finam.model import MomentumBaseline, LightGBMModel
from finam.metrics import evaluate_predictions, print_metrics


def train_model(
    exp_name: str,
    model_type: str = "lightgbm",
    n_estimators: int = 207,
    learning_rate: float = 0.01,
    max_depth: int = 5,
    num_leaves: int = 61,
    min_child_samples: int = 67,
    subsample: float = 0.852,
    colsample_bytree: float = 0.837,
    random_state: int = 42,
):
    """
    Обучение модели с сохранением результатов

    Args:
        exp_name: название эксперимента
        model_type: тип модели ('lightgbm' или 'momentum')
        остальные параметры: для LightGBM
    """
    print("=" * 80)
    print("MODEL TRAINING PIPELINE")
    print("=" * 80 + "\n")

    # Создаем директорию для эксперимента
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = project_root / "outputs" / f"{timestamp}_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output dir: {exp_dir}\n")

    # ========================================================================
    # 1. Загрузка preprocessed данных
    # ========================================================================
    print("[1/5] Loading preprocessed data...")

    preprocessed_dir = project_root / "data" / "preprocessed"

    if not preprocessed_dir.exists():
        print("ERROR Preprocessed data not found!")
        print("   Run first: python scripts/1_prepare_data.py")
        return

    train_df = pd.read_csv(preprocessed_dir / "train.csv", parse_dates=["begin"])
    val_df = pd.read_csv(preprocessed_dir / "val.csv", parse_dates=["begin"])

    # Загружаем metadata
    with open(preprocessed_dir / "metadata.json", "r") as f:
        data_metadata = json.load(f)

    print(f"   OK Train: {len(train_df)} rows ({data_metadata['train_period']})")
    print(f"   OK Val:   {len(val_df)} rows ({data_metadata['val_period']})")
    print(f"   OK Features: {data_metadata['n_features']}\n")

    # ========================================================================
    # 2. Подготовка данных для обучения
    # ========================================================================
    print("[2/5] Preparing features and targets...")

    feature_cols = data_metadata["feature_columns"]
    original_feature_count = len(feature_cols)
    feature_cols = exclude_feature_columns(feature_cols)
    if not feature_cols:
        raise ValueError(
            "После исключения списка EXCLUDED_FEATURES не осталось признаков. Обновите finam.exclude или переподготовьте данные."
        )
    if len(feature_cols) != original_feature_count:
        print(
            f"   [INFO] Признаков для обучения: {len(feature_cols)} (отфильтровано {original_feature_count - len(feature_cols)})"
        )

    X_train = train_df[feature_cols]
    y_returns_train = extract_targets_dict(train_df, horizons=list(range(1, 21)))

    X_val = val_df[feature_cols]
    y_returns_val = extract_targets_dict(val_df, horizons=list(range(1, 21)))

    print(f"   OK X_train shape: {X_train.shape}")
    print(f"   OK X_val shape:   {X_val.shape}")
    print(f"   OK Targets: {len(y_returns_train)} horizons (1-20 days)\n")

    # ========================================================================
    # 3. Создание и обучение модели
    # ========================================================================
    print(f"[3/5] Training {model_type.upper()} model...")

    if model_type.lower() == "lightgbm":
        model = LightGBMModel(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1,
        )

        model_params = {
            "type": "LightGBM",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
        }

    elif model_type.lower() == "momentum":
        model = MomentumBaseline(window_size=5, scaling_1d=0.3, scaling_20d=1.0)

        model_params = {
            "type": "MomentumBaseline",
            "window_size": 5,
            "scaling_1d": 0.3,
            "scaling_20d": 1.0,
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Обучение
    model.fit(X_train, y_returns_train)

    print("   OK Training complete!\n")

    # ========================================================================
    # 4. Предсказание и оценка
    # ========================================================================
    print("[4/5] Evaluating on train, val, and test...")

    # Загружаем test данные
    test_df = pd.read_csv(preprocessed_dir / "test.csv", parse_dates=["begin"])
    X_test = test_df[feature_cols].fillna(0)
    y_returns_test = extract_targets_dict(test_df, horizons=list(range(1, 21)))

    def _empty_predictions() -> dict:
        return {f"pred_return_{horizon}d": np.array([]) for horizon in range(1, 21)}

    def _empty_metrics() -> dict:
        metrics = {f"mae_{horizon}d": np.nan for horizon in range(1, 21)}
        metrics["mae_mean"] = np.nan
        return metrics

    def _safe_predict(split_name: str, features: pd.DataFrame, targets: dict) -> tuple[dict, dict]:
        if features is None or len(features) == 0:
            print(f"   [WARN] {split_name} split is empty, skipping evaluation.")
            return _empty_predictions(), _empty_metrics()
        preds = model.predict(features)
        return preds, evaluate_predictions(targets, preds)

    # Предсказания на train/val/test c обработкой пустых сплитов
    train_preds, train_metrics = _safe_predict("Train", X_train, y_returns_train)
    val_preds, val_metrics = _safe_predict("Validation", X_val, y_returns_val)
    test_preds, test_metrics = _safe_predict("Test", X_test, y_returns_test)

    print("\n" + "=" * 70)
    print("TRAIN METRICS:")
    print("=" * 70)
    print_metrics(train_metrics, model_name="Train")

    print("\n" + "=" * 70)
    print("VALIDATION METRICS:")
    print("=" * 70)
    print_metrics(val_metrics, model_name="Validation")

    print("\n" + "=" * 70)
    print("TEST METRICS:")
    print("=" * 70)
    print_metrics(test_metrics, model_name="Test")

    # ========================================================================
    # 5. Сохранение результатов
    # ========================================================================
    print(f"\n[5/5] Saving results to {exp_dir}...")

    # 5.1 Сохранение моделей
    if model_type.lower() == "lightgbm":
        # Сохраняем все 20 моделей
        for horizon in range(1, 21):
            if horizon in model.models:
                joblib.dump(
                    model.models[horizon], exp_dir / f"model_return_{horizon}d.pkl"
                )
        print(
            f"   OK Saved {len(model.models)} models (model_return_1d.pkl through model_return_20d.pkl)"
        )
    else:
        joblib.dump(model, exp_dir / "model.pkl")
        print("   OK Saved model.pkl")

    # 5.2 Сохранение конфигурации
    config = {
        "experiment": {
            "name": exp_name,
            "timestamp": timestamp,
            "model_type": model_type,
            "created_at": datetime.now().isoformat(),
        },
        "model": model_params,
        "features": {
            "count": len(feature_cols),
            "columns": feature_cols,
            "windows": data_metadata["windows"],
            "cross_sectional": data_metadata["include_cross_sectional"],
            "interactions": data_metadata["include_interactions"],
        },
        "data": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "train_period": data_metadata["train_period"],
            "val_period": data_metadata["val_period"],
        },
        "results": {
            "train": {
                "mae_1d": float(train_metrics.get("mae_1d", np.nan)),
                "mae_20d": float(train_metrics.get("mae_20d", np.nan)),
                "mae_mean": float(train_metrics.get("mae_mean", np.nan)),
            },
            "val": {
                "mae_1d": float(val_metrics.get("mae_1d", np.nan)),
                "mae_20d": float(val_metrics.get("mae_20d", np.nan)),
                "mae_mean": float(val_metrics.get("mae_mean", np.nan)),
            },
            "test": {
                "mae_1d": float(test_metrics.get("mae_1d", np.nan)),
                "mae_20d": float(test_metrics.get("mae_20d", np.nan)),
                "mae_mean": float(test_metrics.get("mae_mean", np.nan)),
            },
        },
    }

    import yaml

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("   OK Saved config.yaml")

    # 5.3 Сохранение метрик в JSON
    # Конвертируем numpy types в Python native
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    metrics_output = {
        "train": convert_numpy(train_metrics),
        "val": convert_numpy(val_metrics),
        "test": convert_numpy(test_metrics),
    }

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)

    print("   OK Saved metrics.json")

    # 5.4 Сохранение feature importance (для LightGBM)
    if model_type.lower() == "lightgbm":
        importance_df = model.get_feature_importance()
        importance_df.to_csv(exp_dir / "feature_importance.csv", index=False)
        print("   OK Saved feature_importance.csv")

    # 5.5 Сохранение предсказаний на val (все 20 горизонтов)
    predictions_df = val_df[["ticker", "begin"]].copy()

    # Добавляем предсказания для всех горизонтов
    for horizon in range(1, 21):
        pred_key = f"pred_return_{horizon}d"
        target_key = f"target_return_{horizon}d"
        if pred_key in val_preds:
            predictions_df[pred_key] = val_preds[pred_key]
        if target_key in y_returns_val:
            predictions_df[target_key] = y_returns_val[target_key]

    predictions_df.to_csv(exp_dir / "predictions_val.csv", index=False)
    print("   OK Saved predictions_val.csv")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80 + "\n")

    print(f" Experiment: {exp_name}")
    print(f"   Output dir: {exp_dir}")
    print("\n   Validation metrics:")
    print(f"      MAE 1d:   {val_metrics['mae_1d']:.6f}")
    print(f"      MAE 20d:  {val_metrics['mae_20d']:.6f}")
    print(f"      MAE mean: {val_metrics['mae_mean']:.6f}")

    print("\n Next steps:")
    print("   # Evaluate model")
    print(f"   python scripts/3_evaluate.py --exp-dir {exp_dir.name}")
    print("\n   # Generate submission files")
    print(f"   python scripts/4_generate_submission.py --run-id {exp_dir.name}")
    print("\n   # Collect all experiments")
    print("   python scripts/collect_experiments.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model and save results")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--model-type",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "momentum"],
        help="Model type (default: lightgbm)",
    )

    # LightGBM parameters
    parser.add_argument(
        "--n-estimators", type=int, default=207, help="Number of trees (default: 207)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=5, help="Max depth (default: 5)"
    )
    parser.add_argument(
        "--num-leaves", type=int, default=61, help="Number of leaves (default: 61)"
    )
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=67,
        help="Min child samples (default: 67)",
    )
    parser.add_argument(
        "--subsample", type=float, default=0.852, help="Subsample ratio (default: 0.852)"
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.837,
        help="Feature subsample ratio (default: 0.837)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state (default: 42)"
    )

    args = parser.parse_args()

    train_model(
        exp_name=args.exp_name,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
    )
