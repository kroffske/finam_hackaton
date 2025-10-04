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
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import joblib

from finam.utils import get_feature_columns
from finam.features_target import extract_targets_dict
from finam.model import MomentumBaseline, LightGBMModel
from finam.metrics import evaluate_predictions, print_metrics


def train_model(
    exp_name: str,
    model_type: str = 'lightgbm',
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
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
    exp_dir = project_root / 'outputs' / f"{timestamp}_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output dir: {exp_dir}\n")

    # ========================================================================
    # 1. Загрузка preprocessed данных
    # ========================================================================
    print("[1/5] Loading preprocessed data...")

    preprocessed_dir = project_root / 'data' / 'preprocessed'

    if not preprocessed_dir.exists():
        print(f"ERROR Preprocessed data not found!")
        print(f"   Run first: python scripts/1_prepare_data.py")
        return

    train_df = pd.read_csv(preprocessed_dir / 'train.csv', parse_dates=['begin'])
    val_df = pd.read_csv(preprocessed_dir / 'val.csv', parse_dates=['begin'])

    # Загружаем metadata
    with open(preprocessed_dir / 'metadata.json', 'r') as f:
        data_metadata = json.load(f)

    print(f"   OK Train: {len(train_df)} rows ({data_metadata['train_period']})")
    print(f"   OK Val:   {len(val_df)} rows ({data_metadata['val_period']})")
    print(f"   OK Features: {data_metadata['n_features']}\n")

    # ========================================================================
    # 2. Подготовка данных для обучения
    # ========================================================================
    print("[2/5] Preparing features and targets...")

    feature_cols = data_metadata['feature_columns']

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

    if model_type.lower() == 'lightgbm':
        model = LightGBMModel(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1
        )

        model_params = {
            'type': 'LightGBM',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state
        }

    elif model_type.lower() == 'momentum':
        model = MomentumBaseline(
            window_size=5,
            scaling_1d=0.3,
            scaling_20d=1.0
        )

        model_params = {
            'type': 'MomentumBaseline',
            'window_size': 5,
            'scaling_1d': 0.3,
            'scaling_20d': 1.0
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Обучение
    model.fit(X_train, y_returns_train)

    print(f"   OK Training complete!\n")

    # ========================================================================
    # 4. Предсказание и оценка
    # ========================================================================
    print("[4/5] Evaluating on train, val, and test...")

    # Загружаем test данные
    test_df = pd.read_csv(preprocessed_dir / 'test.csv', parse_dates=['begin'])
    X_test = test_df[feature_cols].fillna(0)
    y_returns_test = extract_targets_dict(test_df, horizons=list(range(1, 21)))

    # Предсказания на train
    train_preds = model.predict(X_train)
    train_metrics = evaluate_predictions(y_returns_train, train_preds)

    # Предсказания на val
    val_preds = model.predict(X_val)
    val_metrics = evaluate_predictions(y_returns_val, val_preds)

    # Предсказания на test
    test_preds = model.predict(X_test)
    test_metrics = evaluate_predictions(y_returns_test, test_preds)

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
    if model_type.lower() == 'lightgbm':
        # Сохраняем все 20 моделей
        for horizon in range(1, 21):
            if horizon in model.models:
                joblib.dump(model.models[horizon], exp_dir / f'model_return_{horizon}d.pkl')
        print(f"   OK Saved {len(model.models)} models (model_return_1d.pkl through model_return_20d.pkl)")
    else:
        joblib.dump(model, exp_dir / 'model.pkl')
        print(f"   OK Saved model.pkl")

    # 5.2 Сохранение конфигурации
    config = {
        'experiment': {
            'name': exp_name,
            'timestamp': timestamp,
            'model_type': model_type,
            'created_at': datetime.now().isoformat()
        },
        'model': model_params,
        'features': {
            'count': len(feature_cols),
            'columns': feature_cols,
            'windows': data_metadata['windows'],
            'cross_sectional': data_metadata['include_cross_sectional'],
            'interactions': data_metadata['include_interactions']
        },
        'data': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'train_period': data_metadata['train_period'],
            'val_period': data_metadata['val_period']
        },
        'results': {
            'train': {
                'mae_1d': float(train_metrics['mae_1d']),
                'mae_20d': float(train_metrics['mae_20d']),
                'mae_mean': float(train_metrics['mae_mean'])
            },
            'val': {
                'mae_1d': float(val_metrics['mae_1d']),
                'mae_20d': float(val_metrics['mae_20d']),
                'mae_mean': float(val_metrics['mae_mean'])
            },
            'test': {
                'mae_1d': float(test_metrics['mae_1d']),
                'mae_20d': float(test_metrics['mae_20d']),
                'mae_mean': float(test_metrics['mae_mean'])
            }
        }
    }

    import yaml
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"   OK Saved config.yaml")

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
        'train': convert_numpy(train_metrics),
        'val': convert_numpy(val_metrics),
        'test': convert_numpy(test_metrics)
    }

    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=2)

    print(f"   OK Saved metrics.json")

    # 5.4 Сохранение feature importance (для LightGBM)
    if model_type.lower() == 'lightgbm':
        importance_df = model.get_feature_importance()
        importance_df.to_csv(exp_dir / 'feature_importance.csv', index=False)
        print(f"   OK Saved feature_importance.csv")

    # 5.5 Сохранение предсказаний на val (все 20 горизонтов)
    predictions_df = val_df[['ticker', 'begin']].copy()

    # Добавляем предсказания для всех горизонтов
    for horizon in range(1, 21):
        pred_key = f'pred_return_{horizon}d'
        target_key = f'target_return_{horizon}d'
        if pred_key in val_preds:
            predictions_df[pred_key] = val_preds[pred_key]
        if target_key in y_returns_val:
            predictions_df[target_key] = y_returns_val[target_key]

    predictions_df.to_csv(exp_dir / 'predictions_val.csv', index=False)
    print(f"   OK Saved predictions_val.csv")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80 + "\n")

    print(f" Experiment: {exp_name}")
    print(f"   Output dir: {exp_dir}")
    print(f"\n   Validation metrics:")
    print(f"      MAE 1d:   {val_metrics['mae_1d']:.6f}")
    print(f"      MAE 20d:  {val_metrics['mae_20d']:.6f}")
    print(f"      MAE mean: {val_metrics['mae_mean']:.6f}")

    print(f"\n Next steps:")
    print(f"   # Evaluate model")
    print(f"   python scripts/3_evaluate.py --exp-dir {exp_dir.name}")
    print(f"\n   # Generate submission files")
    print(f"   python scripts/4_generate_submission.py --run-id {exp_dir.name}")
    print(f"\n   # Collect all experiments")
    print(f"   python scripts/collect_experiments.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model and save results')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--model-type', type=str, default='lightgbm',
                        choices=['lightgbm', 'momentum'],
                        help='Model type (default: lightgbm)')

    # LightGBM parameters
    parser.add_argument('--n-estimators', type=int, default=500,
                        help='Number of trees (default: 500)')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='Learning rate (default: 0.05)')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Max depth (default: 6)')
    parser.add_argument('--num-leaves', type=int, default=31,
                        help='Number of leaves (default: 31)')
    parser.add_argument('--min-child-samples', type=int, default=20,
                        help='Min child samples (default: 20)')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='Subsample ratio (default: 0.8)')
    parser.add_argument('--colsample-bytree', type=float, default=0.8,
                        help='Feature subsample ratio (default: 0.8)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state (default: 42)')

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
        random_state=args.random_state
    )
