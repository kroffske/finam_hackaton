"""
Train Baseline (Momentum) Model

Обучает Momentum Baseline и сохраняет как эксперимент в outputs/.
Это позволяет сравнивать его с другими моделями.

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --exp-name momentum_baseline
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import json
import yaml

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import joblib

from finam.model import MomentumBaseline
from finam.metrics import evaluate_predictions, print_metrics


def train_baseline(exp_name: str = 'baseline'):
    """
    Обучить Momentum Baseline и сохранить как эксперимент

    Args:
        exp_name: название эксперимента
    """
    print("=" * 80)
    print("TRAIN BASELINE (MOMENTUM) MODEL")
    print("=" * 80 + "\n")

    # ========================================================================
    # 1. Загрузка данных
    # ========================================================================
    print("[1/5] Loading preprocessed data...")

    preprocessed_dir = project_root / 'data' / 'preprocessed'

    if not (preprocessed_dir / 'train.csv').exists():
        print(f"ERROR Preprocessed data not found!")
        print(f"   Run first: python scripts/1_prepare_data.py")
        return

    train_df = pd.read_csv(preprocessed_dir / 'train.csv', parse_dates=['begin'])
    val_df = pd.read_csv(preprocessed_dir / 'val.csv', parse_dates=['begin'])
    test_df = pd.read_csv(preprocessed_dir / 'test.csv', parse_dates=['begin'])

    # Загружаем metadata
    with open(preprocessed_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    feature_cols = metadata['feature_columns']

    print(f"   OK Train: {len(train_df)} rows")
    print(f"   OK Val:   {len(val_df)} rows")
    print(f"   OK Test:  {len(test_df)} rows")
    print(f"   OK Features: {len(feature_cols)}\n")

    # ========================================================================
    # 2. Подготовка данных
    # ========================================================================
    print("[2/5] Preparing features and targets...")

    # Train
    X_train = train_df[feature_cols].fillna(0)
    y_return_1d_train = train_df['target_return_1d'].values
    y_return_20d_train = train_df['target_return_20d'].values
    y_dir_1d_train = train_df['target_direction_1d'].values
    y_dir_20d_train = train_df['target_direction_20d'].values

    # Val
    X_val = val_df[feature_cols].fillna(0)
    y_return_1d_val = val_df['target_return_1d'].values
    y_return_20d_val = val_df['target_return_20d'].values

    # Test
    X_test = test_df[feature_cols].fillna(0)
    y_return_1d_test = test_df['target_return_1d'].values
    y_return_20d_test = test_df['target_return_20d'].values

    print(f"   OK X_train shape: {X_train.shape}")
    print(f"   OK X_val shape:   {X_val.shape}")
    print(f"   OK X_test shape:  {X_test.shape}\n")

    # ========================================================================
    # 3. Создание и обучение модели
    # ========================================================================
    print("[3/5] Training MOMENTUM BASELINE model...")

    model = MomentumBaseline(
        window_size=5,
        scaling_1d=0.3,
        scaling_20d=1.0,
        sensitivity_1d=10.0,
        sensitivity_20d=5.0
    )

    # Momentum не требует обучения, но вызовем для консистентности
    model.fit(
        X_train,
        y_return_1d_train,
        y_return_20d_train,
        y_dir_1d_train,
        y_dir_20d_train
    )

    print(f"   OK Training complete!\n")

    # ========================================================================
    # 4. Оценка на всех splits
    # ========================================================================
    print("[4/5] Evaluating on train, val, and test...")

    all_metrics = {}

    for split_name, X, y_1d, y_20d in [
        ('train', X_train, y_return_1d_train, y_return_20d_train),
        ('val', X_val, y_return_1d_val, y_return_20d_val),
        ('test', X_test, y_return_1d_test, y_return_20d_test)
    ]:
        # Предсказания
        preds = model.predict(X)

        # Оценка
        metrics = evaluate_predictions(
            y_1d,
            y_20d,
            preds['pred_return_1d'],
            preds['pred_return_20d'],
            preds['pred_prob_up_1d'],
            preds['pred_prob_up_20d']
        )

        all_metrics[split_name] = metrics

        # Вывод
        print(f"\n{'='*70}")
        print(f"[METRICS] {split_name.upper()}")
        print(f"{'='*70}")

        print(f"\n1-DAY METRICS:")
        print(f"  MAE:        {metrics['mae_1d']:.6f}")
        print(f"  Brier:      {metrics['brier_1d']:.6f}")
        print(f"  DA:         {metrics['da_1d']:.4f} ({metrics['da_1d']*100:.2f}%)")

        print(f"\n20-DAY METRICS:")
        print(f"  MAE:        {metrics['mae_20d']:.6f}")
        print(f"  Brier:      {metrics['brier_20d']:.6f}")
        print(f"  DA:         {metrics['da_20d']:.4f} ({metrics['da_20d']*100:.2f}%)")

        print(f"{'='*70}\n")

    # ========================================================================
    # 5. Сохранение эксперимента
    # ========================================================================
    print("[5/5] Saving experiment...")

    # Создаем директорию для эксперимента
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir_name = f"{timestamp}_{exp_name}"
    exp_dir = project_root / 'outputs' / exp_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем модель
    model_path = exp_dir / 'model.pkl'
    joblib.dump(model, model_path)
    print(f"   OK Saved model: {model_path.name}")

    # Сохраняем config
    config = {
        'experiment': {
            'name': exp_name,
            'timestamp': timestamp,
            'model_type': 'momentum'
        },
        'model': {
            'window_size': model.window_size,
            'scaling_1d': model.scaling_1d,
            'scaling_20d': model.scaling_20d,
            'sensitivity_1d': model.sensitivity_1d,
            'sensitivity_20d': model.sensitivity_20d
        },
        'features': {
            'columns': feature_cols
        }
    }

    config_path = exp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"   OK Saved config: {config_path.name}")

    # Сохраняем метрики
    # Конвертируем numpy types в Python native
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    metrics_serializable = convert_numpy(all_metrics)

    metrics_path = exp_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"   OK Saved metrics: {metrics_path.name}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASELINE TRAINING COMPLETE!")
    print("=" * 80 + "\n")

    print(f" Experiment: {exp_name}")
    print(f"   Output dir: {exp_dir}")
    print()
    print(f"   Test metrics:")
    print(f"      MAE 1d:  {all_metrics['test']['mae_1d']:.6f}")
    print(f"      MAE 20d: {all_metrics['test']['mae_20d']:.6f}")
    print(f"      Brier 1d:  {all_metrics['test']['brier_1d']:.6f}")
    print(f"      Brier 20d: {all_metrics['test']['brier_20d']:.6f}")
    print(f"      DA 1d:  {all_metrics['test']['da_1d']:.4f} ({all_metrics['test']['da_1d']*100:.2f}%)")
    print(f"      DA 20d: {all_metrics['test']['da_20d']:.4f} ({all_metrics['test']['da_20d']*100:.2f}%)")

    print("\n Next steps:")
    print(f"   # Generate submission files")
    print(f"   python scripts/4_generate_submission.py --run-id {exp_dir.name}")
    print(f"\n   # Collect all experiments")
    print(f"   python scripts/collect_experiments.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Momentum Baseline model')
    parser.add_argument('--exp-name', type=str, default='baseline',
                        help='Experiment name (default: baseline)')

    args = parser.parse_args()

    train_baseline(exp_name=args.exp_name)
