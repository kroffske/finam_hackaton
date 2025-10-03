"""
Compute Baseline (Momentum) Metrics

Этот скрипт вычисляет метрики Momentum Baseline на всех splits (train/val/test)
и сохраняет их в data/baseline_metrics.json для последующего сравнения.

Usage:
    python scripts/compute_baseline_metrics.py
"""

import sys
from pathlib import Path
import json

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np

from finam.model import MomentumBaseline
from finam.metrics import evaluate_predictions


def compute_baseline_for_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str
) -> dict:
    """
    Вычислить baseline метрики для одного split

    Args:
        df: DataFrame с данными
        feature_cols: список feature колонок
        split_name: название split ('train', 'val', 'test')

    Returns:
        dict с метриками
    """
    print(f"\n[{split_name.upper()}] Computing Momentum Baseline metrics...")

    # Подготовка данных
    X = df[feature_cols].fillna(0)
    y_return_1d = df['target_return_1d'].values
    y_return_20d = df['target_return_20d'].values
    y_direction_1d = df['target_direction_1d'].values
    y_direction_20d = df['target_direction_20d'].values

    # Создаем и обучаем Momentum Baseline
    model = MomentumBaseline(
        window_size=5,
        scaling_1d=0.3,
        scaling_20d=1.0,
        sensitivity_1d=10.0,
        sensitivity_20d=5.0
    )

    # Для baseline не нужно обучение, но вызовем fit для консистентности
    model.fit(X, y_return_1d, y_return_20d, y_direction_1d, y_direction_20d)

    # Предсказания
    preds = model.predict(X)

    # Вычисление метрик
    metrics = evaluate_predictions(
        y_return_1d,
        y_return_20d,
        preds['pred_return_1d'],
        preds['pred_return_20d'],
        preds['pred_prob_up_1d'],
        preds['pred_prob_up_20d']
    )

    print(f"   MAE 1d:  {metrics['mae_1d']:.6f}")
    print(f"   MAE 20d: {metrics['mae_20d']:.6f}")
    print(f"   Brier 1d:  {metrics['brier_1d']:.6f}")
    print(f"   Brier 20d: {metrics['brier_20d']:.6f}")
    print(f"   DA 1d:  {metrics['da_1d']:.4f}")
    print(f"   DA 20d: {metrics['da_20d']:.4f}")

    return metrics


def main():
    print("=" * 80)
    print("COMPUTE BASELINE METRICS (Momentum)")
    print("=" * 80 + "\n")

    # ========================================================================
    # 1. Загрузка preprocessed данных
    # ========================================================================
    print("[1/3] Loading preprocessed data...")

    preprocessed_dir = project_root / 'data' / 'preprocessed'

    if not (preprocessed_dir / 'train.parquet').exists():
        print(f"ERROR Preprocessed data not found!")
        print(f"   Run first: python scripts/1_prepare_data.py")
        return

    train_df = pd.read_parquet(preprocessed_dir / 'train.parquet')
    val_df = pd.read_parquet(preprocessed_dir / 'val.parquet')
    test_df = pd.read_parquet(preprocessed_dir / 'test.parquet')

    # Загружаем metadata для feature columns
    with open(preprocessed_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    feature_cols = metadata['feature_columns']

    print(f"   OK Train: {len(train_df)} rows")
    print(f"   OK Val:   {len(val_df)} rows")
    print(f"   OK Test:  {len(test_df)} rows")
    print(f"   OK Features: {len(feature_cols)}")

    # ========================================================================
    # 2. Вычисление baseline метрик для каждого split
    # ========================================================================
    print("\n[2/3] Computing baseline metrics for all splits...")

    baseline_metrics = {}

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        metrics = compute_baseline_for_split(split_df, feature_cols, split_name)
        baseline_metrics[split_name] = metrics

    # ========================================================================
    # 3. Сохранение результатов
    # ========================================================================
    print("\n[3/3] Saving baseline metrics...")

    output_path = project_root / 'data' / 'baseline_metrics.json'

    # Конвертируем numpy types в native Python для JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    baseline_metrics_serializable = convert_numpy_types(baseline_metrics)

    with open(output_path, 'w') as f:
        json.dump(baseline_metrics_serializable, f, indent=2)

    print(f"   OK Saved to: {output_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASELINE METRICS COMPUTED!")
    print("=" * 80 + "\n")

    print("Summary:")
    print(f"   Train MAE 1d:  {baseline_metrics['train']['mae_1d']:.6f}")
    print(f"   Val MAE 1d:    {baseline_metrics['val']['mae_1d']:.6f}")
    print(f"   Test MAE 1d:   {baseline_metrics['test']['mae_1d']:.6f}")
    print()
    print(f"   Train MAE 20d: {baseline_metrics['train']['mae_20d']:.6f}")
    print(f"   Val MAE 20d:   {baseline_metrics['val']['mae_20d']:.6f}")
    print(f"   Test MAE 20d:  {baseline_metrics['test']['mae_20d']:.6f}")
    print()
    print(f"Saved to: {output_path}")
    print("\nNext steps:")
    print("   python scripts/3_evaluate.py --exp-dir <your_experiment> --data test")


if __name__ == "__main__":
    main()
