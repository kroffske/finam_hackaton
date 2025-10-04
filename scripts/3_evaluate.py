"""
Model Evaluation Script

Этот скрипт:
1. Загружает сохраненную модель из outputs/<exp_name>/
2. Запускает оценку на test/val/custom данных
3. Выводит метрики и confusion matrix
4. Сохраняет отчет

Usage:
    python scripts/3_evaluate.py --exp-dir outputs/2025-10-03_14-30-00_lgbm_basic
    python scripts/3_evaluate.py --exp-dir outputs/2025-10-03_14-30-00_lgbm_basic --data test
    python scripts/3_evaluate.py --exp-dir outputs/2025-10-03_14-30-00_lgbm_basic --data val --save-report
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import yaml

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import joblib

from finam.metrics import evaluate_predictions, print_metrics


def evaluate_model(
    exp_dir: str,
    data_split: str = 'test',
    save_report: bool = False
):
    """
    Оценка сохраненной модели

    Args:
        exp_dir: путь к директории эксперимента (outputs/<exp_name>/)
        data_split: на каких данных оценивать ('train', 'val', 'test')
        save_report: сохранить отчет в файл
    """
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80 + "\n")

    exp_path = project_root / exp_dir
    if not exp_path.exists():
        # Попробуем найти в outputs/
        exp_path = project_root / 'outputs' / exp_dir
        if not exp_path.exists():
            print(f"ERROR Experiment not found: {exp_dir}")
            print(f"   Tried: {exp_path}")
            return

    print(f"Experiment dir: {exp_path}")
    print(f"Data split: {data_split}\n")

    # ========================================================================
    # 1. Загрузка конфигурации
    # ========================================================================
    print("[1/4] Loading experiment config...")

    config_path = exp_path / 'config.yaml'
    if not config_path.exists():
        print(f"ERROR Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_name = config['experiment']['name']
    model_type = config['experiment']['model_type']
    feature_cols = config['features']['columns']

    print(f"   OK Experiment: {exp_name}")
    print(f"   OK Model type: {model_type}")
    print(f"   OK Features: {len(feature_cols)}\n")

    # ========================================================================
    # 2. Загрузка модели
    # ========================================================================
    print("[2/4] Loading model...")

    if model_type.lower() == 'lightgbm':
        # Загружаем 2 regression модели
        model_return_1d = joblib.load(exp_path / 'model_return_1d.pkl')
        model_return_20d = joblib.load(exp_path / 'model_return_20d.pkl')

        print("   OK Loaded LightGBM models")

    elif model_type.lower() == 'momentum':
        model = joblib.load(exp_path / 'model.pkl')
        print("   OK Loaded Momentum model")

    else:
        print(f"ERROR Unknown model type: {model_type}")
        return

    # ========================================================================
    # 3. Загрузка данных
    # ========================================================================
    print(f"\n[3/4] Loading {data_split} data...")

    preprocessed_dir = project_root / 'data' / 'preprocessed'
    data_file = preprocessed_dir / f'{data_split}.csv'

    if not data_file.exists():
        print(f"ERROR Data file not found: {data_file}")
        print("   Run first: python scripts/1_prepare_data.py")
        return

    df = pd.read_csv(data_file, parse_dates=['begin'])

    print(f"   OK Loaded {len(df)} rows from {data_split}.csv")

    # Подготовка данных
    X = df[feature_cols]
    y_return_1d = df['target_return_1d'].values
    y_return_20d = df['target_return_20d'].values

    # ========================================================================
    # 4. Предсказание и оценка
    # ========================================================================
    print(f"\n[4/4] Evaluating on {data_split} data...")

    # Предсказания
    if model_type.lower() == 'lightgbm':
        X_filled = X.fillna(0)

        pred_return_1d = model_return_1d.predict(X_filled)
        pred_return_20d = model_return_20d.predict(X_filled)

        pred_return_1d = np.clip(pred_return_1d, -0.5, 0.5)
        pred_return_20d = np.clip(pred_return_20d, -1.0, 1.0)

    elif model_type.lower() == 'momentum':
        preds = model.predict(X)
        pred_return_1d = preds['pred_return_1d']
        pred_return_20d = preds['pred_return_20d']

    # Оценка
    metrics = evaluate_predictions(
        y_return_1d,
        y_return_20d,
        pred_return_1d,
        pred_return_20d
    )

    # ========================================================================
    # Вывод результатов
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS ({data_split.upper()})")
    print("=" * 80 + "\n")

    print_metrics(metrics, model_name=f"{exp_name} ({data_split})")

    # ========================================================================
    # Сохранение отчета (опционально)
    # ========================================================================
    if save_report:
        print("=" * 80)
        print("SAVING REPORT")
        print("=" * 80 + "\n")

        report_path = exp_path / f'evaluation_report_{data_split}.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"EVALUATION REPORT ({data_split.upper()})\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Model type: {model_type}\n")
            f.write(f"Data split: {data_split}\n")
            f.write(f"Evaluated at: {datetime.now().isoformat()}\n\n")

            f.write("METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write("1-DAY:\n")
            f.write(f"  MAE:   {metrics['mae_1d']:.6f}\n\n")

            f.write("20-DAY:\n")
            f.write(f"  MAE:   {metrics['mae_20d']:.6f}\n\n")

            f.write(f"MEAN MAE: {metrics['mae_mean']:.6f}\n")

        print(f"   OK Saved report: {report_path}\n")

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80 + "\n")

    print(f" Results ({data_split}):")
    print(f"   MAE 1d:   {metrics['mae_1d']:.6f}")
    print(f"   MAE 20d:  {metrics['mae_20d']:.6f}")
    print(f"   MAE mean: {metrics['mae_mean']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate saved model')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment directory (e.g., outputs/2025-10-03_14-30-00_lgbm_basic or just 2025-10-03_14-30-00_lgbm_basic)')
    parser.add_argument('--data', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate on (default: test)')
    parser.add_argument('--save-report', action='store_true',
                        help='Save evaluation report to file')

    args = parser.parse_args()

    evaluate_model(
        exp_dir=args.exp_dir,
        data_split=args.data,
        save_report=args.save_report
    )
