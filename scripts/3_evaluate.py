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
import json
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
        # Загружаем 4 модели
        model_return_1d = joblib.load(exp_path / 'model_return_1d.pkl')
        model_return_20d = joblib.load(exp_path / 'model_return_20d.pkl')

        model_prob_up_1d = None
        model_prob_up_20d = None

        if (exp_path / 'model_prob_up_1d.pkl').exists():
            model_prob_up_1d = joblib.load(exp_path / 'model_prob_up_1d.pkl')
        if (exp_path / 'model_prob_up_20d.pkl').exists():
            model_prob_up_20d = joblib.load(exp_path / 'model_prob_up_20d.pkl')

        print(f"   OK Loaded LightGBM models")

    elif model_type.lower() == 'momentum':
        model = joblib.load(exp_path / 'model.pkl')
        print(f"   OK Loaded Momentum model")

    else:
        print(f"ERROR Unknown model type: {model_type}")
        return

    # ========================================================================
    # 3. Загрузка данных
    # ========================================================================
    print(f"\n[3/4] Loading {data_split} data...")

    preprocessed_dir = project_root / 'data' / 'preprocessed'
    data_file = preprocessed_dir / f'{data_split}.parquet'

    if not data_file.exists():
        print(f"ERROR Data file not found: {data_file}")
        print(f"   Run first: python scripts/1_prepare_data.py")
        return

    df = pd.read_parquet(data_file)

    print(f"   OK Loaded {len(df)} rows from {data_split}.parquet")

    # Подготовка данных
    X = df[feature_cols]
    y_return_1d = df['target_return_1d'].values
    y_return_20d = df['target_return_20d'].values
    y_direction_1d = df['target_direction_1d'].values
    y_direction_20d = df['target_direction_20d'].values

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

        if model_prob_up_1d is not None:
            pred_prob_up_1d = model_prob_up_1d.predict_proba(X_filled)[:, 1]
        else:
            pred_prob_up_1d = 1 / (1 + np.exp(-10 * pred_return_1d))

        if model_prob_up_20d is not None:
            pred_prob_up_20d = model_prob_up_20d.predict_proba(X_filled)[:, 1]
        else:
            pred_prob_up_20d = 1 / (1 + np.exp(-5 * pred_return_20d))

        pred_prob_up_1d = np.clip(pred_prob_up_1d, 0.01, 0.99)
        pred_prob_up_20d = np.clip(pred_prob_up_20d, 0.01, 0.99)

    elif model_type.lower() == 'momentum':
        preds = model.predict(X)
        pred_return_1d = preds['pred_return_1d']
        pred_return_20d = preds['pred_return_20d']
        pred_prob_up_1d = preds['pred_prob_up_1d']
        pred_prob_up_20d = preds['pred_prob_up_20d']

    # Оценка
    metrics = evaluate_predictions(
        y_return_1d,
        y_return_20d,
        pred_return_1d,
        pred_return_20d,
        pred_prob_up_1d,
        pred_prob_up_20d
    )

    # ========================================================================
    # Вывод результатов
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS ({data_split.upper()})")
    print("=" * 80 + "\n")

    print_metrics(metrics, model_name=f"{exp_name} ({data_split})")

    # Confusion Matrix для направления
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (Direction)")
    print("=" * 80 + "\n")

    # 1-day direction
    y_true_dir_1d = (y_return_1d > 0).astype(int)
    y_pred_dir_1d = (pred_return_1d > 0).astype(int)

    from sklearn.metrics import confusion_matrix, classification_report

    cm_1d = confusion_matrix(y_true_dir_1d, y_pred_dir_1d)
    print("1-DAY DIRECTION:")
    print(f"                Predicted")
    print(f"               Down   Up")
    print(f"Actual Down  {cm_1d[0, 0]:6d} {cm_1d[0, 1]:5d}")
    print(f"       Up    {cm_1d[1, 0]:6d} {cm_1d[1, 1]:5d}")
    print()

    # 20-day direction
    y_true_dir_20d = (y_return_20d > 0).astype(int)
    y_pred_dir_20d = (pred_return_20d > 0).astype(int)

    cm_20d = confusion_matrix(y_true_dir_20d, y_pred_dir_20d)
    print("20-DAY DIRECTION:")
    print(f"                Predicted")
    print(f"               Down   Up")
    print(f"Actual Down  {cm_20d[0, 0]:6d} {cm_20d[0, 1]:5d}")
    print(f"       Up    {cm_20d[1, 0]:6d} {cm_20d[1, 1]:5d}")
    print()

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
            f.write(f"1-DAY:\n")
            f.write(f"  MAE:   {metrics['mae_1d']:.6f}\n")
            f.write(f"  Brier: {metrics['brier_1d']:.6f}\n")
            f.write(f"  DA:    {metrics['da_1d']:.4f} ({metrics['da_1d']*100:.2f}%)\n\n")

            f.write(f"20-DAY:\n")
            f.write(f"  MAE:   {metrics['mae_20d']:.6f}\n")
            f.write(f"  Brier: {metrics['brier_20d']:.6f}\n")
            f.write(f"  DA:    {metrics['da_20d']:.4f} ({metrics['da_20d']*100:.2f}%)\n\n")

            if 'score_total' in metrics:
                f.write(f"SCORE TOTAL: {metrics['score_total']:.6f}\n\n")

            f.write("\nCONFUSION MATRIX (1-DAY):\n")
            f.write(f"                Predicted\n")
            f.write(f"               Down   Up\n")
            f.write(f"Actual Down  {cm_1d[0, 0]:6d} {cm_1d[0, 1]:5d}\n")
            f.write(f"       Up    {cm_1d[1, 0]:6d} {cm_1d[1, 1]:5d}\n\n")

            f.write("\nCONFUSION MATRIX (20-DAY):\n")
            f.write(f"                Predicted\n")
            f.write(f"               Down   Up\n")
            f.write(f"Actual Down  {cm_20d[0, 0]:6d} {cm_20d[0, 1]:5d}\n")
            f.write(f"       Up    {cm_20d[1, 0]:6d} {cm_20d[1, 1]:5d}\n")

        print(f"   OK Saved report: {report_path}\n")

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80 + "\n")

    print(f" Results ({data_split}):")
    print(f"   MAE 1d:  {metrics['mae_1d']:.6f}")
    print(f"   MAE 20d: {metrics['mae_20d']:.6f}")
    print(f"   Brier 1d:  {metrics['brier_1d']:.6f}")
    print(f"   Brier 20d: {metrics['brier_20d']:.6f}")
    print(f"   DA 1d:  {metrics['da_1d']:.4f} ({metrics['da_1d']*100:.2f}%)")
    print(f"   DA 20d: {metrics['da_20d']:.4f} ({metrics['da_20d']*100:.2f}%)")
    if 'score_total' in metrics:
        print(f"   Score Total: {metrics['score_total']:.6f}")


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
