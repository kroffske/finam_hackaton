"""
Generate Submission Script

Этот скрипт:
1. Загружает обученную модель из outputs/<run_id>/
2. Загружает preprocessed public_test.parquet и private_test.parquet
3. Генерирует предсказания (pred_return_1d/20d, pred_prob_up_1d/20d)
4. Сохраняет submission файлы в outputs/<run_id>/

Usage:
    python scripts/4_generate_submission.py --run-id 2025-10-03_23-41-15_lgbm_with_news
    python scripts/4_generate_submission.py --run-id <run_id> --output-dir submissions/
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import joblib
import yaml


def generate_submission(
    run_id: str,
    output_dir: str = None,
    preprocessed_dir: str = None,
    latest: bool = False
):
    """
    Генерация submission файлов для public и private тестов

    Args:
        run_id: идентификатор эксперимента (например 2025-10-03_23-41-15_lgbm_with_news)
        output_dir: директория для сохранения submission (по умолчанию outputs/<run_id>/)
        preprocessed_dir: директория с preprocessed данными (по умолчанию data/preprocessed/)
        latest: если True, генерировать одно предсказание на последнюю дату для каждого тикера
    """
    print("=" * 80)
    print("GENERATE SUBMISSION")
    print("=" * 80 + "\n")

    # ========================================================================
    # 1. Проверка путей
    # ========================================================================
    print(f"[1/5] Loading experiment configuration...")

    exp_dir = project_root / 'outputs' / run_id

    if not exp_dir.exists():
        print(f"   ERROR: Experiment directory not found: {exp_dir}")
        print(f"\n   Available experiments:")
        outputs_dir = project_root / 'outputs'
        if outputs_dir.exists():
            for d in sorted(outputs_dir.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    print(f"      - {d.name}")
        return

    # Загружаем config
    config_path = exp_dir / 'config.yaml'
    if not config_path.exists():
        print(f"   ERROR: Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"   OK Loaded config from {config_path}")
    print(f"   Experiment: {config['experiment']['name']}")
    print(f"   Model type: {config['experiment']['model_type']}")

    # ========================================================================
    # 2. Загрузка модели
    # ========================================================================
    print(f"\n[2/5] Loading trained model...")

    # Ищем модель (может быть model_1d.pkl, model_20d.pkl или model.pkl)
    model_files = list(exp_dir.glob('model*.pkl'))

    if not model_files:
        print(f"   ERROR: No model files found in {exp_dir}")
        return

    # Загружаем модели
    models = {}
    for model_file in model_files:
        model_name = model_file.stem  # model_1d, model_20d, или model
        models[model_name] = joblib.load(model_file)
        print(f"   OK Loaded {model_file.name}")

    # ========================================================================
    # 3. Загрузка preprocessed данных
    # ========================================================================
    print(f"\n[3/5] Loading preprocessed test data...")

    if preprocessed_dir is None:
        preprocessed_dir = project_root / 'data' / 'preprocessed'
    else:
        preprocessed_dir = Path(preprocessed_dir)

    # Загружаем metadata для получения списка признаков
    metadata_path = preprocessed_dir / 'metadata.json'
    if not metadata_path.exists():
        print(f"   ERROR: metadata.json not found: {metadata_path}")
        return

    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    feature_cols = metadata['feature_columns']
    print(f"   OK Loaded {len(feature_cols)} features from metadata.json")

    # Загружаем holdout_test
    holdout_test_path = preprocessed_dir / 'holdout_test.csv'

    if not holdout_test_path.exists():
        print(f"\n   ERROR: holdout_test.csv not found!")
        print(f"   Run: python scripts/1_prepare_data.py")
        return

    holdout_test_df = pd.read_csv(holdout_test_path, parse_dates=['begin'])
    print(f"   OK Loaded holdout_test.csv: {len(holdout_test_df)} rows")

    # ========================================================================
    # 4. Генерация предсказаний
    # ========================================================================
    print(f"\n[4/5] Generating predictions...")

    # Определяем output директорию
    if output_dir is None:
        submission_dir = exp_dir
    else:
        submission_dir = Path(output_dir)
        submission_dir.mkdir(parents=True, exist_ok=True)

    # Подготовка данных
    print(f"\n   Processing holdout_test...")
    X_test = holdout_test_df[feature_cols].fillna(0)

    # Проверяем структуру моделей и делаем предсказания
    if 'model' in models:
        # Единая модель (Momentum или другая) - использует .predict()
        model = models['model']
        preds = model.predict(X_test)
    else:
        # Множество моделей LightGBM (model_return_1d.pkl, model_return_2d.pkl, ...)
        # Собираем предсказания от всех 20 моделей
        preds = {}
        for horizon in range(1, 21):
            model_key = f'model_return_{horizon}d'
            if model_key in models:
                pred = models[model_key].predict(X_test)
                # Clipping пропорционально горизонту
                max_return = 0.5 + (1.0 - 0.5) * (horizon - 1) / 19
                pred = np.clip(pred, -max_return, max_return)
                preds[f'pred_return_{horizon}d'] = pred

    # Создаем submission DataFrame с 20 предсказаниями
    submission_data = {
        'ticker': holdout_test_df['ticker'],
        'begin': holdout_test_df['begin']
    }

    # Добавляем все 20 предсказаний
    for horizon in range(1, 21):
        pred_key = f'pred_return_{horizon}d'
        if pred_key in preds:
            submission_data[pred_key] = preds[pred_key]
        else:
            # Если модели нет, заполняем нулями
            print(f"      WARNING: {pred_key} not found, filling with zeros")
            submission_data[pred_key] = np.zeros(len(holdout_test_df))

    submission = pd.DataFrame(submission_data)

    # Фильтрация на последнюю дату для каждого тикера, если флаг --latest
    if latest:
        print(f"      Filtering to latest date per ticker...")
        submission = submission.sort_values('begin').groupby('ticker').tail(1).reset_index(drop=True)
        print(f"      Filtered to {len(submission)} rows (one per ticker)")

    print(f"      Generated {len(submission)} predictions with {len(preds)} horizons")
    print(f"      Sample stats:")
    print(f"         pred_return_1d:  mean={submission['pred_return_1d'].mean():.6f}, std={submission['pred_return_1d'].std():.6f}")
    print(f"         pred_return_10d: mean={submission['pred_return_10d'].mean():.6f}, std={submission['pred_return_10d'].std():.6f}")
    print(f"         pred_return_20d: mean={submission['pred_return_20d'].mean():.6f}, std={submission['pred_return_20d'].std():.6f}")

    # ========================================================================
    # 5. Сохранение submission файлов
    # ========================================================================
    print(f"\n[5/5] Saving submission file...")

    submission_path = submission_dir / 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   OK Saved: {submission_path}")
    print(f"      Rows: {len(submission)}")
    print(f"      Columns: {len(submission.columns)} (ticker, begin, pred_return_1d through pred_return_20d)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUBMISSION GENERATION COMPLETE!")
    print("=" * 80 + "\n")

    print("Summary:")
    print(f"   Experiment: {run_id}")
    print(f"   Holdout submission: {len(submission)} rows")
    print(f"   Predictions: 20 horizons (1-20 days)")

    print(f"\n   Saved to: {submission_dir}/")

    print("\nNext steps:")
    print(f"   # View submission predictions")
    print(f"   head {submission_path}")

    print(f"\n   # Submit file to the competition platform")
    print(f"   # (Upload submission.csv)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate submission files from trained model')
    parser.add_argument('--run-id', type=str, required=True,
                        help='Experiment run ID (e.g., 2025-10-03_23-41-15_lgbm_with_news)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for submission files (default: outputs/<run_id>/)')
    parser.add_argument('--preprocessed-dir', type=str, default=None,
                        help='Directory with preprocessed data (default: data/preprocessed/)')
    parser.add_argument('--latest', action='store_true',
                        help='Generate only one prediction per ticker (last date)')

    args = parser.parse_args()

    generate_submission(
        run_id=args.run_id,
        output_dir=args.output_dir,
        preprocessed_dir=args.preprocessed_dir,
        latest=args.latest
    )
